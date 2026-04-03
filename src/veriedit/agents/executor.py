from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl, save_image
from veriedit.metrics.iq_metrics import summarize_image_quality
from veriedit.metrics.similarity import compare_images
from veriedit.observability import record_agent_handoff, record_node_event, record_tool_event
from veriedit.schemas import AgentLog, ExecutionRecord, WorkflowState
from veriedit.tools import build_tool_registry
from veriedit.tools.base import sanitize_numeric_params


class ExecutorAgent:
    def __init__(self) -> None:
        self.registry = build_tool_registry()

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="execute_plan", phase="start")
        current_image, metadata = load_image(state["current_image_path"])
        reference_image = load_image(state["reference_image_path"])[0] if state["reference_image_path"] else None
        plan = state["plan"] or {}
        step_records: list[dict] = []
        intermediate_paths = list(state["intermediate_paths"])
        save_intermediates = bool(state["request"].get("save_intermediates", True))
        allowed_tools = set(state["request"].get("allowed_tools") or self.registry.names())
        latest_output_path = state["current_image_path"]
        mask_cache = _load_execution_masks(state)
        for index, raw_step in enumerate(plan.get("steps", []), start=1):
            if raw_step["tool"] not in allowed_tools:
                output_path = Path(state["run_dir"]) / f"step_{state['iteration']:02d}_{index:02d}_{raw_step['tool']}_blocked.png"
                record = ExecutionRecord(
                    step_index=index,
                    tool=raw_step["tool"],
                    params=raw_step.get("params", {}),
                    execution_mode="blocked",
                    mask_name=None,
                    mask_coverage=0.0,
                    before_metrics={},
                    after_metrics={},
                    output_path=str(output_path),
                    status="failed",
                    notes=["Step skipped because the tool is not in the allowed-tools list."],
                )
                step_records.append(record.model_dump())
                continue
            spec = self.registry.get(raw_step["tool"])
            params = sanitize_numeric_params(raw_step.get("params", {}), spec.parameter_bounds)
            before_metrics = summarize_image_quality(current_image, metadata)
            execution_context = _execution_context(raw_step["tool"], params, mask_cache, current_image.shape[:2])
            candidate, operation_notes, after_metrics, comparison, variant_label = _choose_candidate(
                spec,
                current_image,
                params,
                reference_image,
                execution_context,
            )
            status = "ok"
            notes: list[str] = []
            if _step_is_harmful(raw_step["tool"], before_metrics, after_metrics, comparison, execution_context):
                candidate = current_image
                after_metrics = before_metrics
                status = "rolled_back"
                notes.append("Step exceeded safety heuristics and was rolled back.")
            else:
                notes.append("Step applied successfully.")
            notes.append(f"Variant selected: {variant_label}")
            notes.append(
                f"Execution mode: {execution_context['mode']} mask={execution_context['mask_name'] or 'none'} coverage={execution_context['mask_coverage']:.4f}"
            )
            if operation_notes:
                notes.append(f"Tool details: {operation_notes}")
            if save_intermediates:
                output_path = Path(state["run_dir"]) / f"step_{state['iteration']:02d}_{index:02d}_{raw_step['tool']}.png"
            else:
                output_path = Path(state["run_dir"]) / "current.png"
            save_image(candidate, output_path)
            current_image = candidate
            latest_output_path = str(output_path)
            metadata = {
                "width": current_image.shape[1],
                "height": current_image.shape[0],
                "mode": "RGB",
                "bit_depth": 8,
            }
            if save_intermediates:
                intermediate_paths.append(str(output_path))
            record = ExecutionRecord(
                step_index=index,
                tool=raw_step["tool"],
                params=params,
                execution_mode=execution_context["mode"],
                mask_name=execution_context["mask_name"],
                mask_coverage=execution_context["mask_coverage"],
                before_metrics=_slim_metrics(before_metrics, comparison=False),
                after_metrics={**_slim_metrics(after_metrics, comparison=False), **comparison},
                output_path=str(output_path),
                status=status,
                notes=notes,
            )
            step_records.append(record.model_dump())
            record_tool_event(
                state,
                tool=raw_step["tool"],
                params=params,
                variant=variant_label,
                status=status,
                metrics={
                    "change_area_ratio": comparison["change_area_ratio"],
                    "ssim": comparison["ssim"],
                    "mask_coverage": execution_context["mask_coverage"],
                    "preserved_region_change_ratio": comparison.get("preserved_region_change_ratio", 0.0),
                },
            )
        state["executed_steps"].extend(step_records)
        state["intermediate_paths"] = intermediate_paths
        state["current_image_path"] = latest_output_path
        record_agent_handoff(
            state,
            from_agent="executor",
            to_agent="reviewer",
            summary=f"Executed {len(step_records)} planned step(s) and updated the working image.",
            key_points=[
                f"rolled_back={sum(1 for step in step_records if step['status'] == 'rolled_back')}",
                f"ok={sum(1 for step in step_records if step['status'] == 'ok')}",
                f"latest_image={Path(latest_output_path).name}",
            ],
            payload={"executed_steps": step_records[-6:], "current_image_path": latest_output_path},
        )
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="executor",
                iteration=state["iteration"],
                input_summary={"step_count": len(plan.get("steps", []))},
                decision="executed",
                output_summary={"applied_steps": len(step_records), "latest_image": state["current_image_path"]},
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        record_node_event(state, node="execute_plan", phase="end", summary={"applied_steps": len(step_records)})
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _step_is_harmful(
    tool_name: str,
    before: dict,
    after: dict,
    comparison: dict[str, float],
    execution_context: dict[str, object],
) -> bool:
    mode = str(execution_context.get("mode", "global"))
    if comparison["change_area_ratio"] > 0.8:
        return True
    if mode == "masked_local_repair" and comparison.get("preserved_region_change_ratio", 0.0) > 0.08:
        return True
    if "denoise" in tool_name and after["blur_score"] < before["blur_score"] * 0.45:
        return True
    if "sharpen" in tool_name or tool_name == "unsharp_mask":
        if after["noise_score"] > before["noise_score"] + 0.08:
            return True
    if after["clipping_highlights"] > before["clipping_highlights"] + 0.08:
        return True
    if after["clipping_shadows"] > before["clipping_shadows"] + 0.08:
        return True
    return False


def _choose_candidate(spec, current_image, params, reference_image, execution_context):
    variants = [("base", params)]
    softened = _soften_params(params)
    if softened != params and spec.name in {
        "shadow_highlight_balance",
        "clahe_contrast",
        "non_local_means_denoise",
        "bilateral_denoise",
        "small_defect_heal",
        "unsharp_mask",
        "stroke_paint",
        "clone_stamp",
    }:
        variants.append(("softened", softened))

    best = None
    for label, variant_params in variants:
        base_candidate, operation_notes = spec.operation(current_image, variant_params, reference_image)
        candidate = _apply_execution_context(current_image, base_candidate, execution_context)
        after_metrics = summarize_image_quality(
            candidate,
            {
                "width": candidate.shape[1],
                "height": candidate.shape[0],
                "mode": "RGB",
                "bit_depth": 8,
            },
        )
        comparison = compare_images(current_image, candidate)
        comparison.update(_region_change_metrics(current_image, candidate, execution_context))
        score = _candidate_score(spec.name, after_metrics, comparison, execution_context)
        payload = (candidate, operation_notes, after_metrics, comparison, label, score)
        if best is None or score < best[-1]:
            best = payload
    assert best is not None
    return best[0], best[1], best[2], best[3], best[4]


def _soften_params(params: dict) -> dict:
    softened = {}
    for key, value in params.items():
        if isinstance(value, bool):
            softened[key] = value
        elif isinstance(value, int):
            softened[key] = max(1, int(round(value * 0.75)))
        elif isinstance(value, float):
            softened[key] = round(value * 0.75, 4)
        else:
            softened[key] = value
    return softened


def _candidate_score(tool_name: str, metrics: dict, comparison: dict[str, float], execution_context: dict[str, object]) -> float:
    score = 0.0
    score += comparison["change_area_ratio"] * 1.2
    if str(execution_context.get("mode")) == "masked_local_repair":
        score += comparison.get("preserved_region_change_ratio", 0.0) * 4.0
        score -= comparison.get("target_region_change_ratio", 0.0) * 0.8
    score += max(0.0, metrics["clipping_highlights"] - 0.01) * 5.0
    score += max(0.0, metrics["clipping_shadows"] - 0.01) * 5.0
    if "denoise" in tool_name:
        score += metrics["noise_score"] * 2.0
        score -= min(1.5, metrics["blur_score"] / 1000.0)
    elif "sharpen" in tool_name:
        score += metrics["noise_score"] * 1.5
        score -= min(1.5, metrics["blur_score"] / 1200.0)
    elif tool_name in {"shadow_highlight_balance", "clahe_contrast", "histogram_balance"}:
        score += max(0.0, 0.55 - metrics["contrast_score"])
        score += metrics["fade_score"] * 0.7
    elif tool_name in {"dust_cleanup", "scratch_candidate_cleanup", "small_defect_heal"}:
        score += metrics["dust_candidates"] / 3000.0
        score += metrics["scratch_candidates"] / 1500.0
    elif tool_name == "stroke_paint":
        score += comparison.get("preserved_region_change_ratio", 0.0) * 5.0
        score -= comparison.get("target_region_change_ratio", 0.0) * 0.9
    elif tool_name == "clone_stamp":
        score += comparison.get("preserved_region_change_ratio", 0.0) * 5.5
        score -= comparison.get("target_region_change_ratio", 0.0) * 1.0
    return score


def _slim_metrics(metrics: dict, comparison: bool) -> dict[str, float]:
    selected = {
        "blur_score": float(metrics["blur_score"]),
        "noise_score": float(metrics["noise_score"]),
        "yellow_cast": float(metrics["yellow_cast"]),
        "contrast_score": float(metrics["contrast_score"]),
        "clipping_highlights": float(metrics["clipping_highlights"]),
        "clipping_shadows": float(metrics["clipping_shadows"]),
        "dust_candidates": float(metrics["dust_candidates"]),
        "scratch_candidates": float(metrics["scratch_candidates"]),
        "fade_score": float(metrics["fade_score"]),
        "edge_damage_ratio": float(metrics["edge_damage_ratio"]),
    }
    if comparison:
        return {**selected}
    return selected


def _load_execution_masks(state: WorkflowState) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, path in (state.get("diagnostic_artifacts") or {}).items():
        if not path.endswith(".png"):
            continue
        try:
            mask_image, _ = load_image(path)
        except Exception:
            continue
        masks[name] = mask_image[..., 0] > 127
    return masks


def _execution_context(
    tool_name: str,
    params: dict[str, object],
    mask_cache: dict[str, np.ndarray],
    image_shape: tuple[int, int],
) -> dict[str, object]:
    if tool_name == "stroke_paint":
        boxes = params.get("mask_boxes", [])
        mask = _mask_from_boxes(image_shape, boxes)
        if mask.any():
            return {
                "mode": "masked_local_repair",
                "mask_name": "stroke_roi",
                "mask": mask,
                "mask_coverage": float(np.mean(mask)),
            }
        return {"mode": "global", "mask_name": None, "mask": None, "mask_coverage": 0.0}
    if tool_name == "clone_stamp":
        mask = _clone_mask_from_params(image_shape, params)
        if mask.any():
            return {
                "mode": "masked_local_repair",
                "mask_name": "clone_roi",
                "mask": mask,
                "mask_coverage": float(np.mean(mask)),
            }
        return {"mode": "global", "mask_name": None, "mask": None, "mask_coverage": 0.0}
    local_mask_name = {
        "dust_cleanup": "dust_mask",
        "scratch_candidate_cleanup": "scratch_mask",
        "small_defect_heal": "defect_union",
    }.get(tool_name)
    if not local_mask_name:
        return {"mode": "global", "mask_name": None, "mask": None, "mask_coverage": 0.0}
    mask = mask_cache.get(local_mask_name)
    if mask is None or not mask.any():
        return {"mode": "global", "mask_name": None, "mask": None, "mask_coverage": 0.0}
    return {
        "mode": "masked_local_repair",
        "mask_name": local_mask_name,
        "mask": mask,
        "mask_coverage": float(np.mean(mask)),
    }


def _mask_from_boxes(shape: tuple[int, int], boxes: object) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not isinstance(boxes, list):
        return mask
    for box in boxes:
        if not isinstance(box, dict):
            continue
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        width = int(box.get("width", 0))
        height = int(box.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(shape[1], x + width)
        y1 = min(shape[0], y + height)
        mask[y0:y1, x0:x1] = True
    return mask


def _clone_mask_from_params(shape: tuple[int, int], params: dict[str, object]) -> np.ndarray:
    region = params.get("target_region")
    if isinstance(region, dict):
        radius = int(params.get("radius", 8))
        expanded = {
            "x": int(region.get("x", 0)) - radius * 2,
            "y": int(region.get("y", 0)) - radius * 2,
            "width": int(region.get("width", 0)) + radius * 4,
            "height": int(region.get("height", 0)) + radius * 4,
        }
        return _mask_from_boxes(shape, [expanded])
    strokes = params.get("strokes", [])
    mask = np.zeros(shape, dtype=bool)
    radius = int(params.get("radius", 8))
    if not isinstance(strokes, list):
        return mask
    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        points = stroke.get("points", [])
        if not isinstance(points, list):
            continue
        xs = [int(point[0]) for point in points if isinstance(point, list) and len(point) == 2]
        ys = [int(point[1]) for point in points if isinstance(point, list) and len(point) == 2]
        if not xs or not ys:
            continue
        x = max(0, min(xs) - radius)
        y = max(0, min(ys) - radius)
        width = min(shape[1], max(xs) + radius + 1) - x
        height = min(shape[0], max(ys) + radius + 1) - y
        mask |= _mask_from_boxes(shape, [{"x": x, "y": y, "width": width, "height": height}])
    return mask


def _apply_execution_context(current_image: np.ndarray, candidate: np.ndarray, execution_context: dict[str, object]) -> np.ndarray:
    if execution_context.get("mode") != "masked_local_repair":
        return candidate
    mask = execution_context.get("mask")
    if not isinstance(mask, np.ndarray) or not mask.any():
        return candidate
    alpha = _feathered_alpha(mask)
    blended = current_image.astype(np.float32) * (1.0 - alpha) + candidate.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _feathered_alpha(mask: np.ndarray) -> np.ndarray:
    expanded = mask.astype(np.float32)
    try:
        from veriedit._compat import cv2

        if cv2 is not None:
            expanded = cv2.dilate(expanded, np.ones((3, 3), dtype=np.uint8), iterations=1)
            alpha = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigmaX=1.2)
            alpha = np.clip(alpha, 0.0, 1.0)
            return alpha[..., None]
    except Exception:
        pass
    return expanded[..., None]


def _region_change_metrics(current_image: np.ndarray, candidate: np.ndarray, execution_context: dict[str, object]) -> dict[str, float]:
    mask = execution_context.get("mask")
    if not isinstance(mask, np.ndarray) or not mask.any():
        return {"target_region_change_ratio": 0.0, "preserved_region_change_ratio": 0.0}
    delta = np.abs(current_image.astype(np.float32) - candidate.astype(np.float32)).mean(axis=2) > 10.0
    target_ratio = float(np.mean(delta[mask])) if mask.any() else 0.0
    preserved = ~mask
    preserved_ratio = float(np.mean(delta[preserved])) if preserved.any() else 0.0
    return {
        "target_region_change_ratio": target_ratio,
        "preserved_region_change_ratio": preserved_ratio,
    }
