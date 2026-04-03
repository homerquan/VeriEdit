from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl, save_image
from veriedit.observability import record_agent_handoff, record_node_event
from veriedit.schemas import AgentLog, PlanStep, WorkflowState
from veriedit.tools import build_tool_registry


class ToolTrialAgent:
    def __init__(self) -> None:
        self.registry = build_tool_registry()

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="trial_tools", phase="start")
        plan = state.get("plan") or {}
        allowed_tools = set(state["request"].get("allowed_tools") or self.registry.names())
        trial_budget = int(state["request"].get("max_tool_trials", 10))
        trial_record: dict[str, Any] = {
            "iteration": state["iteration"],
            "tool": "clone_stamp",
            "max_tool_trials": trial_budget,
            "attempted": False,
            "accepted": False,
            "reason": "",
            "selected_region": None,
            "selected_params": None,
            "trials": [],
        }

        if "clone_stamp" not in allowed_tools:
            trial_record["reason"] = "clone_stamp is not allowed for this run."
            state.setdefault("tool_trial_history", []).append(trial_record)
            self._handoff(state, trial_record)
            self._log(state, start, "skipped", {"reason": trial_record["reason"]})
            record_node_event(state, node="trial_tools", phase="end", summary={"accepted": False, "trials": 0})
            return state

        image, _ = load_image(state["current_image_path"])
        region_summary = (state.get("diagnostics") or {}).get("regions", {})
        recent_feedback = " ".join(_recent_feedback_lines(state)).lower()
        if not _should_run_clone_trials(state, region_summary, allowed_tools, recent_feedback):
            trial_record["reason"] = "No clone-worthy local damage was detected for this iteration."
            state.setdefault("tool_trial_history", []).append(trial_record)
            self._handoff(state, trial_record)
            self._log(state, start, "skipped", {"reason": trial_record["reason"]})
            record_node_event(state, node="trial_tools", phase="end", summary={"accepted": False, "trials": 0})
            return state

        trial_record["attempted"] = True
        trial_dir = Path(state["run_dir"]) / "tool_trials" / f"iteration_{state['iteration']:02d}"
        prior_regions = _previous_clone_regions(state)
        regions = _clone_candidate_regions(image, region_summary, prior_regions)
        clone_spec = self.registry.get("clone_stamp")
        best_trial: dict[str, Any] | None = None
        trial_index = 0

        for region in regions:
            for radius in _radius_candidates(region):
                source_points = _source_candidates_for_region(image, region, radius, max_candidates=3)
                if not source_points:
                    continue
                for strokes in _stroke_variants(region, radius):
                    for source_point in source_points:
                        if trial_index >= trial_budget:
                            break
                        trial_index += 1
                        params = {
                            "source_point": [int(source_point[0]), int(source_point[1])],
                            "strokes": strokes,
                            "radius": int(radius),
                            "opacity": 0.95,
                            "feather": max(1.0, round(radius * 0.28, 2)),
                            "aligned": True,
                            "spacing": max(1.0, round(radius * 0.45, 2)),
                            "target_region": {
                                "x": int(region["x"]),
                                "y": int(region["y"]),
                                "width": int(region["width"]),
                                "height": int(region["height"]),
                            },
                        }
                        candidate, _ = clone_spec.operation(image, params, None)
                        metrics = _score_clone_candidate(image, candidate, region)
                        output_path = trial_dir / f"clone_trial_{trial_index:02d}.png"
                        save_image(candidate, output_path)
                        trial = {
                            "trial_index": trial_index,
                            "region": dict(region),
                            "params": params,
                            "output_path": str(output_path),
                            "metrics": metrics,
                            "accepted": bool(metrics["accepted"]),
                        }
                        trial_record["trials"].append(trial)
                        if best_trial is None or float(metrics["score"]) > float(best_trial["metrics"]["score"]):
                            best_trial = trial
                    if trial_index >= trial_budget:
                        break
                if trial_index >= trial_budget:
                    break
            if trial_index >= trial_budget:
                break

        accepted = best_trial is not None and bool(best_trial["metrics"]["accepted"])
        if accepted and best_trial is not None:
            state["plan"] = _inject_clone_step(plan, best_trial, allowed_tools)
            trial_record["accepted"] = True
            trial_record["selected_region"] = best_trial["region"]
            trial_record["selected_params"] = best_trial["params"]
            trial_record["reason"] = (
                "Accepted the best clone-stamp candidate after bounded local trials. "
                f"Improvement={best_trial['metrics']['improvement_ratio']:.4f}"
            )
        else:
            state["plan"] = _filter_plan_to_allowed_tools(plan, allowed_tools)
            trial_record["reason"] = "Tried clone-stamp candidates but none improved the target region enough to keep."

        if state.get("plan_history"):
            state["plan_history"][-1] = {"iteration": state["iteration"], **(state["plan"] or {})}
        state.setdefault("tool_trial_history", []).append(trial_record)
        self._handoff(state, trial_record)
        self._log(
            state,
            start,
            "accepted" if accepted else "rejected",
            {
                "trial_count": len(trial_record["trials"]),
                "accepted": accepted,
                "best_score": None if best_trial is None else best_trial["metrics"]["score"],
            },
        )
        record_node_event(
            state,
            node="trial_tools",
            phase="end",
            summary={"accepted": accepted, "trials": len(trial_record["trials"])},
        )
        return state

    def _handoff(self, state: WorkflowState, trial_record: dict[str, Any]) -> None:
        key_points = [
            f"attempted={trial_record['attempted']}",
            f"accepted={trial_record['accepted']}",
            f"trials={len(trial_record['trials'])}",
        ]
        if trial_record.get("selected_region"):
            region = trial_record["selected_region"]
            key_points.append(f"region=({region['x']},{region['y']},{region['width']},{region['height']})")
        record_agent_handoff(
            state,
            from_agent="tool_trial",
            to_agent="executor",
            summary=trial_record["reason"] or "Tool-trial agent forwarded the plan to execution.",
            key_points=key_points,
            payload={
                "tool": trial_record["tool"],
                "accepted": trial_record["accepted"],
                "trial_count": len(trial_record["trials"]),
                "selected_params": trial_record.get("selected_params"),
            },
        )

    def _log(self, state: WorkflowState, start: float, decision: str, output_summary: dict[str, Any]) -> None:
        record = AgentLog(
            run_id=state["run_id"],
            agent_name="tool_trial",
            iteration=state["iteration"],
            input_summary={
                "allowed_tools": state["request"].get("allowed_tools", []),
                "max_tool_trials": state["request"].get("max_tool_trials", 10),
            },
            decision=decision,
            output_summary=output_summary,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _should_run_clone_trials(
    state: WorkflowState,
    region_summary: dict[str, Any],
    allowed_tools: set[str],
    recent_feedback: str,
) -> bool:
    if "clone_stamp" not in allowed_tools:
        return False
    top_regions = region_summary.get("top_regions", [])
    if not top_regions:
        return False
    if allowed_tools == {"clone_stamp"}:
        return True
    prompt = state["prompt"].lower()
    clone_prompt = any(
        phrase in prompt
        for phrase in (
            "clone",
            "peeled",
            "peeling",
            "missing area",
            "white patch",
            "blank area",
            "stamp",
            "patch repair",
        )
    )
    large_damage = float(region_summary.get("largest_defect_ratio", 0.0)) > 0.001
    feedback_push = any(
        phrase in recent_feedback
        for phrase in (
            "preserve unaffected regions",
            "prefer masked local repair",
            "edit footprint was wider",
            "local repair",
        )
    )
    return clone_prompt or large_damage or feedback_push


def _recent_feedback_lines(state: WorkflowState) -> list[str]:
    lines: list[str] = []
    review = state.get("review") or {}
    lines.extend(review.get("findings", []))
    lines.extend(review.get("recommendations", []))
    retry = state.get("retry_decision") or {}
    if retry.get("reason"):
        lines.append(str(retry["reason"]))
    if retry.get("strategy"):
        lines.append(str(retry["strategy"]))
    return lines


def _previous_clone_regions(state: WorkflowState) -> set[tuple[int, int, int, int]]:
    regions: set[tuple[int, int, int, int]] = set()
    for item in state.get("tool_trial_history", []):
        region = item.get("selected_region")
        if isinstance(region, dict):
            regions.add(
                (
                    int(region.get("x", 0)),
                    int(region.get("y", 0)),
                    int(region.get("width", 0)),
                    int(region.get("height", 0)),
                )
            )
    return regions


def _clone_candidate_regions(
    image: np.ndarray,
    region_summary: dict[str, Any],
    prior_regions: set[tuple[int, int, int, int]],
) -> list[dict[str, int]]:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    ranked: list[tuple[float, dict[str, int]]] = []
    for region in region_summary.get("top_regions", [])[:8]:
        width = int(region.get("width", 0))
        height = int(region.get("height", 0))
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        area = int(region.get("area", width * height))
        if width < 8 or height < 8 or area < 180:
            continue
        if (x, y, width, height) in prior_regions:
            continue
        patch = gray[y : y + height, x : x + width]
        if patch.size == 0:
            continue
        ring = _ring_mask(gray.shape, region, max(12, int(max(width, height) * 0.35)))
        ring_values = gray[ring]
        if ring_values.size == 0:
            continue
        whiteness = float(np.mean(patch > max(210.0, float(ring_values.mean()) + 28.0)))
        region_score = float(area) * 0.003 + whiteness * 2.0 + max(0.0, float(patch.mean() - ring_values.mean())) / 24.0
        ranked.append((region_score, {"x": x, "y": y, "width": width, "height": height, "area": area}))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:3]]


def _radius_candidates(region: dict[str, int]) -> list[int]:
    width = int(region["width"])
    height = int(region["height"])
    base = max(4, min(18, int(round(min(width, height) * 0.2))))
    return sorted({max(3, int(round(base * factor))) for factor in (0.8, 1.0, 1.25)})


def _source_candidates_for_region(
    image: np.ndarray,
    region: dict[str, int],
    radius: int,
    *,
    max_candidates: int,
) -> list[tuple[int, int]]:
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    height, width = gray.shape
    x = int(region["x"])
    y = int(region["y"])
    region_w = int(region["width"])
    region_h = int(region["height"])
    inner = _box_mask(gray.shape, region)
    ring = _ring_mask(gray.shape, region, max(18, radius * 5))
    ring_values = gray[ring]
    if ring_values.size == 0:
        return []
    ring_mean = float(ring_values.mean())
    ring_std = float(ring_values.std())
    search_step = max(6, radius)
    white_threshold = max(210.0, ring_mean + 35.0)
    y_start = max(radius, y - region_h)
    y_stop = min(height - radius, y + region_h * 2)
    x_start = max(radius, x - region_w)
    x_stop = min(width - radius, x + region_w * 2)
    scored: list[tuple[float, tuple[int, int]]] = []

    for cy in range(y_start, y_stop, search_step):
        for cx in range(x_start, x_stop, search_step):
            patch_box = {
                "x": cx - radius,
                "y": cy - radius,
                "width": radius * 2 + 1,
                "height": radius * 2 + 1,
            }
            if _boxes_overlap(region, patch_box, pad=max(radius, 4)):
                continue
            patch = gray[patch_box["y"] : patch_box["y"] + patch_box["height"], patch_box["x"] : patch_box["x"] + patch_box["width"]]
            if patch.size == 0:
                continue
            patch_mean = float(patch.mean())
            patch_std = float(patch.std())
            white_ratio = float(np.mean(patch > white_threshold))
            distance = float(np.hypot(cx - (x + region_w / 2.0), cy - (y + region_h / 2.0)))
            score = abs(patch_mean - ring_mean)
            score += abs(patch_std - ring_std) * 0.6
            score += white_ratio * 160.0
            score += distance * 0.02
            overlap_penalty = float(np.mean(inner[max(0, cy - radius) : min(height, cy + radius + 1), max(0, cx - radius) : min(width, cx + radius + 1)]))
            score += overlap_penalty * 200.0
            scored.append((score, (int(cx), int(cy))))

    scored.sort(key=lambda item: item[0])
    deduped: list[tuple[int, int]] = []
    for _, point in scored:
        if all(np.hypot(point[0] - prior[0], point[1] - prior[1]) >= radius * 2 for prior in deduped):
            deduped.append(point)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _stroke_variants(region: dict[str, int], radius: int) -> list[list[dict[str, list[list[int]]]]]:
    x = int(region["x"])
    y = int(region["y"])
    width = int(region["width"])
    height = int(region["height"])
    padding = max(2, radius // 2)
    x0 = x + padding
    y0 = y + padding
    x1 = x + max(padding, width - padding - 1)
    y1 = y + max(padding, height - padding - 1)
    spacing = max(radius, 6)
    horizontal: list[dict[str, list[list[int]]]] = []
    vertical: list[dict[str, list[list[int]]]] = []

    for row in range(y0, y1 + 1, spacing):
        horizontal.append({"points": [[x0, row], [x1, row]]})
    for col in range(x0, x1 + 1, spacing):
        vertical.append({"points": [[col, y0], [col, y1]]})

    variants: list[list[dict[str, list[list[int]]]]] = []
    if horizontal:
        variants.append(horizontal[: min(5, len(horizontal))])
    if vertical and abs(width - height) <= max(width, height) * 0.4:
        variants.append(vertical[: min(5, len(vertical))])
    if vertical and width < height:
        variants.insert(0, vertical[: min(5, len(vertical))])
    return variants[:2] or [[]]


def _score_clone_candidate(before: np.ndarray, after: np.ndarray, region: dict[str, int]) -> dict[str, float | bool]:
    before_gray = np.dot(before[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    after_gray = np.dot(after[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    target = _box_mask(before_gray.shape, region)
    expanded = _expanded_box_mask(before_gray.shape, region, max(16, int(max(region["width"], region["height"]) * 0.45)))
    ring = expanded & ~target
    ring_values = before_gray[ring]
    if ring_values.size == 0:
        ring_values = before_gray[~target]
    ring_mean = float(ring_values.mean())
    ring_std = float(ring_values.std()) + 1e-6
    before_values = before_gray[target]
    after_values = after_gray[target]
    before_mismatch = float(np.mean(np.abs(before_values - ring_mean))) + abs(float(before_values.std()) - ring_std) * 0.35
    after_mismatch = float(np.mean(np.abs(after_values - ring_mean))) + abs(float(after_values.std()) - ring_std) * 0.35
    improvement = before_mismatch - after_mismatch
    improvement_ratio = improvement / max(before_mismatch, 1e-6)
    preserved_mask = ~expanded
    preserved_change = _masked_change_ratio(before, after, preserved_mask, threshold=8.0)
    target_change = _masked_change_ratio(before, after, target, threshold=8.0)
    score = improvement_ratio * 8.0 + target_change * 1.5 - preserved_change * 9.0
    accepted = bool(improvement_ratio > 0.08 and preserved_change < 0.05 and target_change > 0.02)
    return {
        "before_mismatch": before_mismatch,
        "after_mismatch": after_mismatch,
        "improvement": improvement,
        "improvement_ratio": improvement_ratio,
        "preserved_change_ratio": preserved_change,
        "target_change_ratio": target_change,
        "score": score,
        "accepted": accepted,
    }


def _inject_clone_step(plan: dict[str, Any], best_trial: dict[str, Any], allowed_tools: set[str]) -> dict[str, Any]:
    updated = _filter_plan_to_allowed_tools(plan, allowed_tools)
    filtered_steps = [step for step in updated.get("steps", []) if step.get("tool") != "clone_stamp"]
    clone_step = PlanStep(
        tool="clone_stamp",
        params=best_trial["params"],
        reason="Use bounded clone-stamp trials to repair a large damaged region with a tested local source.",
    ).model_dump()
    insert_at = 0
    for idx, step in enumerate(filtered_steps):
        if step.get("tool") in {"auto_white_balance", "shadow_highlight_balance", "clahe_contrast", "histogram_balance"}:
            insert_at = idx
            break
        insert_at = idx + 1
    filtered_steps.insert(insert_at, clone_step)
    updated["steps"] = filtered_steps
    feedback = list(updated.get("feedback_applied", []))
    feedback.append(
        "Tool-trial agent tested multiple clone-stamp candidates, reverted weaker ones, and inserted the best local repair."
    )
    updated["feedback_applied"] = list(dict.fromkeys(feedback))
    return updated


def _filter_plan_to_allowed_tools(plan: dict[str, Any], allowed_tools: set[str]) -> dict[str, Any]:
    updated = {**plan}
    updated["steps"] = [step for step in updated.get("steps", []) if step.get("tool") in allowed_tools]
    updated["recommended_tools"] = [item for item in updated.get("recommended_tools", []) if item.get("tool") in allowed_tools]
    return updated


def _box_mask(shape: tuple[int, int], region: dict[str, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0 = max(0, int(region["x"]))
    y0 = max(0, int(region["y"]))
    x1 = min(shape[1], x0 + int(region["width"]))
    y1 = min(shape[0], y0 + int(region["height"]))
    mask[y0:y1, x0:x1] = True
    return mask


def _ring_mask(shape: tuple[int, int], region: dict[str, int], pad: int) -> np.ndarray:
    outer_mask = _expanded_box_mask(shape, region, pad)
    inner_mask = _box_mask(shape, region)
    return outer_mask & ~inner_mask


def _expanded_box_mask(shape: tuple[int, int], region: dict[str, int], pad: int) -> np.ndarray:
    return _box_mask(
        shape,
        {
            "x": max(0, int(region["x"]) - pad),
            "y": max(0, int(region["y"]) - pad),
            "width": int(region["width"]) + pad * 2,
            "height": int(region["height"]) + pad * 2,
        },
    )


def _boxes_overlap(a: dict[str, int], b: dict[str, int], *, pad: int = 0) -> bool:
    ax0 = int(a["x"]) - pad
    ay0 = int(a["y"]) - pad
    ax1 = int(a["x"]) + int(a["width"]) + pad
    ay1 = int(a["y"]) + int(a["height"]) + pad
    bx0 = int(b["x"])
    by0 = int(b["y"])
    bx1 = int(b["x"]) + int(b["width"])
    by1 = int(b["y"]) + int(b["height"])
    return not (bx1 <= ax0 or bx0 >= ax1 or by1 <= ay0 or by0 >= ay1)


def _masked_change_ratio(before: np.ndarray, after: np.ndarray, mask: np.ndarray, *, threshold: float) -> float:
    if not mask.any():
        return 0.0
    delta = np.abs(before.astype(np.float32) - after.astype(np.float32)).mean(axis=2)
    return float(np.mean(delta[mask] > threshold))
