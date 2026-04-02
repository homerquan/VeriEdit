from __future__ import annotations

import time
from pathlib import Path

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl, save_image
from veriedit.metrics.iq_metrics import summarize_image_quality
from veriedit.metrics.similarity import compare_images
from veriedit.schemas import AgentLog, ExecutionRecord, WorkflowState
from veriedit.tools import build_tool_registry
from veriedit.tools.base import sanitize_numeric_params


class ExecutorAgent:
    def __init__(self) -> None:
        self.registry = build_tool_registry()

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        current_image, metadata = load_image(state["current_image_path"])
        reference_image = load_image(state["reference_image_path"])[0] if state["reference_image_path"] else None
        plan = state["plan"] or {}
        step_records: list[dict] = []
        intermediate_paths = list(state["intermediate_paths"])
        save_intermediates = bool(state["request"].get("save_intermediates", True))
        latest_output_path = state["current_image_path"]
        for index, raw_step in enumerate(plan.get("steps", []), start=1):
            spec = self.registry.get(raw_step["tool"])
            params = sanitize_numeric_params(raw_step.get("params", {}), spec.parameter_bounds)
            before_metrics = summarize_image_quality(current_image, metadata)
            candidate, operation_notes = spec.operation(current_image, params, reference_image)
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
            status = "ok"
            notes: list[str] = []
            if _step_is_harmful(raw_step["tool"], before_metrics, after_metrics, comparison):
                candidate = current_image
                after_metrics = before_metrics
                status = "rolled_back"
                notes.append("Step exceeded safety heuristics and was rolled back.")
            else:
                notes.append("Step applied successfully.")
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
                before_metrics=_slim_metrics(before_metrics, comparison=False),
                after_metrics={**_slim_metrics(after_metrics, comparison=False), **comparison},
                output_path=str(output_path),
                status=status,
                notes=notes,
            )
            step_records.append(record.model_dump())
        state["executed_steps"].extend(step_records)
        state["intermediate_paths"] = intermediate_paths
        state["current_image_path"] = latest_output_path
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
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _step_is_harmful(tool_name: str, before: dict, after: dict, comparison: dict[str, float]) -> bool:
    if comparison["change_area_ratio"] > 0.8:
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
