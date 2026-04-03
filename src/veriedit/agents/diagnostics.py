from __future__ import annotations

import time

from pathlib import Path

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl
from veriedit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from veriedit.metrics.regions import defect_masks, region_summary, save_mask_artifacts
from veriedit.observability import record_agent_handoff, record_node_event
from veriedit.schemas import AgentLog, DiagnosticsBundle, SourceDiagnostics, StyleProfile, WorkflowState


class DiagnosticsAgent:
    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="diagnose_inputs", phase="start")
        source_image, source_metadata = load_image(state["source_image_path"])
        source_summary = summarize_image_quality(source_image, source_metadata)
        current_path = state["current_image_path"] or state["source_image_path"]
        current_image, current_metadata = load_image(current_path)
        current_summary = summarize_image_quality(current_image, current_metadata)
        masks = defect_masks(current_image)
        regions = region_summary(masks)
        artifacts = save_mask_artifacts(current_image, masks, Path(state["run_dir"]) / "diagnostics" / f"iter_{state['iteration']:02d}")
        reference_profile = None
        if state["reference_image_path"]:
            reference_image, _ = load_image(state["reference_image_path"])
            reference_profile = style_profile_from_image(reference_image)
            state["style_profile"] = reference_profile
        diagnostics = DiagnosticsBundle(
            source=SourceDiagnostics(**source_summary),
            current=SourceDiagnostics(**current_summary),
            reference=StyleProfile(**reference_profile) if reference_profile else None,
            regions=regions,
            artifacts=artifacts,
        )
        state["diagnostics"] = diagnostics.model_dump()
        state["diagnostic_artifacts"] = artifacts
        record_agent_handoff(
            state,
            from_agent="diagnostics",
            to_agent="planner",
            summary="Diagnostics summarized the current working image and localized likely defect regions.",
            key_points=[
                f"current_dust={current_summary['dust_candidates']}",
                f"current_scratch={current_summary['scratch_candidates']}",
                f"largest_defect_ratio={regions.get('largest_defect_ratio', 0.0)}",
            ],
            payload={
                "source": diagnostics.source.model_dump(),
                "current": diagnostics.current.model_dump() if diagnostics.current else {},
                "regions": regions,
                "reference": reference_profile or {},
            },
        )
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="diagnostics",
                iteration=state["iteration"],
                input_summary={
                    "source_image": state["source_image_path"],
                    "current_image": current_path,
                    "reference_image": state["reference_image_path"],
                },
                decision="diagnosed",
                output_summary={
                    "source": diagnostics.source.model_dump(),
                    "current": diagnostics.current.model_dump() if diagnostics.current else {},
                    "regions": regions,
                    "artifacts": artifacts,
                },
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        record_node_event(state, node="diagnose_inputs", phase="end", summary={"defect_region_count": regions["defect_region_count"]})
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')
