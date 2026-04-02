from __future__ import annotations

import time

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl
from veriedit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from veriedit.schemas import AgentLog, DiagnosticsBundle, SourceDiagnostics, StyleProfile, WorkflowState


class DiagnosticsAgent:
    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        source_image, source_metadata = load_image(state["source_image_path"])
        source_summary = summarize_image_quality(source_image, source_metadata)
        reference_profile = None
        if state["reference_image_path"]:
            reference_image, _ = load_image(state["reference_image_path"])
            reference_profile = style_profile_from_image(reference_image)
            state["style_profile"] = reference_profile
        diagnostics = DiagnosticsBundle(
            source=SourceDiagnostics(**source_summary),
            reference=StyleProfile(**reference_profile) if reference_profile else None,
        )
        state["diagnostics"] = diagnostics.model_dump()
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="diagnostics",
                iteration=state["iteration"],
                input_summary={"source_image": state["source_image_path"], "reference_image": state["reference_image_path"]},
                decision="diagnosed",
                output_summary=diagnostics.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')
