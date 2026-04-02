from __future__ import annotations

import time
from typing import Any

from veriedit.io.writer import append_jsonl
from veriedit.observability import record_node_event
from veriedit.schemas import AgentLog, PolicyStatus, WorkflowState

REJECT_PATTERNS = [
    "remove proof of editing",
    "hide that something was added",
    "make this manipulated image look original",
    "conceal evidence",
    "without detection",
    "forge",
    "counterfeit",
    "tamper with evidence",
]

CAUTION_PATTERNS = [
    "remove person",
    "erase object",
    "document",
    "receipt",
    "license",
    "passport",
    "id card",
]


class PolicyAgent:
    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="policy_check", phase="start", summary={"prompt": state["prompt"][:160]})
        prompt = state["prompt"].lower()
        status = "allow"
        risk_level = "low"
        warnings: list[str] = []
        reason = None
        if any(pattern in prompt for pattern in REJECT_PATTERNS):
            status = "reject"
            risk_level = "high"
            reason = "Request appears to seek deceptive or fraudulent image manipulation."
        elif any(pattern in prompt for pattern in CAUTION_PATTERNS):
            status = "caution"
            risk_level = "medium"
            warnings.append("Sensitive request detected; only low-level restorative edits are allowed.")
        policy = PolicyStatus(
            status=status,
            risk_level=risk_level,
            constraints=[
                "non-generative only",
                "do not alter semantic identity",
                "do not fabricate missing content",
                "reference image may influence style only",
            ],
            warnings=warnings,
            reason=reason,
        )
        state["policy_status"] = policy.model_dump()
        state["stop_reason"] = reason if status == "reject" else None
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="policy",
                iteration=state["iteration"],
                input_summary={"prompt": state["prompt"][:160]},
                decision=policy.status,
                output_summary=policy.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        record_node_event(state, node="policy_check", phase="end", summary={"status": policy.status, "risk_level": policy.risk_level})
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')
