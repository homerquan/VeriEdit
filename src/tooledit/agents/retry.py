from __future__ import annotations

import time

from tooledit.config import WorkflowConfig
from tooledit.io.writer import append_jsonl
from tooledit.schemas import AgentLog, RetryDecision, WorkflowState


class RetryAgent:
    def __init__(self, config: WorkflowConfig | None = None) -> None:
        self.config = config or WorkflowConfig()

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        review = state["review"] or {}
        policy = state["policy_status"] or {}
        if policy.get("status") == "reject":
            decision = RetryDecision(decision="stop", reason=policy.get("reason", "Rejected by policy"))
        elif (
            review.get("prompt_score", 0.0) >= self.config.prompt_satisfaction_threshold
            and review.get("artifact_risk", 1.0) <= self.config.artifact_risk_threshold
            and review.get("semantic_fabrication_risk", 1.0) <= 0.55
        ):
            decision = RetryDecision(decision="accept", reason="Prompt satisfied and artifacts remain within thresholds.")
        elif state["iteration"] >= state["max_iterations"] or review.get("status") == "stop":
            decision = RetryDecision(decision="stop", reason="Reached max iterations or improvement stalled.")
        else:
            strategy = _strategy_from_review(review)
            decision = RetryDecision(decision="retry", reason="Recoverable issues detected.", strategy=strategy)
        state["retry_decision"] = decision.model_dump()
        if decision.decision != "retry":
            state["stop_reason"] = decision.reason
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="retry",
                iteration=state["iteration"],
                input_summary={"review_status": review.get("status"), "prompt_score": review.get("prompt_score")},
                decision=decision.decision,
                output_summary=decision.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _strategy_from_review(review: dict) -> str:
    recommendations = " ".join(review.get("recommendations", [])).lower()
    if "reduce sharpen" in recommendations:
        return "reduce sharpen and rerun"
    if "stop further denoise" in recommendations:
        return "reduce denoise and rerun"
    return "revise plan conservatively"
