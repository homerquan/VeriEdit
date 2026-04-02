from __future__ import annotations

import time

from veriedit.config import WorkflowConfig
from veriedit.human_review import human_approval_path, load_human_approval
from veriedit.io.writer import append_jsonl
from veriedit.observability import record_node_event
from veriedit.schemas import AgentLog, HumanReviewRequest, WorkflowState


class HumanApprovalAgent:
    def __init__(self, config: WorkflowConfig | None = None) -> None:
        self.config = config or WorkflowConfig()

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="human_approval_gate", phase="start")
        review = state.get("review") or {}
        approval_file = load_human_approval(state["run_dir"])
        if approval_file:
            status = approval_file.get("status", "approved")
            human_review = HumanReviewRequest(
                status="approved" if status == "approved" else "rejected",
                reason="Human reviewer recorded a decision.",
                reasons=[approval_file.get("notes", "")] if approval_file.get("notes") else [],
                confidence=1.0,
                approval_json=str(human_approval_path(state["run_dir"])),
            )
        elif not state["request"].get("enable_human_approval", True):
            human_review = HumanReviewRequest(status="not_needed", reason="Human approval loop disabled for this request.")
        else:
            human_review = _decide_human_review(review, state, self.config)
        state["human_review"] = human_review.model_dump()
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="human_approval",
                iteration=state["iteration"],
                input_summary={
                    "review_status": review.get("status"),
                    "prompt_score": review.get("prompt_score"),
                    "artifact_risk": review.get("artifact_risk"),
                },
                decision=human_review.status,
                output_summary=human_review.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        record_node_event(
            state,
            node="human_approval_gate",
            phase="end",
            summary={"status": human_review.status, "reason": human_review.reason},
        )
        return state

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _decide_human_review(review: dict, state: WorkflowState, config: WorkflowConfig) -> HumanReviewRequest:
    if not review:
        return HumanReviewRequest(status="not_needed", reason="No review available.")
    if review.get("status") == "accept":
        return HumanReviewRequest(status="not_needed", reason="Reviewer accepted the result confidently.")

    confidence = float(review.get("confidence", 0.75))
    prompt_score = float(review.get("prompt_score", 0.0))
    artifact_risk = float(review.get("artifact_risk", 1.0))
    semantic_risk = float(review.get("semantic_fabrication_risk", 1.0))
    preserved_change = float((review.get("patch_metrics") or {}).get("preserved_region_change_ratio", 0.0))

    if artifact_risk > 0.7 or semantic_risk > 0.8:
        return HumanReviewRequest(status="not_needed", reason="Result is clearly unsafe or over-edited; automatic stop is sufficient.")

    reasons: list[str] = []
    if confidence < config.human_review_confidence_threshold:
        reasons.append("Reviewer confidence is low enough that a human check is safer.")
    if preserved_change > config.human_review_preserved_region_threshold:
        reasons.append("Preserved regions changed more than expected outside detected defect areas.")
    if abs(prompt_score - config.prompt_satisfaction_threshold) <= config.human_review_margin:
        reasons.append("Prompt satisfaction is close to the acceptance threshold.")
    if abs(artifact_risk - config.artifact_risk_threshold) <= config.human_review_margin:
        reasons.append("Artifact risk is close to the rejection threshold.")
    if review.get("status") == "stop" and artifact_risk <= 0.6:
        reasons.append("Automatic iterations stalled, but the result may still be acceptable to a human.")

    if not reasons:
        return HumanReviewRequest(status="not_needed", reason="No ambiguity signal crossed the human-review threshold.", confidence=confidence)

    return HumanReviewRequest(
        status="pending",
        reason="Human approval requested for an ambiguous result.",
        reasons=reasons,
        confidence=confidence,
        approval_json=str(human_approval_path(state["run_dir"])),
        manual_eval_md=str((state["run_dir"])) + "/manual_eval.md",
    )
