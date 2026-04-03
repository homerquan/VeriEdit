from __future__ import annotations

import shutil
from pathlib import Path

from veriedit.agents import DiagnosticsAgent, ExecutorAgent, HumanApprovalAgent, PlannerAgent, RetryAgent, ReviewerAgent
from veriedit.config import WorkflowConfig
from veriedit.io.writer import ensure_run_artifacts
from veriedit.policy import PolicyAgent
from veriedit.reports.report_builder import finalize_outputs
from veriedit.runtime import AG2WorkflowRuntime, has_ag2_runtime
from veriedit.schemas import EditRequest, FinalResult, WorkflowState


class VeriEditWorkflow:
    def __init__(self, config: WorkflowConfig | None = None) -> None:
        self.config = config or WorkflowConfig()
        self.policy_agent = PolicyAgent()
        self.diagnostics_agent = DiagnosticsAgent()
        self.planner_agent = PlannerAgent(model=self.config.default_llm_model)
        self.executor_agent = ExecutorAgent()
        self.reviewer_agent = ReviewerAgent(model=self.config.default_llm_model)
        self.human_approval_agent = HumanApprovalAgent(config=self.config)
        self.retry_agent = RetryAgent(config=self.config)
        self.runtime = (
            AG2WorkflowRuntime(
                policy_agent=self.policy_agent,
                diagnostics_agent=self.diagnostics_agent,
                planner_agent=self.planner_agent,
                executor_agent=self.executor_agent,
                reviewer_agent=self.reviewer_agent,
                human_approval_agent=self.human_approval_agent,
                retry_agent=self.retry_agent,
                finalize_state=self._finalize_state,
            )
            if has_ag2_runtime()
            else None
        )
        self.graph = None

    def run(self, request: EditRequest):
        artifacts = ensure_run_artifacts(
            self.config.artifact_root,
            source_image=request.source_image,
            output_path=request.output_path,
            reference_image=request.reference_image,
        )
        state: WorkflowState = {
            "run_id": artifacts.run_id,
            "request": request.model_dump(),
            "source_image_path": str(artifacts.source_copy),
            "reference_image_path": str(artifacts.reference_copy) if artifacts.reference_copy else None,
            "prompt": request.prompt,
            "output_path": str(artifacts.output_image),
            "run_dir": str(artifacts.run_dir),
            "current_image_path": str(artifacts.source_copy),
            "policy_status": {},
            "diagnostics": {},
            "diagnostic_artifacts": {},
            "style_profile": None,
            "plan": None,
            "plan_history": [],
            "executed_steps": [],
            "agent_handoffs": [],
            "observation_trace": [],
            "intermediate_paths": [],
            "review": None,
            "review_history": [],
            "human_review": None,
            "retry_decision": None,
            "final_result": None,
            "logs": [],
            "iteration": 1,
            "max_iterations": request.max_iterations,
            "stop_reason": None,
        }
        if self.runtime is not None:
            state = self.runtime.run(state)
        else:
            state = self._run_without_runtime(state)
        return finalize_outputs(state)

    def _run_without_runtime(self, state: WorkflowState) -> WorkflowState:
        state = self.policy_agent.run(state)
        if state["policy_status"].get("status") == "reject":
            return self._finalize_state(state)
        while True:
            state = self.diagnostics_agent.run(state)
            state = self.planner_agent.run(state)
            state = self.executor_agent.run(state)
            state = self.reviewer_agent.run(state)
            state = self.human_approval_agent.run(state)
            state = self.retry_agent.run(state)
            decision = (state["retry_decision"] or {}).get("decision")
            if decision != "retry":
                return self._finalize_state(state)
            state["iteration"] += 1

    def _finalize_state(self, state: WorkflowState) -> WorkflowState:
        review = state["review"] or {}
        human_review = state.get("human_review") or {}
        final_output_path = Path(state["output_path"])
        if state["policy_status"].get("status") != "reject" and state["current_image_path"]:
            current_path = Path(state["current_image_path"])
            if current_path != final_output_path:
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current_path, final_output_path)
        final = FinalResult(
            success=bool(state["policy_status"].get("status") != "reject" and (state["retry_decision"] or {}).get("decision") == "accept"),
            output_image=str(final_output_path) if state["policy_status"].get("status") != "reject" else None,
            report_json=str(Path(state["run_dir"]) / "report.json"),
            report_md=str(Path(state["run_dir"]) / "report.md"),
            summary_md=str(Path(state["run_dir"]) / "edit_summary.md"),
            observation_json=str(Path(state["run_dir"]) / "observation_trace.json"),
            observation_md=str(Path(state["run_dir"]) / "observation_trace.md"),
            iterations=state["iteration"],
            applied_tools=[step["tool"] for step in state["executed_steps"] if step["status"] == "ok"],
            review_summary="; ".join(review.get("findings", [])[:3]) or state["stop_reason"] or "No review findings available.",
            stop_reason=state["stop_reason"] or (state["retry_decision"] or {}).get("reason"),
            run_dir=state["run_dir"],
            human_review_status=human_review.get("status"),
            human_review_reason=human_review.get("reason"),
            manual_eval_md=human_review.get("manual_eval_md"),
            human_approval_json=human_review.get("approval_json"),
        )
        state["final_result"] = final.model_dump()
        return state
