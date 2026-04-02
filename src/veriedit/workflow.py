from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from veriedit.agents import DiagnosticsAgent, ExecutorAgent, HumanApprovalAgent, PlannerAgent, RetryAgent, ReviewerAgent
from veriedit.config import WorkflowConfig
from veriedit.io.writer import ensure_run_artifacts
from veriedit.policy import PolicyAgent
from veriedit.reports.report_builder import finalize_outputs
from veriedit.schemas import EditRequest, FinalResult, WorkflowState

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency/runtime dependent
    END = "__end__"
    StateGraph = None


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
        self.graph = self._build_graph() if StateGraph is not None else None

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
            "executed_steps": [],
            "observation_trace": [],
            "intermediate_paths": [],
            "review": None,
            "human_review": None,
            "retry_decision": None,
            "final_result": None,
            "logs": [],
            "iteration": 1,
            "max_iterations": request.max_iterations,
            "stop_reason": None,
        }
        if self.graph is not None:
            state = self.graph.invoke(state)
        else:
            state = self._run_without_langgraph(state)
        return finalize_outputs(state)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("policy_check", self.policy_agent.run)
        graph.add_node("diagnose_inputs", self.diagnostics_agent.run)
        graph.add_node("plan_edits", self.planner_agent.run)
        graph.add_node("execute_plan", self.executor_agent.run)
        graph.add_node("review_result", self.reviewer_agent.run)
        graph.add_node("human_approval_gate", self.human_approval_agent.run)
        graph.add_node("decide_retry", self.retry_agent.run)
        graph.add_node("finalize_report", self._finalize_state)
        graph.set_entry_point("policy_check")
        graph.add_conditional_edges(
            "policy_check",
            self._route_policy,
            {"reject": "finalize_report", "allow": "diagnose_inputs"},
        )
        graph.add_edge("diagnose_inputs", "plan_edits")
        graph.add_edge("plan_edits", "execute_plan")
        graph.add_edge("execute_plan", "review_result")
        graph.add_edge("review_result", "human_approval_gate")
        graph.add_edge("human_approval_gate", "decide_retry")
        graph.add_conditional_edges(
            "decide_retry",
            self._route_retry,
            {"accept": "finalize_report", "retry": "plan_edits", "stop": "finalize_report"},
        )
        graph.add_edge("finalize_report", END)
        return graph.compile()

    def _run_without_langgraph(self, state: WorkflowState) -> WorkflowState:
        state = self.policy_agent.run(state)
        if state["policy_status"].get("status") == "reject":
            return self._finalize_state(state)
        state = self.diagnostics_agent.run(state)
        while True:
            state = self.planner_agent.run(state)
            state = self.executor_agent.run(state)
            state = self.reviewer_agent.run(state)
            state = self.human_approval_agent.run(state)
            state = self.retry_agent.run(state)
            decision = (state["retry_decision"] or {}).get("decision")
            if decision != "retry":
                return self._finalize_state(state)
            state["iteration"] += 1

    def _route_policy(self, state: WorkflowState) -> Literal["allow", "reject"]:
        return "reject" if state["policy_status"].get("status") == "reject" else "allow"

    def _route_retry(self, state: WorkflowState) -> Literal["accept", "retry", "stop"]:
        decision = (state["retry_decision"] or {}).get("decision", "stop")
        if decision == "retry":
            state["iteration"] += 1
        return decision

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
