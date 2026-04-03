from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable

from veriedit.observability import record_node_event
from veriedit.schemas import WorkflowState

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependency/runtime dependent
    ConversableAgent = None
    GroupChat = None
    GroupChatManager = None
    UserProxyAgent = None


def has_ag2_runtime() -> bool:
    return all(item is not None for item in (ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent))


class AG2WorkflowRuntime:
    def __init__(
        self,
        *,
        policy_agent: Any,
        diagnostics_agent: Any,
        planner_agent: Any,
        tool_trial_agent: Any,
        executor_agent: Any,
        reviewer_agent: Any,
        human_approval_agent: Any,
        retry_agent: Any,
        finalize_state: Callable[[WorkflowState], WorkflowState],
    ) -> None:
        if not has_ag2_runtime():  # pragma: no cover - dependency/runtime dependent
            raise RuntimeError("AG2 runtime is unavailable. Install ag2 to enable the multi-agent runtime layer.")
        self.policy_agent = policy_agent
        self.diagnostics_agent = diagnostics_agent
        self.planner_agent = planner_agent
        self.tool_trial_agent = tool_trial_agent
        self.executor_agent = executor_agent
        self.reviewer_agent = reviewer_agent
        self.human_approval_agent = human_approval_agent
        self.retry_agent = retry_agent
        self.finalize_state = finalize_state
        self.state: WorkflowState | None = None
        self._groupchat = None

    def run(self, state: WorkflowState) -> WorkflowState:
        self.state = state
        record_node_event(state, node="ag2_runtime", phase="start", summary={"runtime": "ag2"})
        coordinator = UserProxyAgent(
            name="Coordinator",
            system_message="Kick off the workflow and let the runtime route agent turns.",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            default_auto_reply="",
            silent=True,
        )
        policy = self._make_agent("PolicyAgent", "Policy gate for non-generative image editing.")
        diagnostics = self._make_agent("DiagnosticsAgent", "Image diagnostics and defect mapping.")
        planner = self._make_agent("PlannerAgent", "Conservative edit planning agent.")
        tool_trial = self._make_agent("ToolTrialAgent", "Tests bounded local repair candidates, reverts weak ones, and forwards the best step.")
        executor = self._make_agent("ExecutorAgent", "Executes approved tool steps on the image.")
        reviewer = self._make_agent("ReviewerAgent", "Reviews realism, preservation, and prompt satisfaction.")
        human_approval = self._make_agent("HumanApprovalAgent", "Requests human review when ambiguity is high.")
        retry = self._make_agent("RetryAgent", "Decides whether to accept, retry, or stop.")
        finalizer = self._make_agent("FinalizerAgent", "Finalizes artifacts and outputs.")

        groupchat = GroupChat(
            agents=[coordinator, policy, diagnostics, planner, tool_trial, executor, reviewer, human_approval, retry, finalizer],
            messages=[],
            max_round=max(12, state["max_iterations"] * 8 + 4),
            speaker_selection_method=self._select_next_speaker,
            allow_repeat_speaker=False,
            send_introductions=False,
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=max(12, state["max_iterations"] * 8 + 4),
            silent=True,
        )
        self._groupchat = groupchat

        policy.register_reply(manager, self._build_reply("policy", self.policy_agent.run))
        diagnostics.register_reply(manager, self._build_reply("diagnostics", self.diagnostics_agent.run))
        planner.register_reply(manager, self._build_reply("planner", self.planner_agent.run))
        tool_trial.register_reply(manager, self._build_reply("tool_trial", self.tool_trial_agent.run))
        executor.register_reply(manager, self._build_reply("executor", self.executor_agent.run))
        reviewer.register_reply(manager, self._build_reply("reviewer", self.reviewer_agent.run))
        human_approval.register_reply(manager, self._build_reply("human_approval", self.human_approval_agent.run))
        retry.register_reply(manager, self._build_reply("retry", self.retry_agent.run))
        finalizer.register_reply(manager, self._build_reply("finalizer", self.finalize_state))

        kickoff = {
            "content": self._initial_message(state),
            "role": "user",
        }
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            coordinator.initiate_chat(manager, message=kickoff, silent=True, summary_method="last_msg")
        assert self.state is not None
        self._save_chat_history(Path(self.state["run_dir"]))
        record_node_event(self.state, node="ag2_runtime", phase="end", summary={"runtime": "ag2", "messages": len(groupchat.messages)})
        return self.state

    def _make_agent(self, name: str, system_message: str):
        return ConversableAgent(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            default_auto_reply="",
            silent=True,
        )

    def _build_reply(self, label: str, runner: Callable[[WorkflowState], WorkflowState]):
        def reply(recipient, messages=None, sender=None, config=None):
            assert self.state is not None
            self.state = runner(self.state)
            return True, self._summary_message(label, self.state)

        return reply

    def _select_next_speaker(self, last_speaker, groupchat):
        assert self.state is not None
        agents = {agent.name: agent for agent in groupchat.agents}
        if last_speaker is None:
            return agents["Coordinator"]
        if last_speaker.name == "Coordinator":
            return agents["PolicyAgent"]
        if last_speaker.name == "PolicyAgent":
            if (self.state["policy_status"] or {}).get("status") == "reject":
                return agents["FinalizerAgent"]
            return agents["DiagnosticsAgent"]
        if last_speaker.name == "DiagnosticsAgent":
            return agents["PlannerAgent"]
        if last_speaker.name == "PlannerAgent":
            return agents["ToolTrialAgent"]
        if last_speaker.name == "ToolTrialAgent":
            return agents["ExecutorAgent"]
        if last_speaker.name == "ExecutorAgent":
            return agents["ReviewerAgent"]
        if last_speaker.name == "ReviewerAgent":
            return agents["HumanApprovalAgent"]
        if last_speaker.name == "HumanApprovalAgent":
            return agents["RetryAgent"]
        if last_speaker.name == "RetryAgent":
            decision = (self.state["retry_decision"] or {}).get("decision", "stop")
            if decision == "retry":
                self.state["iteration"] += 1
                return agents["DiagnosticsAgent"]
            return agents["FinalizerAgent"]
        if last_speaker.name == "FinalizerAgent":
            return None
        return None

    def _initial_message(self, state: WorkflowState) -> str:
        request = {
            "run_id": state["run_id"],
            "prompt": state["prompt"],
            "source_image_path": state["source_image_path"],
            "reference_image_path": state["reference_image_path"],
            "allowed_tools": state["request"].get("allowed_tools", []),
            "max_iterations": state["max_iterations"],
            "max_tool_trials": state["request"].get("max_tool_trials", 10),
        }
        return (
            "Run the VeriEdit restoration workflow through the AG2 runtime.\n"
            f"State:\n{json.dumps(request, indent=2)}"
        )

    def _summary_message(self, label: str, state: WorkflowState) -> str:
        latest = self._latest_handoff(label, state)
        if latest is not None:
            points = latest.get("key_points", [])
            suffix = f" Key points: {'; '.join(points[:3])}" if points else ""
            return f"{latest.get('summary')}{suffix}"
        if label == "policy":
            policy = state["policy_status"] or {}
            return f"Policy complete: status={policy.get('status')} risk={policy.get('risk_level')}"
        if label == "diagnostics":
            source = (state["diagnostics"] or {}).get("source", {})
            return (
                "Diagnostics complete: "
                f"dust={source.get('dust_candidates')} scratch={source.get('scratch_candidates')} "
                f"yellow_cast={source.get('yellow_cast')}"
            )
        if label == "planner":
            plan = state["plan"] or {}
            return f"Plan complete: objective={plan.get('objective')} steps={len(plan.get('steps', []))}"
        if label == "tool_trial":
            latest_trial = (state.get("tool_trial_history") or [])[-1] if state.get("tool_trial_history") else {}
            return (
                "Tool trials complete: "
                f"accepted={latest_trial.get('accepted')} "
                f"trials={len(latest_trial.get('trials', []))}"
            )
        if label == "executor":
            return f"Execution complete: executed_steps={len(state['executed_steps'])} current_image={state['current_image_path']}"
        if label == "reviewer":
            review = state["review"] or {}
            return (
                "Review complete: "
                f"status={review.get('status')} prompt_score={review.get('prompt_score')} artifact_risk={review.get('artifact_risk')}"
            )
        if label == "human_approval":
            human = state.get("human_review") or {}
            return f"Human approval gate: status={human.get('status')} reason={human.get('reason')}"
        if label == "retry":
            retry = state.get("retry_decision") or {}
            return f"Retry decision: decision={retry.get('decision')} strategy={retry.get('strategy')}"
        if label == "finalizer":
            final = state.get("final_result") or {}
            return f"Finalized: success={final.get('success')} stop_reason={final.get('stop_reason')}"
        return f"{label} complete"

    def _latest_handoff(self, label: str, state: WorkflowState) -> dict[str, Any] | None:
        mapping = {
            "policy": "policy",
            "diagnostics": "diagnostics",
            "planner": "planner",
            "tool_trial": "tool_trial",
            "executor": "executor",
            "reviewer": "reviewer",
            "human_approval": "human_approval",
            "retry": "retry",
        }
        target = mapping.get(label)
        if target is None:
            return None
        for handoff in reversed(state.get("agent_handoffs", [])):
            if handoff.get("from_agent") == target:
                return handoff
        return None

    def _save_chat_history(self, run_dir: Path) -> None:
        if self._groupchat is None or self.state is None:
            return
        chat_history = list(self._groupchat.messages)
        json_path = run_dir / "ag2_chat_history.json"
        md_path = run_dir / "ag2_chat_history.md"
        json_path.write_text(json.dumps(chat_history, indent=2), encoding="utf-8")
        lines = ["# AG2 Chat History", ""]
        for message in chat_history:
            name = message.get("name") or message.get("role", "unknown")
            content = message.get("content", "")
            lines.append(f"## {name}")
            lines.append("")
            lines.append(str(content))
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
