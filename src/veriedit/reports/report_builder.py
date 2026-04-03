from __future__ import annotations

import json
import os
from pathlib import Path

from veriedit.human_review import human_approval_path
from veriedit.io.writer import write_json, write_text
from veriedit.manual_eval import build_manual_eval_markdown
from veriedit.schemas import EditResult, FinalResult, WorkflowState


def build_report_payload(state: WorkflowState) -> dict:
    return {
        "run_id": state["run_id"],
        "request": state["request"],
        "source_image_path": state["source_image_path"],
        "reference_image_path": state["reference_image_path"],
        "policy_status": state["policy_status"],
        "diagnostics": state["diagnostics"],
        "diagnostic_artifacts": state.get("diagnostic_artifacts", {}),
        "style_profile": state["style_profile"],
        "plan": state["plan"],
        "plan_history": state.get("plan_history", []),
        "tool_trial_history": state.get("tool_trial_history", []),
        "executed_steps": state["executed_steps"],
        "agent_handoffs": state.get("agent_handoffs", []),
        "observation_trace": state.get("observation_trace", []),
        "review": state["review"],
        "review_history": state.get("review_history", []),
        "human_review": state.get("human_review"),
        "retry_decision": state["retry_decision"],
        "final_result": state["final_result"],
        "intermediate_paths": state["intermediate_paths"],
        "logs": state["logs"],
    }


def build_markdown_report(state: WorkflowState) -> str:
    final = state["final_result"] or {}
    review = state["review"] or {}
    human_review = state.get("human_review") or {}
    report_path = Path(final.get("report_md") or Path(state["run_dir"]) / "report.md").resolve()
    source_path = Path(state["source_image_path"]).resolve()
    reference_path = Path(state["reference_image_path"]).resolve() if state["reference_image_path"] else None
    result_path = Path(final["output_image"]).resolve() if final.get("output_image") else Path(state["current_image_path"]).resolve()
    lines = [
        f"# VeriEdit Report: {state['run_id']}",
        "",
        "## Request",
        f"- Source image: `{state['source_image_path']}`",
        f"- Reference image: `{state['reference_image_path']}`" if state["reference_image_path"] else "- Reference image: none",
        f"- Prompt: {state['prompt']}",
        f"- Allowed tools: {', '.join(state['request'].get('allowed_tools', [])) or 'all'}",
        "",
        "## Images",
        "",
        "### Source",
        f"![Source]({_relative_markdown_path(source_path, report_path)})",
        "",
    ]
    if reference_path:
        lines.extend(
            [
                "### Reference",
                f"![Reference]({_relative_markdown_path(reference_path, report_path)})",
                "",
            ]
        )
    lines.extend(
        [
            "### Result",
            f"![Result]({_relative_markdown_path(result_path, report_path)})",
            "",
            "## Policy",
            f"- Status: {state['policy_status'].get('status')}",
            f"- Risk level: {state['policy_status'].get('risk_level')}",
            f"- Constraints: {', '.join(state['policy_status'].get('constraints', []))}",
            "",
            "## Plan",
            f"- Objective: {(state['plan'] or {}).get('objective', 'n/a')}",
            f"- Acceptance: {', '.join((state['plan'] or {}).get('acceptance', [])) or 'n/a'}",
            f"- Recommended tools: {', '.join(item['tool'] for item in (state['plan'] or {}).get('recommended_tools', [])[:6]) or 'n/a'}",
            f"- Diagnostic board: `{(state.get('diagnostic_artifacts') or {}).get('regions_board', 'n/a')}`",
            "",
            "### Detected Problems",
        ]
    )
    detected_problems = (state["plan"] or {}).get("detected_problems", [])
    if detected_problems:
        for problem in detected_problems:
            lines.append(
                f"- `{problem['problem']}` ({problem['severity']}): {problem.get('desired_outcome') or 'n/a'} "
                f"[evidence: {', '.join(problem.get('evidence', [])) or 'n/a'}]"
            )
    else:
        lines.append("- No explicit problem assessment recorded.")
    lines.extend(
        [
            "",
            "### Repair Strategy",
        ]
    )
    repair_strategy = (state["plan"] or {}).get("repair_strategy", [])
    if repair_strategy:
        for stage in repair_strategy:
            lines.append(
                f"- `{stage['stage']}`: {stage['goal']} "
                f"(tools: {', '.join(stage.get('selected_tools', [])) or 'none'}; rationale: {stage.get('rationale', 'n/a')})"
            )
    else:
        lines.append("- No staged repair strategy recorded.")
    lines.extend(
        [
            "",
            "### Feedback Applied",
        ]
    )
    feedback_applied = (state["plan"] or {}).get("feedback_applied", [])
    if feedback_applied:
        for item in feedback_applied:
            lines.append(f"- {item}")
    else:
        lines.append("- No explicit planner feedback notes recorded.")
    lines.extend(
        [
            "",
            "## Tool Trials",
        ]
    )
    tool_trials = state.get("tool_trial_history", [])
    if tool_trials:
        for item in tool_trials:
            lines.append(
                f"- Iteration {item.get('iteration')}: tool=`{item.get('tool')}` "
                f"attempted={item.get('attempted')} accepted={item.get('accepted')} "
                f"trials={len(item.get('trials', []))} reason={item.get('reason') or 'n/a'}"
            )
            region = item.get("selected_region")
            if isinstance(region, dict):
                lines.append(
                    f"selected region: `{region['x']},{region['y']},{region['width']},{region['height']}`"
                )
    else:
        lines.append("- No tool-trial history recorded.")
    lines.extend(
        [
            "",
            "## What Changed",
        ]
    )
    for record in state["executed_steps"]:
        mode = record.get("execution_mode", "global")
        mask_name = record.get("mask_name") or "none"
        lines.append(
            f"- `{record['tool']}` -> {record['status']} ({Path(record['output_path']).name}, mode={mode}, mask={mask_name})"
        )
    lines.extend(["", "## Step Snapshots"])
    if state["executed_steps"]:
        for record in state["executed_steps"]:
            step_path = Path(record["output_path"]).resolve()
            lines.extend(
                [
                    f"### Step {record['step_index']}: `{record['tool']}`",
                    f"- Status: `{record['status']}`",
                    f"- Mode: `{record.get('execution_mode', 'global')}`",
                    f"- Mask: `{record.get('mask_name') or 'none'}`",
                    f"- Params: `{json.dumps(record.get('params', {}), sort_keys=True)}`",
                    f"![{record['tool']}]({_relative_markdown_path(step_path, report_path)})",
                    "",
                ]
            )
    else:
        lines.extend(["- No executed steps recorded.", ""])
    lines.extend(
        [
            "",
            "## Agent Handoffs",
        ]
    )
    if state.get("agent_handoffs"):
        for handoff in state["agent_handoffs"]:
            lines.append(f"- `{handoff['from_agent']}` -> `{handoff['to_agent']}`: {handoff['summary']}")
    else:
        lines.append("- No structured handoffs recorded.")
    lines.extend(
        [
            "",
            "## Review",
            f"- Status: {review.get('status')}",
            f"- Prompt score: {review.get('prompt_score')}",
            f"- Artifact risk: {review.get('artifact_risk')}",
            f"- Patch metrics: {json.dumps(review.get('patch_metrics', {}), sort_keys=True)}",
            f"- Findings: {', '.join(review.get('findings', [])) or 'n/a'}",
            f"- Recommendations: {', '.join(review.get('recommendations', [])) or 'n/a'}",
            "",
            "## Human Review",
            f"- Status: {human_review.get('status', 'not_needed')}",
            f"- Reason: {human_review.get('reason', 'n/a')}",
            f"- Reasons: {', '.join(human_review.get('reasons', [])) or 'n/a'}",
            f"- Manual eval markdown: `{final.get('manual_eval_md') or 'n/a'}`",
            f"- Approval json: `{final.get('human_approval_json') or str(human_approval_path(state['run_dir']))}`",
            "",
            "## Final Decision",
            f"- Success: {final.get('success')}",
            f"- Output image: `{final.get('output_image')}`",
            f"- Review summary: {final.get('review_summary')}",
            f"- Stop reason: {final.get('stop_reason')}",
            "",
            "## Iteration Loop",
        ]
    )
    plan_history = state.get("plan_history", [])
    review_history = state.get("review_history", [])
    if plan_history or review_history:
        for plan_item in plan_history:
            iteration = plan_item.get("iteration")
            strategy = ", ".join(stage["stage"] for stage in plan_item.get("repair_strategy", [])[:3]) or "n/a"
            tools = ", ".join(step["tool"] for step in plan_item.get("steps", [])[:4]) or "none"
            matching_review = next((item for item in review_history if item.get("iteration") == iteration), None)
            if matching_review:
                review_summary = (
                    f"review={matching_review.get('status')} "
                    f"prompt_score={matching_review.get('prompt_score')} "
                    f"artifact_risk={matching_review.get('artifact_risk')}"
                )
            else:
                review_summary = "review=n/a"
            lines.append(f"- Iteration {iteration}: stages={strategy}; tools={tools}; {review_summary}")
    else:
        lines.append("- No iteration history recorded.")
    lines.extend(
        [
            "",
            "## Summary",
            _build_edit_summary_text(state),
            "",
            "## Observability",
            f"- Observation markdown: `{final.get('observation_md')}`",
            f"- Observation json: `{final.get('observation_json')}`",
            f"- Edit summary markdown: `{final.get('summary_md')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_observation_payload(state: WorkflowState) -> dict:
    return {
        "run_id": state["run_id"],
        "prompt": state["prompt"],
        "iteration": state["iteration"],
        "trace": state.get("observation_trace", []),
    }


def build_observation_markdown(state: WorkflowState) -> str:
    lines = [
        f"# VeriEdit Observation Trace: {state['run_id']}",
        "",
        "## Node Flow",
        "```mermaid",
        "flowchart TD",
        '    A["policy_check"] --> B["diagnose_inputs"] --> C["plan_edits"] --> D["trial_tools"] --> E["execute_plan"] --> F["review_result"] --> G["human_approval_gate"] --> H["decide_retry"]',
        "```",
        "",
        "## Trace Events",
    ]
    for event in state.get("observation_trace", []):
        if event["kind"] == "node":
            lines.append(
                f"- node `{event['node']}` {event['phase']} iteration {event['iteration']} summary={json.dumps(event.get('summary', {}), sort_keys=True)}"
            )
        elif event["kind"] == "handoff":
            lines.append(
                f"- handoff `{event['from_agent']}` -> `{event['to_agent']}` summary={event['summary']}"
            )
        else:
            lines.append(
                f"- tool `{event['tool']}` variant `{event['variant']}` status `{event['status']}` params={json.dumps(event['params'], sort_keys=True)}"
            )
    return "\n".join(lines) + "\n"


def build_edit_summary_markdown(state: WorkflowState) -> str:
    return "\n".join(
        [
            f"# Edit Summary: {state['run_id']}",
            "",
            _build_edit_summary_text(state),
            "",
        ]
    )


def _build_edit_summary_text(state: WorkflowState) -> str:
    review = state["review"] or {}
    plan = state["plan"] or {}
    human_review = state.get("human_review") or {}
    applied_tools = [step["tool"] for step in state["executed_steps"] if step["status"] == "ok"]
    findings = ", ".join(review.get("findings", [])[:4]) or "No notable findings."
    detected = ", ".join(item["problem"] for item in plan.get("detected_problems", [])[:4]) or "none"
    feedback = " ".join(plan.get("feedback_applied", [])[:2]) or "No feedback adjustments recorded."
    return (
        f"Prompt: {state['prompt']}\n\n"
        f"Objective: {plan.get('objective', 'n/a')}\n\n"
        f"Detected problems: {detected}\n\n"
        f"Applied tools: {', '.join(applied_tools) or 'none'}\n\n"
        f"Planner feedback: {feedback}\n\n"
        f"Review outcome: status={review.get('status')} prompt_score={review.get('prompt_score')} artifact_risk={review.get('artifact_risk')}\n\n"
        f"Human review: status={human_review.get('status', 'not_needed')} reason={human_review.get('reason', 'n/a')}\n\n"
        f"Highlights: {findings}"
    )


def finalize_outputs(state: WorkflowState) -> EditResult:
    payload = build_report_payload(state)
    final = state["final_result"]
    if final.get("human_approval_json") and not Path(final["human_approval_json"]).exists():
        write_json({"status": "pending", "notes": "", "timestamp": None}, final["human_approval_json"])
    write_json(build_observation_payload(state), state["final_result"]["observation_json"])
    write_text(build_observation_markdown(state), state["final_result"]["observation_md"])
    write_text(build_edit_summary_markdown(state), state["final_result"]["summary_md"])
    write_json(payload, state["final_result"]["report_json"])
    write_text(build_markdown_report(state), state["final_result"]["report_md"])
    if final.get("manual_eval_md"):
        build_manual_eval_markdown(
            source_image=state["source_image_path"],
            reference_image=state["reference_image_path"],
            result_image=final["output_image"] or state["current_image_path"],
            output_path=final["manual_eval_md"],
            prompt=state["prompt"],
            report_json=final["report_json"],
            observation_json=final["observation_json"],
            title=f"Manual Eval: {state['run_id']}",
            embed_images=False,
        )
    return EditResult(**state["final_result"], run_id=state["run_id"])


def _relative_markdown_path(target: Path, output_md_path: Path) -> str:
    return Path(os.path.relpath(str(target.resolve()), start=str(output_md_path.parent))).as_posix()
