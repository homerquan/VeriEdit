from __future__ import annotations

import json
from pathlib import Path

from veriedit.io.writer import write_json, write_text
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
        "executed_steps": state["executed_steps"],
        "observation_trace": state.get("observation_trace", []),
        "review": state["review"],
        "retry_decision": state["retry_decision"],
        "final_result": state["final_result"],
        "intermediate_paths": state["intermediate_paths"],
        "logs": state["logs"],
    }


def build_markdown_report(state: WorkflowState) -> str:
    final = state["final_result"] or {}
    review = state["review"] or {}
    lines = [
        f"# VeriEdit Report: {state['run_id']}",
        "",
        "## Request",
        f"- Source image: `{state['source_image_path']}`",
        f"- Reference image: `{state['reference_image_path']}`" if state["reference_image_path"] else "- Reference image: none",
        f"- Prompt: {state['prompt']}",
        "",
        "## Policy",
        f"- Status: {state['policy_status'].get('status')}",
        f"- Risk level: {state['policy_status'].get('risk_level')}",
        f"- Constraints: {', '.join(state['policy_status'].get('constraints', []))}",
        "",
        "## Plan",
        f"- Objective: {(state['plan'] or {}).get('objective', 'n/a')}",
        f"- Acceptance: {', '.join((state['plan'] or {}).get('acceptance', [])) or 'n/a'}",
        f"- Diagnostic board: `{(state.get('diagnostic_artifacts') or {}).get('regions_board', 'n/a')}`",
        "",
        "## What Changed",
    ]
    for record in state["executed_steps"]:
        lines.append(f"- `{record['tool']}` -> {record['status']} ({Path(record['output_path']).name})")
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
        "## Final Decision",
        f"- Success: {final.get('success')}",
        f"- Output image: `{final.get('output_image')}`",
        f"- Review summary: {final.get('review_summary')}",
        f"- Stop reason: {final.get('stop_reason')}",
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
        '    A["policy_check"] --> B["diagnose_inputs"] --> C["plan_edits"] --> D["execute_plan"] --> E["review_result"] --> F["decide_retry"]',
        "```",
        "",
        "## Trace Events",
    ]
    for event in state.get("observation_trace", []):
        if event["kind"] == "node":
            lines.append(
                f"- node `{event['node']}` {event['phase']} iteration {event['iteration']} summary={json.dumps(event.get('summary', {}), sort_keys=True)}"
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
    applied_tools = [step["tool"] for step in state["executed_steps"] if step["status"] == "ok"]
    findings = ", ".join(review.get("findings", [])[:4]) or "No notable findings."
    return (
        f"Prompt: {state['prompt']}\n\n"
        f"Objective: {plan.get('objective', 'n/a')}\n\n"
        f"Applied tools: {', '.join(applied_tools) or 'none'}\n\n"
        f"Review outcome: status={review.get('status')} prompt_score={review.get('prompt_score')} artifact_risk={review.get('artifact_risk')}\n\n"
        f"Highlights: {findings}"
    )


def finalize_outputs(state: WorkflowState) -> EditResult:
    payload = build_report_payload(state)
    write_json(build_observation_payload(state), state["final_result"]["observation_json"])
    write_text(build_observation_markdown(state), state["final_result"]["observation_md"])
    write_text(build_edit_summary_markdown(state), state["final_result"]["summary_md"])
    write_json(payload, state["final_result"]["report_json"])
    write_text(build_markdown_report(state), state["final_result"]["report_md"])
    return EditResult(**state["final_result"], run_id=state["run_id"])
