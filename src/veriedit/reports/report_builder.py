from __future__ import annotations

from pathlib import Path

from veriedit.io.writer import write_json, write_text
from veriedit.schemas import EditResult, FinalResult, WorkflowState


def build_report_payload(state: WorkflowState) -> dict:
    return {
        "run_id": state["run_id"],
        "request": state["request"],
        "policy_status": state["policy_status"],
        "diagnostics": state["diagnostics"],
        "style_profile": state["style_profile"],
        "plan": state["plan"],
        "executed_steps": state["executed_steps"],
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
            f"- Findings: {', '.join(review.get('findings', [])) or 'n/a'}",
            f"- Recommendations: {', '.join(review.get('recommendations', [])) or 'n/a'}",
            "",
            "## Final Decision",
            f"- Success: {final.get('success')}",
            f"- Output image: `{final.get('output_image')}`",
            f"- Review summary: {final.get('review_summary')}",
            f"- Stop reason: {final.get('stop_reason')}",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_outputs(state: WorkflowState) -> EditResult:
    payload = build_report_payload(state)
    write_json(payload, state["final_result"]["report_json"])
    write_text(build_markdown_report(state), state["final_result"]["report_md"])
    return EditResult(**state["final_result"], run_id=state["run_id"])
