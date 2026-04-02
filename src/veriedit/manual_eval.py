from __future__ import annotations

import base64
import json
import os
import mimetypes
from pathlib import Path
from typing import Any, Optional

from veriedit.io.writer import write_text


def build_manual_eval_markdown(
    *,
    source_image: str,
    result_image: str,
    output_path: str,
    prompt: str | None = None,
    reference_image: str | None = None,
    report_json: str | None = None,
    observation_json: str | None = None,
    title: str | None = None,
    embed_images: bool = True,
) -> Path:
    source_path = Path(source_image).resolve()
    result_path = Path(result_image).resolve()
    reference_path = Path(reference_image).resolve() if reference_image else None
    output_md_path = Path(output_path).resolve()
    report_data = _read_json(report_json)
    observation_data = _read_json(observation_json)

    if report_data and not prompt:
        prompt = report_data.get("request", {}).get("prompt")
    if report_data and not reference_path:
        ref = report_data.get("request", {}).get("reference_image") or report_data.get("reference_image_path")
        if ref:
            reference_path = Path(ref).resolve()
    if report_data and not observation_data:
        final = report_data.get("final_result", {})
        trace_path = final.get("observation_json")
        if trace_path and Path(trace_path).exists():
            observation_data = json.loads(Path(trace_path).read_text(encoding="utf-8"))

    markdown = _render_markdown(
        source_path=source_path,
        result_path=result_path,
        reference_path=reference_path,
        output_md_path=output_md_path,
        prompt=prompt,
        title=title or _default_title(report_data, source_path),
        report_data=report_data,
        observation_data=observation_data,
        embed_images=embed_images,
    )
    return write_text(markdown, output_md_path)


def build_manual_eval_from_run(
    run_id: str,
    artifact_root: str | Path = "runs",
    output_path: str | None = None,
    embed_images: bool = True,
) -> Path:
    run_dir = Path(artifact_root) / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No report.json found for run_id={run_id} in {run_dir}")
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    final = report_data.get("final_result", {})
    return build_manual_eval_markdown(
        source_image=report_data.get("source_image_path") or report_data["request"]["source_image"],
        reference_image=report_data.get("reference_image_path") or report_data.get("request", {}).get("reference_image"),
        result_image=final["output_image"],
        output_path=output_path or str(run_dir / "manual_eval.md"),
        prompt=report_data.get("request", {}).get("prompt"),
        report_json=str(report_path),
        observation_json=final.get("observation_json"),
        title=f"Manual Eval: {run_id}",
        embed_images=embed_images,
    )


def _render_markdown(
    *,
    source_path: Path,
    result_path: Path,
    reference_path: Optional[Path],
    output_md_path: Path,
    prompt: Optional[str],
    title: str,
    report_data: Optional[dict[str, Any]],
    observation_data: Optional[dict[str, Any]],
    embed_images: bool,
) -> str:
    review = (report_data or {}).get("review", {})
    plan = (report_data or {}).get("plan", {})
    executed_steps = (report_data or {}).get("executed_steps", [])
    trace = (observation_data or {}).get("trace", [])

    lines = [f"# {title}", ""]
    if prompt:
        lines.extend(["## Prompt", prompt, ""])

    lines.extend(["## Images", ""])
    lines.append("### Source")
    lines.append(f"![Source]({_image_markdown_src(source_path, output_md_path, embed_images)})")
    lines.append("")
    if reference_path:
        lines.append("### Reference")
        lines.append(f"![Reference]({_image_markdown_src(reference_path, output_md_path, embed_images)})")
        lines.append("")
    lines.append("### Result")
    lines.append(f"![Result]({_image_markdown_src(result_path, output_md_path, embed_images)})")
    lines.append("")

    if report_data:
        lines.extend(
            [
                "## Review Summary",
                f"- Status: {review.get('status')}",
                f"- Prompt score: {review.get('prompt_score')}",
                f"- Artifact risk: {review.get('artifact_risk')}",
                f"- Patch metrics: `{json.dumps(review.get('patch_metrics', {}), sort_keys=True)}`",
                f"- Findings: {', '.join(review.get('findings', [])) or 'n/a'}",
                f"- Recommendations: {', '.join(review.get('recommendations', [])) or 'n/a'}",
                "",
                "## Plan",
                f"- Objective: {plan.get('objective', 'n/a')}",
                f"- Acceptance: {', '.join(plan.get('acceptance', [])) or 'n/a'}",
                "",
            ]
        )

    lines.append("## Tool Usage")
    if executed_steps:
        for step in executed_steps:
            notes = "; ".join(step.get("notes", []))
            lines.append(
                f"- step {step['step_index']}: `{step['tool']}` status=`{step['status']}` params=`{json.dumps(step.get('params', {}), sort_keys=True)}` {notes}".rstrip()
            )
    else:
        lines.append("- No executed steps available.")
    lines.append("")

    lines.extend(["## Trace", "```mermaid", "flowchart TD"])
    flow_nodes = [event["node"] for event in trace if event.get("kind") == "node" and event.get("phase") == "start"]
    if flow_nodes:
        unique_nodes = []
        for node in flow_nodes:
            if node not in unique_nodes:
                unique_nodes.append(node)
        if len(unique_nodes) == 1:
            lines.append(f'    A["{unique_nodes[0]}"]')
        else:
            for index, node in enumerate(unique_nodes):
                node_id = chr(ord("A") + index)
                if index == 0:
                    lines.append(f'    {node_id}["{node}"]')
                else:
                    prev_id = chr(ord("A") + index - 1)
                    lines.append(f'    {prev_id} --> {node_id}["{node}"]')
    else:
        lines.append('    A["No trace available"]')
    lines.extend(["```", ""])

    lines.append("### Trace Events")
    if trace:
        for event in trace:
            if event.get("kind") == "node":
                lines.append(
                    f"- node `{event['node']}` {event['phase']} iteration {event['iteration']} summary=`{json.dumps(event.get('summary', {}), sort_keys=True)}`"
                )
            else:
                lines.append(
                    f"- tool `{event['tool']}` variant `{event['variant']}` status `{event['status']}` params=`{json.dumps(event.get('params', {}), sort_keys=True)}` metrics=`{json.dumps(event.get('metrics', {}), sort_keys=True)}`"
                )
    else:
        lines.append("- No trace events available.")
    lines.append("")

    return "\n".join(lines)


def _read_json(path: str | None) -> Optional[dict[str, Any]]:
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _default_title(report_data: Optional[dict[str, Any]], source_path: Path) -> str:
    if report_data and report_data.get("run_id"):
        return f"Manual Eval: {report_data['run_id']}"
    return f"Manual Eval: {source_path.stem}"


def _relative_markdown_path(target: Path, output_md_path: Path) -> str:
    return Path(os.path.relpath(str(target.resolve()), start=str(output_md_path.parent))).as_posix()


def _image_markdown_src(target: Path, output_md_path: Path, embed_images: bool) -> str:
    if not embed_images:
        return _relative_markdown_path(target, output_md_path)
    mime_type = mimetypes.guess_type(str(target))[0] or "image/png"
    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
