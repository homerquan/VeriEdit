from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from veriedit import edit_image
from veriedit.config import WorkflowConfig
from veriedit.io.loader import load_image
from veriedit.manual_eval import build_manual_eval_from_run, build_manual_eval_markdown
from veriedit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from veriedit.workflow import VeriEditWorkflow

app = typer.Typer(help="Non-generative image editing orchestration.")
console = Console()


@app.command()
def edit(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
    prompt: str = typer.Option(..., "--prompt", help="Natural-language editing prompt."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True, help="Optional reference image."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output image path."),
    max_iterations: int = typer.Option(3, "--max-iterations", min=1, max=10),
    save_intermediates: bool = typer.Option(True, "--save-intermediates/--no-save-intermediates"),
) -> None:
    result = edit_image(
        source_image=str(input),
        prompt=prompt,
        reference_image=str(reference) if reference else None,
        output_path=str(output) if output else None,
        max_iterations=max_iterations,
        save_intermediates=save_intermediates,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Success: {result.success}")
    console.print(f"Output image: {result.output_image}")
    console.print(f"Report JSON: {result.report_json}")
    console.print(f"Report Markdown: {result.report_md}")
    console.print(f"Summary: {result.review_summary}")


@app.command()
def inspect(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
) -> None:
    image, metadata = load_image(input)
    summary = summarize_image_quality(image, metadata)
    style = style_profile_from_image(image)
    table = Table(title=f"Inspection: {input.name}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in {**summary, **{f"style_{k}": v for k, v in style.items()}}.items():
        table.add_row(str(key), f"{value}")
    console.print(table)


@app.command()
def batch(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, readable=True),
    prompt: str = typer.Option(..., "--prompt"),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
) -> None:
    target_dir = output_dir or (input_dir / "veriedit_batch")
    target_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str | bool | None]] = []
    for path in sorted(p for p in input_dir.iterdir() if p.is_file()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            continue
        result = edit_image(
            source_image=str(path),
            prompt=prompt,
            reference_image=str(reference) if reference else None,
            output_path=str(target_dir / f"{path.stem}.edited.png"),
        )
        results.append({"input": str(path), "success": result.success, "output": result.output_image})
    console.print_json(json.dumps(results))


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Workflow run identifier."),
) -> None:
    config = WorkflowConfig()
    report_path = config.artifact_root / run_id / "report.json"
    if not report_path.exists():
        raise typer.BadParameter(f"No report found for run_id={run_id}")
    console.print_json(report_path.read_text(encoding="utf-8"))


@app.command()
def graph() -> None:
    workflow = VeriEditWorkflow()
    console.print("LangGraph available." if workflow.graph is not None else "LangGraph dependency unavailable; fallback loop is active.")


@app.command()
def manual_eval(
    output: Path = typer.Option(..., "--output", help="Output markdown path."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Existing workflow run id."),
    artifact_root: Path = typer.Option(Path("runs"), "--artifact-root", help="Artifact root for --run-id mode."),
    source: Optional[Path] = typer.Option(None, "--source", exists=True, readable=True, help="Source image path."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True, help="Reference image path."),
    result: Optional[Path] = typer.Option(None, "--result", exists=True, readable=True, help="Result image path."),
    report_json: Optional[Path] = typer.Option(None, "--report-json", exists=True, readable=True, help="Optional report.json path."),
    observation_json: Optional[Path] = typer.Option(None, "--observation-json", exists=True, readable=True, help="Optional observation trace json path."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Optional prompt override."),
    title: Optional[str] = typer.Option(None, "--title", help="Optional markdown title."),
    embed_images: bool = typer.Option(True, "--embed-images/--link-images", help="Embed images as data URIs for portable markdown previews."),
) -> None:
    if run_id:
        markdown_path = build_manual_eval_from_run(
            run_id,
            artifact_root=artifact_root,
            output_path=str(output),
            embed_images=embed_images,
        )
    else:
        if not source or not result:
            raise typer.BadParameter("Either --run-id or both --source and --result are required.")
        markdown_path = build_manual_eval_markdown(
            source_image=str(source),
            reference_image=str(reference) if reference else None,
            result_image=str(result),
            report_json=str(report_json) if report_json else None,
            observation_json=str(observation_json) if observation_json else None,
            prompt=prompt,
            title=title,
            output_path=str(output),
            embed_images=embed_images,
        )
    console.print(f"Manual eval markdown: {markdown_path}")


if __name__ == "__main__":
    app()
