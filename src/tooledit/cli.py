from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tooledit import edit_image
from tooledit.config import WorkflowConfig
from tooledit.io.loader import load_image
from tooledit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from tooledit.workflow import ToolEditWorkflow

app = typer.Typer(help="Non-generative image editing orchestration.")
console = Console()


@app.command()
def edit(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
    prompt: str = typer.Option(..., "--prompt", help="Natural-language editing prompt."),
    reference: Path | None = typer.Option(None, "--reference", exists=True, readable=True, help="Optional reference image."),
    output: Path | None = typer.Option(None, "--output", help="Output image path."),
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
    reference: Path | None = typer.Option(None, "--reference", exists=True, readable=True),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    target_dir = output_dir or (input_dir / "tooledit_batch")
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
    workflow = ToolEditWorkflow()
    console.print("LangGraph available." if workflow.graph is not None else "LangGraph dependency unavailable; fallback loop is active.")


if __name__ == "__main__":
    app()
