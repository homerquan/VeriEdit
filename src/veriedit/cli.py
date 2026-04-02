from __future__ import annotations

import json
import shlex
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from veriedit import edit_image
from veriedit.config import WorkflowConfig
from veriedit.human_review import write_human_approval
from veriedit.io.loader import load_image
from veriedit.io.writer import save_image
from veriedit.manual_eval import build_manual_eval_from_run, build_manual_eval_markdown
from veriedit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from veriedit.schemas import EditRequest
from veriedit.tools import build_tool_registry
from veriedit.workflow import VeriEditWorkflow

app = typer.Typer(
    help="Non-generative image editing orchestration with an interactive shell.",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()

VERIEDIT_BANNER = r"""
__     __         _ ______    _ _ _
\ \   / /__ _ __ (_)  _ \ \  / /(_)
 \ \ / / _ \ '__|| | | | \ \/ / | |
  \ V /  __/ |   | | |_| |\  /  | |
   \_/ \___|_|   |_|____/  \/   |_|
"""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _start_repl()


@app.command()
def shell() -> None:
    _start_repl()


@app.command()
def edit(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
    prompt: str = typer.Option(..., "--prompt", help="Natural-language editing prompt."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True, help="Optional reference image."),
    output_folder: Optional[Path] = typer.Option(None, "--output-folder", file_okay=False, help="Folder where run directories should be created."),
    max_iterations: int = typer.Option(3, "--max-iterations", min=1, max=10),
    save_intermediates: bool = typer.Option(True, "--save-intermediates/--no-save-intermediates"),
    enable_human_approval: bool = typer.Option(True, "--human-approval/--no-human-approval"),
) -> None:
    _run_edit_command(
        input=input,
        prompt=prompt,
        reference=reference,
        output_folder=output_folder,
        max_iterations=max_iterations,
        save_intermediates=save_intermediates,
        enable_human_approval=enable_human_approval,
    )


@app.command()
def paint(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
    output: Path = typer.Option(..., "--output", help="Painted output path."),
    tool: str = typer.Option("paint", "--tool", help="Tool mode: paint, spot-heal, heal, clone, or stroke."),
    prompt: str = typer.Option("", "--prompt", help="Optional instruction to record or guide stroke-engine emphasis."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True, help="Optional reference/target image for stroke paint."),
    stroke: list[str] = typer.Option([], "--stroke", help="Stroke polyline as \"x1,y1 x2,y2 ...\". Repeat for multiple strokes."),
    strokes_file: Optional[Path] = typer.Option(None, "--strokes-file", exists=True, readable=True, help="Optional JSON file with stroke objects."),
    mask_box: list[str] = typer.Option([], "--mask-box", help="ROI box for stroke paint as x,y,width,height. Repeat for multiple boxes."),
    color: Optional[str] = typer.Option(None, "--color", help="Brush color as #RRGGBB or r,g,b. If omitted, sample from the first stroke point."),
    sample_color: Optional[str] = typer.Option(None, "--sample-color", help="Sample brush color from x,y in the source image."),
    source_point: Optional[str] = typer.Option(None, "--source-point", help="Healing/clone source point as x,y."),
    pen: str = typer.Option("soft", "--pen", help="Brush type: soft, round, or square."),
    blend_mode: str = typer.Option("normal", "--blend-mode", help="Healing mode: normal or replace."),
    size: int = typer.Option(8, "--size", min=1, max=128, help="Brush size in pixels."),
    opacity: float = typer.Option(0.65, "--opacity", min=0.0, max=1.0, help="Brush opacity."),
    feather: float = typer.Option(4.0, "--feather", min=0.0, help="Feather radius for healing/clone."),
    rotation: float = typer.Option(0.0, "--rotation", help="Clone/heal source rotation in degrees."),
    stroke_budget: int = typer.Option(12, "--stroke-budget", min=1, max=128, help="Stroke budget for --tool stroke."),
    candidate_count: int = typer.Option(18, "--candidate-count", min=4, max=64, help="Candidate strokes evaluated per iteration for --tool stroke."),
    engine_debug_dir: Optional[Path] = typer.Option(None, "--engine-debug-dir", file_okay=False, help="Optional directory for stroke-engine debug artifacts."),
    flip_horizontal: bool = typer.Option(False, "--flip-horizontal/--no-flip-horizontal"),
    flip_vertical: bool = typer.Option(False, "--flip-vertical/--no-flip-vertical"),
) -> None:
    _run_paint_command(
        input=input,
        output=output,
        tool=tool,
        prompt=prompt,
        reference=reference,
        stroke=stroke,
        strokes_file=strokes_file,
        mask_box=mask_box,
        color=color,
        sample_color=sample_color,
        source_point=source_point,
        pen=pen,
        blend_mode=blend_mode,
        size=size,
        opacity=opacity,
        feather=feather,
        rotation=rotation,
        stroke_budget=stroke_budget,
        candidate_count=candidate_count,
        engine_debug_dir=engine_debug_dir,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )


@app.command()
def inspect(
    input: Path = typer.Option(..., "--input", exists=True, readable=True, help="Source image path."),
) -> None:
    _run_inspect_command(input)


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
        result = _run_with_spinner(
            label=f"Editing {path.name}",
            phases=["policy", "diagnostics", "planning", "execution", "review", "reporting"],
            fn=lambda path=path: edit_image(
                source_image=str(path),
                prompt=prompt,
                reference_image=str(reference) if reference else None,
                output_path=str(target_dir / f"{path.stem}.edited.png"),
            ),
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
    artifact_root: Path = typer.Option(Path("/tmp/veriedit"), "--artifact-root", help="Artifact root for --run-id mode."),
    source: Optional[Path] = typer.Option(None, "--source", exists=True, readable=True, help="Source image path."),
    reference: Optional[Path] = typer.Option(None, "--reference", exists=True, readable=True, help="Reference image path."),
    result: Optional[Path] = typer.Option(None, "--result", exists=True, readable=True, help="Result image path."),
    report_json: Optional[Path] = typer.Option(None, "--report-json", exists=True, readable=True, help="Optional report.json path."),
    observation_json: Optional[Path] = typer.Option(None, "--observation-json", exists=True, readable=True, help="Optional observation trace json path."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Optional prompt override."),
    title: Optional[str] = typer.Option(None, "--title", help="Optional markdown title."),
    embed_images: bool = typer.Option(False, "--embed-images/--link-images", help="Use filesystem image links by default; enable embedding only if needed."),
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


@app.command()
def manual_approve(
    run_id: str = typer.Option(..., "--run-id", help="Workflow run identifier."),
    decision: str = typer.Option(..., "--decision", help="Manual decision: approved or rejected."),
    notes: str = typer.Option("", "--notes", help="Optional human review notes."),
    artifact_root: Path = typer.Option(Path("/tmp/veriedit"), "--artifact-root", help="Artifact root containing the run."),
) -> None:
    normalized = decision.strip().lower()
    if normalized not in {"approved", "rejected"}:
        raise typer.BadParameter("--decision must be 'approved' or 'rejected'.")
    run_dir = artifact_root / run_id
    if not run_dir.exists():
        raise typer.BadParameter(f"No run directory found for run_id={run_id}")
    approval_path = write_human_approval(run_dir, normalized, notes)
    console.print(f"Recorded manual decision: {normalized}")
    console.print(f"Approval JSON: {approval_path}")


def _start_repl() -> None:
    _print_banner()
    _print_shell_help()
    while True:
        try:
            raw = Prompt.ask("[bold cyan]veriedit[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold green]Exiting VeriEdit shell.[/bold green]")
            return
        if not raw:
            continue
        if raw.lower() in {"exit", "quit", ":q"}:
            console.print("[bold green]Exiting VeriEdit shell.[/bold green]")
            return
        if raw.lower() in {"help", "?"}:
            _print_shell_help()
            continue
        if raw.lower() == "edit":
            _interactive_edit_wizard()
            continue
        if raw.lower() == "paint":
            _interactive_paint_wizard()
            continue
        try:
            tokens = shlex.split(raw)
            app(prog_name="veriedit", args=tokens, standalone_mode=False)
        except typer.Abort:
            console.print("[yellow]Command aborted.[/yellow]")
        except typer.Exit:
            continue
        except Exception as exc:
            console.print(f"[red]Command failed:[/red] {exc}")


def _print_banner() -> None:
    title = Text(VERIEDIT_BANNER, style="bold cyan")
    console.print(
        Panel.fit(
            title,
            border_style="cyan",
            title="VeriEdit",
            subtitle="repl shell",
        )
    )
    console.print("[dim]Type `help` for commands, `edit` for guided editing, or `paint` for guided touch-up.[/dim]")


def _print_shell_help() -> None:
    table = Table(box=box.SIMPLE_HEAVY, title="Commands")
    table.add_column("Command", style="bold cyan")
    table.add_column("What It Does")
    table.add_row("edit", "Guided image-edit workflow with policy, review, and reports.")
    table.add_row("paint", "Guided brush tool for localized touch-up with soft/round/square pens.")
    table.add_row("paint --tool heal", "Use Photoshop-style healing or clone tools from explicit coordinates.")
    table.add_row("paint --tool stroke", "Use iterative planned strokes inside a selected ROI.")
    table.add_row("inspect --input <path>", "Show diagnostic metrics for one image.")
    table.add_row("report --run-id <id>", "Print a run report as JSON.")
    table.add_row("manual-eval ...", "Generate a self-contained markdown review sheet.")
    table.add_row("manual-approve ...", "Record a human approval or rejection for a run.")
    table.add_row("graph", "Show whether LangGraph is active.")
    table.add_row("quit", "Exit the shell.")
    console.print(table)


def _interactive_edit_wizard() -> None:
    input_path = Path(Prompt.ask("Source image path")).expanduser()
    prompt = Prompt.ask("Editing prompt")
    reference_raw = Prompt.ask("Reference image path (optional)", default="", show_default=False).strip()
    output_folder_raw = Prompt.ask("Output folder (optional, default /tmp/veriedit)", default="", show_default=False).strip()
    max_iterations = IntPrompt.ask("Max iterations", default=3)
    enable_human_approval = Confirm.ask("Enable human approval for ambiguous results?", default=True)
    _run_edit_command(
        input=input_path,
        prompt=prompt,
        reference=Path(reference_raw).expanduser() if reference_raw else None,
        output_folder=Path(output_folder_raw).expanduser() if output_folder_raw else None,
        max_iterations=max_iterations,
        save_intermediates=True,
        enable_human_approval=enable_human_approval,
    )


def _interactive_paint_wizard() -> None:
    input_path = Path(Prompt.ask("Source image path")).expanduser()
    output_path = Path(Prompt.ask("Output image path")).expanduser()
    tool = Prompt.ask("Tool", choices=["paint", "spot-heal", "heal", "clone", "stroke"], default="paint")
    pen = Prompt.ask("Pen type", choices=["soft", "round", "square"], default="soft") if tool == "paint" else "soft"
    size = IntPrompt.ask("Brush size", default=8)
    opacity = FloatPrompt.ask("Opacity", default=0.65)
    color = (
        Prompt.ask("Brush color as #RRGGBB or r,g,b (leave blank to sample first point)", default="", show_default=False).strip()
        if tool == "paint"
        else ""
    )
    source_point = ""
    blend_mode = "normal"
    feather = 4.0
    rotation = 0.0
    stroke_budget = 12
    candidate_count = 18
    flip_horizontal = False
    flip_vertical = False
    if tool in {"heal", "clone"}:
        source_point = Prompt.ask("Source point as x,y")
        blend_mode = Prompt.ask("Blend mode", choices=["normal", "replace"], default="normal" if tool == "heal" else "replace")
        feather = FloatPrompt.ask("Feather", default=4.0)
        rotation = FloatPrompt.ask("Rotation", default=0.0)
        flip_horizontal = Confirm.ask("Flip source horizontally?", default=False)
        flip_vertical = Confirm.ask("Flip source vertically?", default=False)
    mask_boxes: list[str] = []
    if tool == "stroke":
        stroke_budget = IntPrompt.ask("Stroke budget", default=12)
        candidate_count = IntPrompt.ask("Candidate count", default=18)
        console.print("[dim]Enter mask boxes as `x,y,width,height`. Press Enter on a blank line when finished.[/dim]")
        while True:
            box_line = Prompt.ask("Mask box", default="", show_default=False).strip()
            if not box_line:
                break
            mask_boxes.append(box_line)
    stroke_specs: list[str] = []
    console.print("[dim]Enter stroke points as `x1,y1 x2,y2 ...`. Press Enter on a blank line when finished.[/dim]")
    while True:
        stroke_line = Prompt.ask("Stroke", default="", show_default=False).strip()
        if not stroke_line:
            break
        stroke_specs.append(stroke_line)
    if tool != "stroke" and not stroke_specs:
        console.print("[yellow]No strokes entered, nothing to paint.[/yellow]")
        return
    _run_paint_command(
        input=input_path,
        output=output_path,
        tool=tool,
        prompt="",
        reference=None,
        stroke=stroke_specs,
        strokes_file=None,
        mask_box=mask_boxes,
        color=color or None,
        sample_color=None,
        source_point=source_point or None,
        pen=pen,
        blend_mode=blend_mode,
        size=size,
        opacity=opacity,
        feather=feather,
        rotation=rotation,
        stroke_budget=stroke_budget,
        candidate_count=candidate_count,
        engine_debug_dir=None,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )


def _run_edit_command(
    *,
    input: Path,
    prompt: str,
    reference: Optional[Path],
    output_folder: Optional[Path],
    max_iterations: int,
    save_intermediates: bool,
    enable_human_approval: bool,
) -> None:
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=output_folder or WorkflowConfig().artifact_root))
    result = _run_with_spinner(
        label=f"Editing {input.name}",
        phases=["policy check", "diagnostics", "planning", "execution", "review", "approval gate", "reporting"],
        fn=lambda: workflow.run(
            EditRequest(
                source_image=str(input),
                prompt=prompt,
                reference_image=str(reference) if reference else None,
                output_path=None,
                max_iterations=max_iterations,
                save_intermediates=save_intermediates,
                enable_human_approval=enable_human_approval,
            )
        ),
    )
    _print_result_summary(result)


def _run_paint_command(
    *,
    input: Path,
    output: Path,
    tool: str,
    prompt: str,
    reference: Optional[Path],
    stroke: list[str],
    strokes_file: Optional[Path],
    mask_box: list[str],
    color: Optional[str],
    sample_color: Optional[str],
    source_point: Optional[str],
    pen: str,
    blend_mode: str,
    size: int,
    opacity: float,
    feather: float,
    rotation: float,
    stroke_budget: int,
    candidate_count: int,
    engine_debug_dir: Optional[Path],
    flip_horizontal: bool,
    flip_vertical: bool,
) -> None:
    image, _ = load_image(input)
    reference_image = None
    if reference is not None:
        reference_image, _ = load_image(reference)
    strokes = _load_strokes(stroke, strokes_file)
    registry = build_tool_registry()
    tool_name = _resolve_paint_tool_name(tool)
    if tool_name != "stroke_paint" and not strokes:
        raise typer.BadParameter("Provide at least one --stroke or a --strokes-file with stroke objects.")
    payload, resolved_color = _build_paint_payload(
        image=image,
        strokes=strokes,
        mask_boxes=mask_box,
        tool_name=tool_name,
        color=color,
        sample_color=sample_color,
        source_point=source_point,
        pen=pen,
        blend_mode=blend_mode,
        size=size,
        opacity=opacity,
        feather=feather,
        rotation=rotation,
        stroke_budget=stroke_budget,
        candidate_count=candidate_count,
        prompt=prompt,
        engine_debug_dir=engine_debug_dir,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
    )

    def _apply() -> tuple[Any, dict[str, Any]]:
        return registry.get(tool_name).operation(image, payload, reference_image)

    painted, details = _run_with_spinner(
        label=f"Painting {input.name}",
        phases=["preparing canvas", "applying strokes", "saving output"],
        fn=_apply,
    )
    save_image(painted, output)
    table = Table(title="Paint Result", box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Tool", tool_name)
    table.add_row("Output", str(output))
    table.add_row("Stroke count", str(details.get("stroke_count", details.get("point_count", details.get("target_count", 0)))))
    if prompt:
        table.add_row("Prompt", prompt)
    if tool_name == "paint_strokes":
        table.add_row("Pen types", ", ".join(details.get("pen_types", [])) or pen)
        table.add_row("Brush color", str(tuple(resolved_color)))
    if tool_name == "stroke_paint":
        table.add_row("Backend", str(details.get("backend", "stroke_paint")))
        table.add_row("Target source", str(details.get("target_source")))
        if details.get("debug_dir"):
            table.add_row("Engine debug", str(details.get("debug_dir")))
    if tool_name in {"healing_brush", "clone_source_paint"}:
        table.add_row("Blend mode", str(details.get("mode")))
        table.add_row("Source rotation", str(details.get("rotation")))
    console.print(table)


def _run_inspect_command(input: Path) -> None:
    image, metadata = load_image(input)
    summary = summarize_image_quality(image, metadata)
    style = style_profile_from_image(image)
    table = Table(title=f"Inspection: {input.name}", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    for key, value in {**summary, **{f"style_{k}": v for k, v in style.items()}}.items():
        table.add_row(str(key), f"{value}")
    console.print(table)


def _print_result_summary(result: Any) -> None:
    table = Table(title="Edit Result", box=box.SIMPLE_HEAVY)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Run ID", str(result.run_id))
    table.add_row("Success", str(result.success))
    table.add_row("Output image", str(result.output_image))
    table.add_row("Report JSON", str(result.report_json))
    table.add_row("Report Markdown", str(result.report_md))
    table.add_row("Summary", str(result.review_summary))
    table.add_row("Human review", str(result.human_review_status))
    table.add_row("Manual eval", str(result.manual_eval_md))
    table.add_row("Approval JSON", str(result.human_approval_json))
    console.print(table)


def _run_with_spinner(*, label: str, phases: list[str], fn):
    queue: Queue[tuple[str, Any]] = Queue()

    def worker() -> None:
        try:
            queue.put(("result", fn()))
        except Exception as exc:  # pragma: no cover - exercised by CLI behavior
            queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    with console.status(f"[bold green]{label}[/bold green]", spinner="dots12") as status:
        index = 0
        while thread.is_alive():
            status.update(f"[bold green]{label}[/bold green] [dim]{phases[index % len(phases)]}[/dim]")
            time.sleep(0.12)
            index += 1
    kind, payload = queue.get()
    if kind == "error":
        raise payload
    return payload


def _load_strokes(stroke_specs: list[str], strokes_file: Optional[Path]) -> list[dict[str, Any]]:
    strokes: list[dict[str, Any]] = []
    for spec in stroke_specs:
        strokes.append({"points": _parse_stroke_spec(spec)})
    if strokes_file:
        payload = json.loads(strokes_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("strokes", [])
        if not isinstance(payload, list):
            raise typer.BadParameter("Stroke file must contain a JSON list or an object with a `strokes` list.")
        for item in payload:
            if not isinstance(item, dict):
                continue
            points = item.get("points", [])
            item["points"] = [[int(point[0]), int(point[1])] for point in points if isinstance(point, list) and len(point) == 2]
            strokes.append(item)
    return [stroke for stroke in strokes if stroke.get("points")]


def _resolve_paint_tool_name(tool: str) -> str:
    normalized = tool.strip().lower()
    mapping = {
        "paint": "paint_strokes",
        "spot-heal": "spot_healing_brush",
        "spot_heal": "spot_healing_brush",
        "heal": "healing_brush",
        "clone": "clone_source_paint",
        "stroke": "stroke_paint",
        "stroke-engine": "stroke_paint",
        "stroke_engine": "stroke_paint",
    }
    if normalized not in mapping:
        raise typer.BadParameter("--tool must be one of: paint, spot-heal, heal, clone, stroke, stroke-engine.")
    return mapping[normalized]


def _build_paint_payload(
    *,
    image: Any,
    strokes: list[dict[str, Any]],
    mask_boxes: list[str],
    tool_name: str,
    color: Optional[str],
    sample_color: Optional[str],
    source_point: Optional[str],
    pen: str,
    blend_mode: str,
    size: int,
    opacity: float,
    feather: float,
    rotation: float,
    stroke_budget: int,
    candidate_count: int,
    prompt: str,
    engine_debug_dir: Optional[Path],
    flip_horizontal: bool,
    flip_vertical: bool,
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    resolved_color = _resolve_brush_color(image, strokes, color, sample_color)
    if tool_name == "paint_strokes":
        for item in strokes:
            item.setdefault("pen", pen)
            item.setdefault("size", size)
            item.setdefault("opacity", opacity)
            item.setdefault("color", list(resolved_color))
        return {"strokes": strokes, "pen": pen, "size": size, "opacity": opacity, "color": list(resolved_color)}, resolved_color
    points = [stroke["points"][0] for stroke in strokes if stroke.get("points")]
    if tool_name == "spot_healing_brush":
        return {"points": points, "radius": size}, resolved_color
    if tool_name == "stroke_paint":
        return (
            {
                "mask_boxes": [_parse_mask_box(box) for box in mask_boxes],
                "points": points,
                "radius": size,
                "stroke_budget": stroke_budget,
                "candidate_count": candidate_count,
                "min_size": max(1, size // 3),
                "max_size": max(size, int(size * 1.5)),
                "opacity": opacity,
                "pen": pen,
                "prompt": prompt,
                "debug_dir": str(engine_debug_dir) if engine_debug_dir else None,
            },
            resolved_color,
        )
    if not source_point:
        raise typer.BadParameter("--source-point is required for heal and clone tools.")
    return (
        {
            "source_point": list(_parse_xy(source_point)),
            "target_points": points,
            "radius": size,
            "opacity": opacity,
            "feather": feather,
            "mode": blend_mode,
            "rotation": rotation,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
        },
        resolved_color,
    )


def _parse_stroke_spec(spec: str) -> list[list[int]]:
    points: list[list[int]] = []
    for token in spec.split():
        coords = token.split(",")
        if len(coords) != 2:
            raise typer.BadParameter(f"Invalid stroke point `{token}`. Use x,y.")
        points.append([int(coords[0]), int(coords[1])])
    if not points:
        raise typer.BadParameter("A stroke must include at least one point.")
    return points


def _resolve_brush_color(
    image: Any,
    strokes: list[dict[str, Any]],
    color: Optional[str],
    sample_color: Optional[str],
) -> tuple[int, int, int]:
    if sample_color:
        x, y = _parse_xy(sample_color)
        return _sample_rgb(image, x, y)
    if color:
        return _parse_color(color)
    if strokes:
        first_point = strokes[0]["points"][0]
        return _sample_rgb(image, int(first_point[0]), int(first_point[1]))
    center_y = image.shape[0] // 2
    center_x = image.shape[1] // 2
    return _sample_rgb(image, center_x, center_y)


def _parse_color(value: str) -> tuple[int, int, int]:
    raw = value.strip()
    if raw.startswith("#") and len(raw) == 7:
        return tuple(int(raw[index : index + 2], 16) for index in range(1, 7, 2))
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter("Color must be #RRGGBB or r,g,b.")
    return tuple(max(0, min(255, int(part))) for part in parts)


def _parse_xy(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise typer.BadParameter("Expected coordinates as x,y.")
    return int(parts[0]), int(parts[1])


def _parse_mask_box(value: str) -> dict[str, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("Expected mask box as x,y,width,height.")
    x, y, width, height = (int(part) for part in parts)
    return {"x": x, "y": y, "width": width, "height": height}


def _sample_rgb(image: Any, x: int, y: int) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    clipped_x = max(0, min(width - 1, x))
    clipped_y = max(0, min(height - 1, y))
    sample = image[clipped_y, clipped_x]
    return int(sample[0]), int(sample[1]), int(sample[2])


if __name__ == "__main__":
    app()
