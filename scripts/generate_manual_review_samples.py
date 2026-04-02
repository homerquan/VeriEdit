from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from veriedit.config import WorkflowConfig
from veriedit.schemas import EditRequest
from veriedit.workflow import VeriEditWorkflow


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"
OUTPUT_DIR = ROOT / "manual_review"
RUNS_DIR = OUTPUT_DIR / "runs"


def load_manifest() -> list[dict[str, str]]:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (245, 243, 238))
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - thumbnail.width) // 2
    y = (size[1] - thumbnail.height) // 2
    canvas.paste(thumbnail, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    x0, y0, x1, _ = box
    draw.rectangle((x0, y0, x1, y0 + 28), fill=(34, 38, 45))
    draw.text((x0 + 10, y0 + 6), text, fill=(250, 250, 250), font=font)


def make_board(name: str, source: Path, reference: Path, output: Path) -> Path:
    panel_size = (380, 380)
    gap = 18
    margin = 24
    label_h = 28
    board_size = (margin * 2 + panel_size[0] * 3 + gap * 2, margin * 2 + panel_size[1] + label_h)
    board = Image.new("RGB", board_size, (232, 228, 220))
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()

    entries = [("Source", source), ("Reference", reference), ("Output", output)]
    for index, (label, path) in enumerate(entries):
        x = margin + index * (panel_size[0] + gap)
        y = margin + label_h
        box = (x, margin, x + panel_size[0], y + panel_size[1])
        panel = fit_panel(load_image(path), panel_size)
        board.paste(panel, (x, y))
        draw.rounded_rectangle((x, y, x + panel_size[0], y + panel_size[1]), radius=10, outline=(180, 174, 164), width=2)
        draw_label(draw, box, label, font)

    title = f"VeriEdit Manual Review: {name}"
    draw.text((margin, 8), title, fill=(48, 42, 34), font=font)
    output_path = OUTPUT_DIR / f"{name}_board.png"
    board.save(output_path)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=RUNS_DIR))
    manifest = load_manifest()
    summary: list[dict[str, str | bool | float | None]] = []

    for record in manifest:
        source = ROOT / record["source"]
        reference = ROOT / record["reference"]
        output = OUTPUT_DIR / f"{record['name']}_edited.png"
        result = workflow.run(
            EditRequest(
                source_image=str(source),
                reference_image=str(reference),
                prompt=record["prompt"],
                output_path=str(output),
                max_iterations=2,
                save_intermediates=False,
            )
        )
        board = make_board(record["name"], source, reference, output)
        summary.append(
            {
                "name": record["name"],
                "prompt": record["prompt"],
                "success": result.success,
                "output_image": str(output.relative_to(ROOT)),
                "board_image": str(board.relative_to(ROOT)),
                "report_json": result.report_json,
                "report_md": result.report_md,
                "summary_md": result.summary_md,
                "observation_md": result.observation_md,
                "review_summary": result.review_summary,
            }
        )

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Manual Review Samples", ""]
    for item in summary:
        lines.extend(
            [
                f"## {item['name']}",
                f"- Prompt: {item['prompt']}",
                f"- Success: {item['success']}",
                f"- Review summary: {item['review_summary']}",
                f"- Board: `{item['board_image']}`",
                f"- Output: `{item['output_image']}`",
                f"- Report: `{item['report_md']}`",
                f"- Edit summary: `{item['summary_md']}`",
                f"- Observation trace: `{item['observation_md']}`",
                "",
            ]
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
