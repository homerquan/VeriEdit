from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from tooledit.schemas import RunArtifacts


def ensure_run_artifacts(
    artifact_root: Path,
    source_image: str,
    output_path: str | None = None,
    reference_image: str | None = None,
) -> RunArtifacts:
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_copy = run_dir / Path(source_image).name
    shutil.copy2(source_image, source_copy)
    reference_copy = None
    if reference_image:
        reference_copy = run_dir / Path(reference_image).name
        shutil.copy2(reference_image, reference_copy)
    resolved_output = Path(output_path) if output_path else run_dir / "edited.png"
    if not resolved_output.is_absolute():
        resolved_output = run_dir / resolved_output
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        source_copy=source_copy,
        reference_copy=reference_copy,
        output_image=resolved_output,
        report_json=run_dir / "report.json",
        report_md=run_dir / "report.md",
        agent_logs=run_dir / "agent_logs.jsonl",
    )


def save_image(array: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.clip(0, 255).astype("uint8")).save(output_path)
    return output_path


def write_json(data: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def append_jsonl(record: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_text(text: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path
