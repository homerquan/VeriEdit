from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from veriedit.schemas import RunArtifacts


def ensure_run_artifacts(
    artifact_root: Path,
    source_image: str,
    output_path: str | None = None,
    reference_image: str | None = None,
) -> RunArtifacts:
    artifact_root = Path(artifact_root).expanduser()
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = _generate_run_id(artifact_root)
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_copy = run_dir / Path(source_image).name
    shutil.copy2(source_image, source_copy)
    reference_copy = None
    if reference_image:
        reference_copy = run_dir / Path(reference_image).name
        shutil.copy2(reference_image, reference_copy)
    output_name = _output_filename(source_image=source_image, output_path=output_path)
    resolved_output = run_dir / output_name
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


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255), mode="L").save(output_path)
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


def _generate_run_id(artifact_root: Path, length: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(32):
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if not (artifact_root / candidate).exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique VeriEdit run id.")


def _output_filename(source_image: str, output_path: str | None) -> str:
    if output_path:
        candidate = Path(output_path).name
        if candidate:
            return candidate
    suffix = Path(source_image).suffix or ".png"
    return f"result{suffix}"
