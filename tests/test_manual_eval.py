from pathlib import Path

import numpy as np
from PIL import Image

from veriedit.config import WorkflowConfig
from veriedit.manual_eval import build_manual_eval_from_run
from veriedit.schemas import EditRequest
from veriedit.workflow import VeriEditWorkflow


def _save_fixture(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype("uint8")).save(path)


def test_build_manual_eval_from_run(tmp_path: Path) -> None:
    source = np.full((64, 64, 3), 160, dtype=np.uint8)
    source[..., 2] = 100
    source[12:16, 12:16] = 255
    reference = np.full((64, 64, 3), 150, dtype=np.uint8)
    source_path = tmp_path / "source.png"
    reference_path = tmp_path / "reference.png"
    _save_fixture(source_path, source)
    _save_fixture(reference_path, reference)

    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            reference_image=str(reference_path),
            prompt="Clean dust and keep it natural.",
            output_path=str(tmp_path / "result.png"),
            max_iterations=1,
            save_intermediates=False,
        )
    )
    markdown_path = build_manual_eval_from_run(result.run_id or "", artifact_root=tmp_path / "runs")
    text = markdown_path.read_text(encoding="utf-8")
    assert "# Manual Eval:" in text
    assert "## Images" in text
    assert "## Tool Usage" in text
    assert "## Trace" in text
    assert "![Source](" in text
    assert "![Result](" in text
