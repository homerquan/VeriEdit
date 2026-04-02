from pathlib import Path

import numpy as np
from PIL import Image

from veriedit.config import WorkflowConfig
from veriedit.schemas import EditRequest
from veriedit.workflow import VeriEditWorkflow


def _save_fixture(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype("uint8")).save(path)


def test_workflow_runs_end_to_end(tmp_path: Path) -> None:
    source = np.full((96, 96, 3), 180, dtype=np.uint8)
    source[..., 2] = 90
    source[30:34, 20:24] = 255
    source[50:53, 60:62] = 0
    reference = np.full((96, 96, 3), 150, dtype=np.uint8)
    reference[..., 0] = 170
    reference[..., 2] = 120
    source_path = tmp_path / "source.png"
    reference_path = tmp_path / "reference.png"
    _save_fixture(source_path, source)
    _save_fixture(reference_path, reference)
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            reference_image=str(reference_path),
            prompt="Clean dust, reduce yellow cast, and lightly sharpen while keeping it natural.",
            output_path=str(tmp_path / "result.png"),
            max_iterations=2,
        )
    )
    assert result.output_image is not None
    assert Path(result.output_image).exists()
    assert result.report_json is not None
    assert Path(result.report_json).exists()
    assert result.report_md is not None
    assert Path(result.report_md).exists()
