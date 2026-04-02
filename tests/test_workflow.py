from pathlib import Path
import json

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
            output_path=str(tmp_path / "outside-result.png"),
            max_iterations=2,
        )
    )
    assert result.output_image is not None
    assert Path(result.output_image).exists()
    assert result.run_id is not None
    assert len(result.run_id) == 8
    assert Path(result.output_image).parent.name == result.run_id
    assert Path(result.output_image).name == "outside-result.png"
    assert result.report_json is not None
    assert Path(result.report_json).exists()
    assert result.report_md is not None
    assert Path(result.report_md).exists()
    assert result.summary_md is not None
    assert Path(result.summary_md).exists()
    assert result.observation_json is not None
    assert Path(result.observation_json).exists()
    assert result.observation_md is not None
    assert Path(result.observation_md).exists()
    payload = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    report_md = Path(result.report_md).read_text(encoding="utf-8")
    assert "diagnostic_artifacts" in payload
    assert payload["diagnostic_artifacts"].get("regions_board")
    assert Path(payload["diagnostic_artifacts"]["regions_board"]).exists()
    assert any("Variant selected:" in note for step in payload["executed_steps"] for note in step["notes"])
    assert "## Images" in report_md
    assert "## Step Snapshots" in report_md
    assert "data:image" not in report_md
    assert "![" in report_md
    local_steps = [step for step in payload["executed_steps"] if step["tool"] in {"dust_cleanup", "scratch_candidate_cleanup", "small_defect_heal"}]
    assert local_steps
    assert any(step["execution_mode"] == "masked_local_repair" for step in local_steps)
    for step in local_steps:
        if step["execution_mode"] == "masked_local_repair":
            assert step["mask_name"] in {"dust_mask", "scratch_mask", "defect_union"}
            assert step["mask_coverage"] > 0.0
            assert step["after_metrics"].get("preserved_region_change_ratio", 0.0) <= 0.08
    observation = json.loads(Path(result.observation_json).read_text(encoding="utf-8"))
    assert any(event["kind"] == "node" for event in observation["trace"])
    assert any(event["kind"] == "tool" for event in observation["trace"])
