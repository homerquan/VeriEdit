import json
from pathlib import Path

import pytest

from veriedit.config import WorkflowConfig
from veriedit.schemas import EditRequest
from veriedit.workflow import VeriEditWorkflow


def _load_manifest() -> list[dict[str, str]]:
    manifest_path = Path("tests/data/manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_fixture_manifest_has_expected_pairs() -> None:
    manifest = _load_manifest()
    assert len(manifest) >= 5
    for record in manifest:
        assert Path(record["source"]).exists()
        assert Path(record["reference"]).exists()
        assert record["prompt"]


@pytest.mark.parametrize("record", _load_manifest())
def test_fixture_pair_smoke_run(tmp_path: Path, record: dict[str, str]) -> None:
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=record["source"],
            reference_image=record["reference"],
            prompt=record["prompt"],
            output_path=str(tmp_path / f"{record['name']}.png"),
            max_iterations=1,
            save_intermediates=False,
        )
    )
    assert result.output_image is not None
    assert Path(result.output_image).exists()
    assert result.report_json is not None
    assert Path(result.report_json).exists()
