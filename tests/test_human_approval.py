from pathlib import Path
import json

import numpy as np
from PIL import Image

from veriedit.config import WorkflowConfig
from veriedit.schemas import EditRequest, ReviewResult
from veriedit.workflow import VeriEditWorkflow


def _save_fixture(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype("uint8")).save(path)


def test_workflow_requests_human_review_for_ambiguous_result(tmp_path: Path) -> None:
    source = np.full((72, 72, 3), 170, dtype=np.uint8)
    source[..., 2] = 105
    source[16:19, 16:19] = 255
    source[44:47, 52:55] = 0
    source_path = tmp_path / "source.png"
    _save_fixture(source_path, source)

    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    workflow.reviewer_agent._review_with_gemini = lambda state: None  # type: ignore[attr-defined]
    workflow.reviewer_agent._heuristic_review = lambda state: ReviewResult(  # type: ignore[method-assign]
        status="revise",
        prompt_score=0.7,
        artifact_risk=0.33,
        naturalness_score=0.67,
        semantic_fabrication_risk=0.12,
        patch_metrics={
            "defect_region_change_ratio": 0.21,
            "preserved_region_change_ratio": 0.24,
            "defect_region_improvement": 0.05,
        },
        findings=["targeted defect regions improved", "preserved regions changed more than expected"],
        recommendations=["reduce non-local edits outside detected defect regions"],
        confidence=0.65,
    )

    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            prompt="Clean dust while keeping it natural.",
            output_path=str(tmp_path / "result.png"),
            max_iterations=1,
        )
    )

    assert result.success is False
    assert result.human_review_status == "pending"
    assert result.manual_eval_md is not None
    assert Path(result.manual_eval_md).exists()
    assert result.human_approval_json is not None
    approval_payload = json.loads(Path(result.human_approval_json).read_text(encoding="utf-8"))
    assert approval_payload["status"] == "pending"
    report_payload = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    assert report_payload["human_review"]["status"] == "pending"
