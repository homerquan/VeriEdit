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
    assert "recommended_tools" in (payload["plan"] or {})
    assert payload["plan"].get("detected_problems")
    assert payload["plan"].get("repair_strategy")
    assert payload["plan"].get("feedback_applied")
    assert payload.get("plan_history")
    assert payload.get("tool_trial_history") is not None
    assert payload.get("review_history")
    assert payload["diagnostics"].get("current")
    assert payload.get("agent_handoffs")
    assert any(handoff["from_agent"] == "planner" and handoff["to_agent"] == "tool_trial" for handoff in payload["agent_handoffs"])
    assert any(handoff["from_agent"] == "tool_trial" and handoff["to_agent"] == "executor" for handoff in payload["agent_handoffs"])
    assert any("Variant selected:" in note for step in payload["executed_steps"] for note in step["notes"])
    assert "## Images" in report_md
    assert "### Detected Problems" in report_md
    assert "### Repair Strategy" in report_md
    assert "### Feedback Applied" in report_md
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
    assert any(event["kind"] == "handoff" for event in observation["trace"])
    assert any(event["kind"] == "tool" for event in observation["trace"])


def test_workflow_clone_trials_can_inject_clone_stamp_step(tmp_path: Path) -> None:
    source = np.full((96, 96, 3), 125, dtype=np.uint8)
    for x in range(96):
        source[:, x, 0] = 85 + x
        source[:, x, 1] = 118
        source[:, x, 2] = 108
    source[28:58, 18:42] = 255
    source_path = tmp_path / "source.png"
    _save_fixture(source_path, source)
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            prompt="Repair the large peeled white patch using clone stamp only and keep it natural.",
            allowed_tools=["clone_stamp"],
            max_iterations=1,
            max_tool_trials=8,
            save_intermediates=True,
            enable_human_approval=False,
        )
    )
    assert result.report_json is not None
    payload = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    trials = payload.get("tool_trial_history") or []
    assert trials
    latest_trial = trials[-1]
    assert latest_trial["attempted"] is True
    assert len(latest_trial["trials"]) <= 8
    assert latest_trial["accepted"] is True
    tools = [step["tool"] for step in payload["executed_steps"]]
    assert tools
    assert set(tools) == {"clone_stamp"}
    clone_step = payload["executed_steps"][0]
    assert clone_step["execution_mode"] == "masked_local_repair"
    assert clone_step["mask_name"] == "clone_roi"
    result_image = np.asarray(Image.open(result.output_image)).astype(np.uint8)
    assert float(result_image[34:52, 22:38, 0].mean()) < float(source[34:52, 22:38, 0].mean())


def test_workflow_can_plan_and_execute_stroke_paint(tmp_path: Path) -> None:
    source = np.full((96, 96, 3), 140, dtype=np.uint8)
    source[30:38, 28:70] = 245
    source[40:45, 40:48] = 20
    source_path = tmp_path / "source.png"
    _save_fixture(source_path, source)
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            prompt="Repair the damaged region naturally with local retouching.",
            max_iterations=1,
            save_intermediates=True,
            enable_human_approval=False,
        )
    )
    assert result.report_json is not None
    payload = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    tools = [step["tool"] for step in payload["executed_steps"]]
    assert "stroke_paint" in tools
    stroke_step = next(step for step in payload["executed_steps"] if step["tool"] == "stroke_paint")
    assert stroke_step["execution_mode"] == "masked_local_repair"
    assert stroke_step["mask_name"] == "stroke_roi"


def test_workflow_respects_allowed_tools_filter(tmp_path: Path) -> None:
    source = np.full((96, 96, 3), 140, dtype=np.uint8)
    source[30:38, 28:70] = 245
    source[40:45, 40:48] = 20
    source_path = tmp_path / "source.png"
    _save_fixture(source_path, source)
    workflow = VeriEditWorkflow(config=WorkflowConfig(artifact_root=tmp_path / "runs"))
    result = workflow.run(
        EditRequest(
            source_image=str(source_path),
            prompt="Repair the damaged region naturally with local retouching.",
            allowed_tools=["stroke_paint"],
            max_iterations=1,
            save_intermediates=True,
            enable_human_approval=False,
        )
    )
    assert result.report_json is not None
    payload = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    tools = [step["tool"] for step in payload["executed_steps"]]
    assert tools
    assert set(tools) == {"stroke_paint"}
    assert payload["request"]["allowed_tools"] == ["stroke_paint"]
