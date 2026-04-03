from pathlib import Path

from veriedit.agents.planner import PlannerAgent


def test_planner_prioritizes_local_repair_and_applies_feedback(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = {
        "run_id": "testplan",
        "request": {
            "allowed_tools": [],
            "llm_model": "",
        },
        "source_image_path": str(tmp_path / "source.png"),
        "reference_image_path": None,
        "prompt": "Repair scratches and damaged spots naturally with local retouching.",
        "output_path": str(tmp_path / "result.png"),
        "run_dir": str(run_dir),
        "current_image_path": str(tmp_path / "source.png"),
        "policy_status": {"status": "allow", "constraints": ["non-generative only"]},
        "diagnostics": {
            "source": {
                "width": 96,
                "height": 96,
                "mode": "RGB",
                "bit_depth": 8,
                "blur_score": 82.0,
                "noise_score": 0.05,
                "yellow_cast": 0.48,
                "contrast_score": 0.58,
                "clipping_highlights": 0.01,
                "clipping_shadows": 0.01,
                "skew_angle": 0.0,
                "dust_candidates": 140,
                "scratch_candidates": 46,
                "fade_score": 0.22,
                "sepia_score": 0.08,
                "edge_damage_ratio": 0.14,
                "underexposed": False,
            },
            "reference": None,
            "regions": {
                "largest_defect_ratio": 0.002,
                "top_regions": [
                    {"x": 12, "y": 16, "width": 18, "height": 16},
                    {"x": 38, "y": 22, "width": 14, "height": 18},
                ],
            },
            "artifacts": {},
        },
        "diagnostic_artifacts": {},
        "style_profile": None,
        "plan": None,
        "plan_history": [],
        "executed_steps": [],
        "agent_handoffs": [],
        "observation_trace": [],
        "intermediate_paths": [],
        "review": {
            "status": "revise",
            "prompt_score": 0.61,
            "artifact_risk": 0.46,
            "patch_metrics": {"preserved_region_change_ratio": 0.31},
            "findings": ["edit footprint is broader than ideal"],
            "recommendations": [
                "reduce non-local edits outside detected defect regions",
                "reduce sharpen amount by 30%",
            ],
        },
        "review_history": [],
        "human_review": None,
        "retry_decision": {
            "decision": "retry",
            "reason": "Recoverable issues detected.",
            "strategy": "preserve unaffected regions; prefer masked local repair; avoid broad tonal edits",
        },
        "final_result": None,
        "logs": [],
        "iteration": 2,
        "max_iterations": 3,
        "stop_reason": None,
    }

    planner = PlannerAgent(model=None)
    new_state = planner.run(state)
    plan = new_state["plan"]

    assert plan["detected_problems"]
    assert any(problem["problem"] == "localized_damage" for problem in plan["detected_problems"])
    assert plan["repair_strategy"]
    assert plan["repair_strategy"][0]["stage"] == "local_damage_repair"
    assert "stroke_paint" in {step["tool"] for step in plan["steps"]}
    assert not any(step["tool"] == "unsharp_mask" for step in plan["steps"])
    assert any("masked local repair" in item.lower() for item in plan["feedback_applied"])


def test_planner_advances_to_tone_stage_after_local_repair_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    state = {
        "run_id": "testplan2",
        "request": {
            "allowed_tools": [],
            "llm_model": "",
        },
        "source_image_path": str(tmp_path / "source.png"),
        "reference_image_path": None,
        "prompt": "Repair damage, reduce yellow cast, and keep it natural.",
        "output_path": str(tmp_path / "result.png"),
        "run_dir": str(run_dir),
        "current_image_path": str(tmp_path / "current.png"),
        "policy_status": {"status": "allow", "constraints": ["non-generative only"]},
        "diagnostics": {
            "source": {
                "width": 96,
                "height": 96,
                "mode": "RGB",
                "bit_depth": 8,
                "blur_score": 82.0,
                "noise_score": 0.05,
                "yellow_cast": 0.7,
                "contrast_score": 0.45,
                "clipping_highlights": 0.01,
                "clipping_shadows": 0.01,
                "skew_angle": 0.0,
                "dust_candidates": 120,
                "scratch_candidates": 30,
                "fade_score": 0.5,
                "sepia_score": 0.08,
                "edge_damage_ratio": 0.12,
                "underexposed": False,
            },
            "current": {
                "width": 96,
                "height": 96,
                "mode": "RGB",
                "bit_depth": 8,
                "blur_score": 80.0,
                "noise_score": 0.05,
                "yellow_cast": 0.68,
                "contrast_score": 0.43,
                "clipping_highlights": 0.01,
                "clipping_shadows": 0.01,
                "skew_angle": 0.0,
                "dust_candidates": 8,
                "scratch_candidates": 5,
                "fade_score": 0.47,
                "sepia_score": 0.08,
                "edge_damage_ratio": 0.03,
                "underexposed": False,
            },
            "reference": None,
            "regions": {
                "largest_defect_ratio": 0.0002,
                "top_regions": [
                    {"x": 12, "y": 16, "width": 12, "height": 12},
                ],
            },
            "artifacts": {},
        },
        "diagnostic_artifacts": {},
        "style_profile": None,
        "plan": None,
        "plan_history": [],
        "executed_steps": [
            {"tool": "dust_cleanup", "status": "ok"},
            {"tool": "scratch_candidate_cleanup", "status": "ok"},
            {"tool": "small_defect_heal", "status": "ok"},
        ],
        "agent_handoffs": [],
        "observation_trace": [],
        "intermediate_paths": [],
        "review": {
            "status": "revise",
            "prompt_score": 0.66,
            "artifact_risk": 0.28,
            "patch_metrics": {
                "preserved_region_change_ratio": 0.08,
                "defect_region_improvement": 0.14,
            },
            "findings": ["targeted defect regions improved"],
            "recommendations": [],
        },
        "review_history": [],
        "human_review": None,
        "retry_decision": {
            "decision": "retry",
            "reason": "Recoverable issues detected.",
            "strategy": "continue progressive refinement",
        },
        "final_result": None,
        "logs": [],
        "iteration": 2,
        "max_iterations": 5,
        "stop_reason": None,
    }

    planner = PlannerAgent(model=None)
    new_state = planner.run(state)
    plan = new_state["plan"]

    assert plan["repair_strategy"]
    assert plan["repair_strategy"][0]["stage"] == "tone_and_color"
    tools = [step["tool"] for step in plan["steps"]]
    assert "auto_white_balance" in tools
    assert not any(tool in {"dust_cleanup", "scratch_candidate_cleanup", "small_defect_heal"} for tool in tools[:1])
