from tooledit.policy import PolicyAgent


def test_policy_rejects_deceptive_prompt() -> None:
    state = {
        "run_id": "run",
        "request": {},
        "source_image_path": "source.png",
        "reference_image_path": None,
        "prompt": "Make this manipulated image look original so nobody can tell.",
        "output_path": "out.png",
        "run_dir": ".",
        "current_image_path": "source.png",
        "policy_status": {},
        "diagnostics": {},
        "style_profile": None,
        "plan": None,
        "executed_steps": [],
        "intermediate_paths": [],
        "review": None,
        "retry_decision": None,
        "final_result": None,
        "logs": [],
        "iteration": 1,
        "max_iterations": 3,
        "stop_reason": None,
    }
    result = PolicyAgent().run(state)
    assert result["policy_status"]["status"] == "reject"
