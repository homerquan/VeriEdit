from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from veriedit.io.writer import append_jsonl
from veriedit.llm import GeminiStructuredClient, has_gemini_support
from veriedit.observability import record_agent_handoff, record_node_event
from veriedit.schemas import (
    AgentLog,
    EditPlan,
    PlanStep,
    ProblemAssessment,
    RepairStage,
    ToolRecommendation,
    WorkflowState,
)
from veriedit.tools import build_tool_registry
from veriedit.tools.selector import rank_tools


class PlannerAgent:
    def __init__(self, model: str | None = None) -> None:
        self.registry = build_tool_registry()
        self.model = model

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        record_node_event(state, node="plan_edits", phase="start")
        plan = self._plan_with_gemini(state) or self._heuristic_plan(state)
        state["plan"] = plan.model_dump()
        state.setdefault("plan_history", []).append({"iteration": state["iteration"], **plan.model_dump()})
        problem_labels = [problem.problem for problem in plan.detected_problems[:3]]
        feedback_labels = plan.feedback_applied[:3]
        record_agent_handoff(
            state,
            from_agent="planner",
            to_agent="tool_trial",
            summary=f"Planned {len(plan.steps)} steps for '{plan.objective}' across {len(plan.repair_strategy)} stages.",
            key_points=[
                f"problems={', '.join(problem_labels) or 'none'}",
                f"top_tools={', '.join(step.tool for step in plan.steps[:4]) or 'none'}",
                f"feedback={'; '.join(feedback_labels) or 'diagnostics-first staging'}",
            ],
            payload={
                "objective": plan.objective,
                "steps": [step.model_dump() for step in plan.steps],
                "detected_problems": [item.model_dump() for item in plan.detected_problems],
                "repair_strategy": [item.model_dump() for item in plan.repair_strategy],
                "feedback_applied": plan.feedback_applied,
                "recommended_tools": [item.model_dump() for item in plan.recommended_tools[:5]],
            },
        )
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="planner",
                iteration=state["iteration"],
                input_summary={
                    "prompt": state["prompt"][:200],
                    "constraints": state["policy_status"].get("constraints", []),
                    "tool_count": len(self.registry.names()),
                },
                decision="planned",
                output_summary=plan.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        record_node_event(
            state,
            node="plan_edits",
            phase="end",
            summary={"step_count": len(plan.steps), "problem_count": len(plan.detected_problems)},
        )
        return state

    def _plan_with_gemini(self, state: WorkflowState) -> EditPlan | None:
        model = self.model or state["request"].get("llm_model")
        if not model or not has_gemini_support():
            return None
        system_prompt = (
            "You are the Planner Agent for a non-generative image editing workflow. "
            "First identify the concrete problems in the image, then build a minimal staged repair plan. "
            "Prefer structure-first and local defect repair before broader tonal polish when damage is localized. "
            "Listen to reviewer and retry feedback when present. Use only provided registry tools. "
            "Keep parameters conservative. Never propose generative fill. Respect policy constraints. "
            "Return JSON matching the EditPlan schema with keys objective, must_preserve, must_avoid, steps, acceptance, "
            "recommended_tools, detected_problems, repair_strategy, and feedback_applied."
        )
        payload = {
            "prompt": state["prompt"],
            "policy_constraints": state["policy_status"].get("constraints", []),
            "diagnostics": state["diagnostics"],
            "style_profile": state["style_profile"],
            "tool_registry": [
                spec.model_dump(exclude={"operation"})
                for spec in self.registry.specs()
                if spec.name in _allowed_or_all_tools(state, self.registry.names())
            ],
            "retry_context": state.get("retry_decision"),
            "review_context": state.get("review"),
            "agent_feedback": _recent_handoff_notes(state),
        }
        try:  # pragma: no cover - depends on external API
            response = GeminiStructuredClient(model=model).generate_json(system_prompt, payload)
            return EditPlan(**response)
        except (RuntimeError, ValidationError, ValueError, KeyError):
            return None

    def _heuristic_plan(self, state: WorkflowState) -> EditPlan:
        diagnostics_bundle = state["diagnostics"]
        diagnostics = diagnostics_bundle.get("current") or diagnostics_bundle["source"]
        region_summary = diagnostics_bundle.get("regions", {})
        prompt = state["prompt"].lower()
        retry = state.get("retry_decision") or {}
        blocked_tools = _blocked_tools_from_history(state["executed_steps"])
        allowed_tools = _allowed_or_all_tools(state, self.registry.names())
        blocked_tools |= {tool for tool in self.registry.names() if tool not in allowed_tools}

        detected_problems = _detect_problems(prompt, diagnostics, region_summary, bool(state["reference_image_path"]))
        feedback_flags, feedback_applied = _derive_feedback_directives(state, blocked_tools)
        recommendations = rank_tools(
            registry=self.registry,
            prompt=prompt,
            diagnostics=diagnostics_bundle,
            region_summary=region_summary,
            retry_context=retry,
            blocked_tools=blocked_tools,
            has_reference=bool(state["reference_image_path"]),
        )

        should_deskew = any(word in prompt for word in ("deskew", "de-skew", "straighten", "rotate", "align scan"))
        plausible_skew = 0.5 <= abs(float(diagnostics["skew_angle"])) <= 12.0
        steps, repair_strategy, acceptance = _build_staged_plan(
            state=state,
            recommendations=recommendations,
            detected_problems=detected_problems,
            feedback_flags=feedback_flags,
            blocked_tools=blocked_tools,
            region_summary=region_summary,
            should_deskew=should_deskew,
            plausible_skew=plausible_skew,
        )

        if not steps:
            steps.append(PlanStep(tool="histogram_balance", params={"strength": 0.4}, reason="Apply a safe baseline tonal normalization"))
            repair_strategy.append(
                RepairStage(
                    stage="baseline_normalization",
                    goal="Make one reversible tonal adjustment when no stronger diagnosis is available.",
                    selected_tools=["histogram_balance"],
                    rationale="Fallback plan used because no higher-confidence staged repairs were selected.",
                )
            )
            acceptance.append("overall balance improved naturally")

        objective = _objective_from_prompt(prompt, detected_problems)
        must_preserve = _must_preserve_from_context(prompt, detected_problems)
        must_avoid = _must_avoid_from_context(feedback_flags)
        if retry.get("strategy") and "reduce denoise" in retry["strategy"].lower():
            steps = [
                step.model_copy(
                    update={"params": {**step.params, "h": max(2.0, float(step.params.get("h", 4.0)) * 0.7)}}
                )
                if step.tool == "non_local_means_denoise"
                else step
                for step in steps
            ]

        return EditPlan(
            objective=objective,
            must_preserve=must_preserve,
            must_avoid=must_avoid,
            steps=steps,
            acceptance=list(dict.fromkeys(acceptance)) or ["result stays realistic"],
            recommended_tools=recommendations,
            detected_problems=detected_problems,
            repair_strategy=repair_strategy,
            feedback_applied=feedback_applied,
        )

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _objective_from_prompt(prompt: str, problems: list[ProblemAssessment]) -> str:
    names = {problem.problem for problem in problems}
    if {"surface_dust", "scratch_damage"} & names and "faded_tones" in names:
        return "repair visible damage and restore faded tone naturally"
    if {"surface_dust", "scratch_damage", "localized_damage"} & names:
        return "repair visible surface damage while preserving authentic structure"
    if "faded_tones" in names or "color_cast" in names:
        return "restore tone and color conservatively while preserving realism"
    if "noise" in names:
        return "reduce noise conservatively without erasing texture"
    if "restore" in prompt:
        return "restore the image naturally"
    if "clean" in prompt:
        return "clean the image conservatively"
    if "contrast" in prompt or "tone" in prompt:
        return "improve tone and contrast while preserving realism"
    return "improve the image with explicit non-generative edits"


def _must_preserve_from_context(prompt: str, problems: list[ProblemAssessment]) -> list[str]:
    preserve = ["identity", "overall composition"]
    if any(problem.problem in {"surface_dust", "scratch_damage", "localized_damage"} for problem in problems):
        preserve.append("unaffected regions")
        preserve.append("edge structure")
    if "face" in prompt or "portrait" in prompt:
        preserve.append("facial structure")
    else:
        preserve.append("subject structure")
    return list(dict.fromkeys(preserve))


def _must_avoid_from_context(feedback_flags: dict[str, bool]) -> list[str]:
    avoid = ["oversmoothing", "halo artifacts", "semantic content changes", "fabricated detail"]
    if feedback_flags.get("avoid_global_tone"):
        avoid.append("broad non-local tone changes")
    if feedback_flags.get("prefer_local_repair"):
        avoid.append("overpainting outside defect regions")
    return list(dict.fromkeys(avoid))


def _blocked_tools_from_history(executed_steps: list[dict[str, Any]]) -> set[str]:
    blocked: set[str] = set()
    outcomes: dict[str, list[str]] = {}
    for step in executed_steps:
        outcomes.setdefault(step["tool"], []).append(step["status"])
    for tool, statuses in outcomes.items():
        if statuses and all(status == "rolled_back" for status in statuses):
            blocked.add(tool)
    return blocked


def _allowed_or_all_tools(state: WorkflowState, all_tools: list[str]) -> set[str]:
    requested = state["request"].get("allowed_tools") or []
    if not requested:
        return set(all_tools)
    return {tool for tool in requested if tool in set(all_tools)}


def _recent_handoff_notes(state: WorkflowState) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for handoff in state.get("agent_handoffs", [])[-6:]:
        notes.append(
            {
                "from_agent": handoff.get("from_agent"),
                "to_agent": handoff.get("to_agent"),
                "summary": handoff.get("summary"),
                "key_points": handoff.get("key_points", []),
            }
        )
    return notes


def _recent_plan_tools(state: WorkflowState) -> list[str]:
    history = state.get("plan_history", [])
    if not history:
        return []
    return [step.get("tool") for step in history[-1].get("steps", []) if step.get("tool")]


def _detect_problems(
    prompt: str,
    diagnostics: dict[str, Any],
    region_summary: dict[str, Any],
    has_reference: bool,
) -> list[ProblemAssessment]:
    problems: list[ProblemAssessment] = []
    yellow_cast = float(diagnostics.get("yellow_cast", 0.0))
    contrast_score = float(diagnostics.get("contrast_score", 1.0))
    fade_score = float(diagnostics.get("fade_score", 0.0))
    noise_score = float(diagnostics.get("noise_score", 0.0))
    blur_score = float(diagnostics.get("blur_score", 0.0))
    dust_candidates = int(diagnostics.get("dust_candidates", 0))
    scratch_candidates = int(diagnostics.get("scratch_candidates", 0))
    edge_damage_ratio = float(diagnostics.get("edge_damage_ratio", 0.0))
    largest_defect_ratio = float(region_summary.get("largest_defect_ratio", 0.0))

    if dust_candidates > 18 or "dust" in prompt or "speck" in prompt:
        problems.append(
            ProblemAssessment(
                problem="surface_dust",
                severity=_severity_from_value(dust_candidates, medium=80.0, high=220.0),
                evidence=[f"dust_candidates={dust_candidates}"],
                desired_outcome="Remove isolated dust and speckles without flattening real texture.",
                preferred_scope="local",
            )
        )
    if scratch_candidates > 10 or "scratch" in prompt or "streak" in prompt:
        problems.append(
            ProblemAssessment(
                problem="scratch_damage",
                severity=_severity_from_value(scratch_candidates, medium=24.0, high=80.0),
                evidence=[f"scratch_candidates={scratch_candidates}"],
                desired_outcome="Suppress narrow scratch artifacts while preserving contours.",
                preferred_scope="local",
            )
        )
    if edge_damage_ratio > 0.06 or largest_defect_ratio > 0.0007 or any(word in prompt for word in ("repair", "retouch", "damaged", "patch")):
        problems.append(
            ProblemAssessment(
                problem="localized_damage",
                severity="high" if largest_defect_ratio > 0.0015 else "medium",
                evidence=[
                    f"edge_damage_ratio={edge_damage_ratio:.4f}",
                    f"largest_defect_ratio={largest_defect_ratio:.5f}",
                ],
                desired_outcome="Repair concentrated damaged regions before broad tonal edits.",
                preferred_scope="local",
            )
        )
    if yellow_cast > 0.58 or "yellow" in prompt or "white balance" in prompt:
        problems.append(
            ProblemAssessment(
                problem="color_cast",
                severity=_severity_from_value(yellow_cast, medium=0.62, high=0.72),
                evidence=[f"yellow_cast={yellow_cast:.4f}"],
                desired_outcome="Neutralize unwanted cast without removing intentional warmth.",
                preferred_scope="global",
            )
        )
    if contrast_score < 0.5 or fade_score > 0.4 or any(word in prompt for word in ("contrast", "tone", "faded", "exposure")):
        problems.append(
            ProblemAssessment(
                problem="faded_tones",
                severity="high" if contrast_score < 0.38 or fade_score > 0.58 else "medium",
                evidence=[f"contrast_score={contrast_score:.4f}", f"fade_score={fade_score:.4f}"],
                desired_outcome="Recover tonal separation without clipping or creating halos.",
                preferred_scope="global",
            )
        )
    if noise_score > 0.075 or "noise" in prompt or "grain" in prompt:
        problems.append(
            ProblemAssessment(
                problem="noise",
                severity="high" if noise_score > 0.12 else "medium",
                evidence=[f"noise_score={noise_score:.4f}"],
                desired_outcome="Reduce noise conservatively while keeping fine structure believable.",
                preferred_scope="mixed",
            )
        )
    if any(word in prompt for word in ("sharp", "detail", "clarity")) or blur_score < 70.0:
        problems.append(
            ProblemAssessment(
                problem="detail_recovery",
                severity="medium",
                evidence=[f"blur_score={blur_score:.2f}"],
                desired_outcome="Restore perceived crispness only after cleanup and tone are stable.",
                preferred_scope="global",
            )
        )
    if has_reference:
        problems.append(
            ProblemAssessment(
                problem="reference_style",
                severity="low",
                evidence=["reference image provided"],
                desired_outcome="Borrow warmth, contrast, and texture feel only after structural repair.",
                preferred_scope="global",
            )
        )
    return problems


def _severity_from_value(value: float, *, medium: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _derive_feedback_directives(state: WorkflowState, blocked_tools: set[str]) -> tuple[dict[str, bool], list[str]]:
    review = state.get("review") or {}
    retry = state.get("retry_decision") or {}
    recommendations_text = " ".join(review.get("recommendations", [])).lower()
    findings_text = " ".join(review.get("findings", [])).lower()
    strategy_text = " ".join(
        str(value) for value in [retry.get("reason"), retry.get("strategy")] if value
    ).lower()
    patch_metrics = review.get("patch_metrics", {})
    feedback = {
        "prefer_local_repair": False,
        "avoid_global_tone": False,
        "avoid_sharpen": False,
        "avoid_denoise": False,
        "avoid_repetition": False,
    }
    notes: list[str] = []

    if patch_metrics.get("preserved_region_change_ratio", 0.0) > 0.24 or "reduce non-local edits" in recommendations_text:
        feedback["prefer_local_repair"] = True
        feedback["avoid_global_tone"] = True
        notes.append("Prioritized masked local repair because review found too much change outside detected defect regions.")
    if "edit footprint is broader than ideal" in findings_text:
        feedback["avoid_global_tone"] = True
        notes.append("Held broad tonal edits back because the prior edit footprint was wider than ideal.")
    if "prefer local heal" in recommendations_text or "prefer local repair" in strategy_text or "replace broad cleanup with local heal" in strategy_text:
        feedback["prefer_local_repair"] = True
        notes.append("Shifted the plan toward local healing and away from broad cleanup based on reviewer feedback.")
    if "reduce sharpen" in recommendations_text or "reduce sharpen" in strategy_text:
        feedback["avoid_sharpen"] = True
        notes.append("Removed or softened sharpening after the reviewer flagged halo risk.")
    if "stop further denoise" in recommendations_text or "reduce denoise" in strategy_text:
        feedback["avoid_denoise"] = True
        notes.append("Avoided stronger denoise because earlier review indicated texture loss risk.")
    review_history = state.get("review_history", [])
    if len(review_history) >= 2:
        previous = review_history[-2]
        current = review_history[-1]
        prompt_gain = float(current.get("prompt_score", 0.0)) - float(previous.get("prompt_score", 0.0))
        artifact_drop = float(previous.get("artifact_risk", 1.0)) - float(current.get("artifact_risk", 1.0))
        if prompt_gain < 0.01 and artifact_drop < 0.02:
            feedback["avoid_repetition"] = True
            notes.append("Rotated away from the last tool batch because recent iterations were not improving enough.")
    if blocked_tools:
        notes.append(f"Kept previously rolled-back tools out of the plan: {', '.join(sorted(blocked_tools))}.")
    if not notes:
        notes.append("Used diagnostics-first staging so visible defects are addressed before tonal polish.")
    return feedback, notes


def _build_staged_plan(
    *,
    state: WorkflowState,
    recommendations: list[ToolRecommendation],
    detected_problems: list[ProblemAssessment],
    feedback_flags: dict[str, bool],
    blocked_tools: set[str],
    region_summary: dict[str, Any],
    should_deskew: bool,
    plausible_skew: bool,
) -> tuple[list[PlanStep], list[RepairStage], list[str]]:
    prompt = state["prompt"].lower()
    problem_names = {problem.problem for problem in detected_problems}
    stage_specs: list[dict[str, Any]] = []
    if should_deskew and plausible_skew:
        stage_specs.append(
            {
                "stage": "geometry_setup",
                "goal": "Straighten the scan before repairing or balancing tone.",
                "tools": {"deskew"},
                "limit": 1,
                "rationale": "Geometry problems should be corrected before local repair so masks and strokes line up.",
                "min_score": 0.0,
            }
        )
    if {"surface_dust", "scratch_damage", "localized_damage"} & problem_names:
        stage_specs.append(
            {
                "stage": "local_damage_repair",
                "goal": "Repair explicit damage with local tools before global polish.",
                "tools": {"dust_cleanup", "scratch_candidate_cleanup", "small_defect_heal", "stroke_paint"},
                "limit": 3,
                "rationale": "Visible defects should be cleaned first so later tonal moves do not amplify them.",
                "min_score": 0.8 if feedback_flags.get("prefer_local_repair") else 1.0,
            }
        )
    if {"color_cast", "faded_tones"} & problem_names:
        stage_specs.append(
            {
                "stage": "tone_and_color",
                "goal": "Normalize color cast and faded tone conservatively after structure is stable.",
                "tools": {"auto_white_balance", "shadow_highlight_balance", "clahe_contrast", "histogram_balance"},
                "limit": 1 if feedback_flags.get("avoid_global_tone") else 2,
                "rationale": "Global tone edits come after structural repair and should stay restrained if preserved regions were disturbed before.",
                "min_score": 1.0,
            }
        )
    if "noise" in problem_names and not feedback_flags.get("avoid_denoise"):
        stage_specs.append(
            {
                "stage": "noise_control",
                "goal": "Reduce residual noise conservatively after the main structural fixes.",
                "tools": {"bilateral_denoise", "non_local_means_denoise", "wavelet_denoise", "median_cleanup"},
                "limit": 1,
                "rationale": "Noise cleanup should stay gentle and avoid erasing repaired structure.",
                "min_score": 1.0,
            }
        )
    if state["reference_image_path"] and "reference_style" in problem_names:
        stage_specs.append(
            {
                "stage": "reference_bias",
                "goal": "Borrow only the reference tone and texture feel after the repair is structurally sound.",
                "tools": {"bounded_histogram_match_to_reference", "texture_softness_bias_from_reference"},
                "limit": 1,
                "rationale": "Reference guidance should remain a finishing pass, never the source of new content.",
                "min_score": 1.0,
            }
        )
    if "detail_recovery" in problem_names and not feedback_flags.get("avoid_sharpen"):
        stage_specs.append(
            {
                "stage": "detail_finish",
                "goal": "Restore perceived detail only at the end and only if it stays natural.",
                "tools": {"unsharp_mask", "edge_preserving_sharpen"},
                "limit": 1,
                "rationale": "Sharpening belongs at the end so it does not exaggerate damage or noise.",
                "min_score": 1.0,
            }
        )

    steps: list[PlanStep] = []
    strategy: list[RepairStage] = []
    acceptance: list[str] = []
    max_steps = 3 if state["max_iterations"] > 1 else 5
    active_stage_names = _select_active_stages(state, stage_specs, detected_problems, feedback_flags)
    recent_planned_tools = set(_recent_plan_tools(state))
    for stage_spec in stage_specs:
        if stage_spec["stage"] not in active_stage_names:
            continue
        selected_tools: list[str] = []
        stage_candidate_pool = [item.tool for item in recommendations if item.tool in stage_spec["tools"]]
        for recommendation in recommendations:
            if len(steps) >= max_steps or len(selected_tools) >= stage_spec["limit"]:
                break
            if recommendation.tool not in stage_spec["tools"]:
                continue
            if recommendation.score < stage_spec["min_score"]:
                continue
            candidate = _step_from_recommendation(recommendation, state, region_summary, blocked_tools)
            if candidate is None:
                continue
            if any(step.tool == candidate.tool for step in steps):
                continue
            if (
                feedback_flags.get("avoid_repetition")
                and candidate.tool in recent_planned_tools
                and any(tool not in recent_planned_tools for tool in stage_candidate_pool)
            ):
                continue
            if feedback_flags.get("avoid_global_tone") and stage_spec["stage"] == "tone_and_color" and candidate.tool == "clahe_contrast":
                continue
            steps.append(candidate)
            selected_tools.append(candidate.tool)
            acceptance.extend(_acceptance_for_tool(candidate.tool))
        if selected_tools:
            strategy.append(
                RepairStage(
                    stage=stage_spec["stage"],
                    goal=stage_spec["goal"],
                    selected_tools=selected_tools,
                    rationale=stage_spec["rationale"],
                )
            )
        if len(steps) >= max_steps:
            break
    if {"surface_dust", "scratch_damage", "localized_damage"} & problem_names:
        acceptance.append("unaffected regions remain stable while damaged regions improve")
    if "natural" in prompt or "realistic" in prompt:
        acceptance.append("overall result remains natural")
    return _ordered_unique_steps(steps), strategy, acceptance


def _select_active_stages(
    state: WorkflowState,
    stage_specs: list[dict[str, Any]],
    detected_problems: list[ProblemAssessment],
    feedback_flags: dict[str, bool],
) -> list[str]:
    if not stage_specs:
        return []
    problem_lookup = {problem.problem: problem for problem in detected_problems}
    ok_tools = {step["tool"] for step in state.get("executed_steps", []) if step.get("status") == "ok"}
    review = state.get("review") or {}
    patch_metrics = review.get("patch_metrics", {})

    priorities: list[tuple[float, str]] = []
    for stage_spec in stage_specs:
        stage = stage_spec["stage"]
        stage_tools = set(stage_spec["tools"])
        started = len(ok_tools & stage_tools)
        priority = 0.0

        if stage == "geometry_setup":
            priority = 12.0 if started == 0 else 0.0
        elif stage == "local_damage_repair":
            local_problems = [problem_lookup.get(name) for name in ("surface_dust", "scratch_damage", "localized_damage")]
            active_local_problems = [problem for problem in local_problems if problem is not None]
            if active_local_problems:
                priority = 11.0 if started == 0 else 7.5
                if feedback_flags.get("prefer_local_repair"):
                    priority += 2.0
                if patch_metrics.get("defect_region_improvement", 0.0) > 0.05 and started >= 2:
                    priority -= 2.0
        elif stage == "tone_and_color":
            tone_problems = [problem_lookup.get(name) for name in ("color_cast", "faded_tones")]
            if any(problem is not None for problem in tone_problems):
                priority = 8.5
                if started > 0:
                    priority -= 1.0
                if feedback_flags.get("avoid_global_tone"):
                    priority -= 4.5
                if started == 0 and any(tool in ok_tools for tool in {"dust_cleanup", "scratch_candidate_cleanup", "small_defect_heal", "stroke_paint"}):
                    priority += 1.0
        elif stage == "noise_control":
            if problem_lookup.get("noise") is not None and not feedback_flags.get("avoid_denoise"):
                priority = 6.0 if any(ok_tools) else 4.5
                if started > 0:
                    priority -= 1.0
        elif stage == "reference_bias":
            if problem_lookup.get("reference_style") is not None:
                priority = 5.5 if any(tool in ok_tools for tool in {"auto_white_balance", "shadow_highlight_balance", "clahe_contrast", "histogram_balance"}) else 3.5
                if started > 0:
                    priority -= 0.5
        elif stage == "detail_finish":
            if problem_lookup.get("detail_recovery") is not None and not feedback_flags.get("avoid_sharpen"):
                priority = 4.0 if any(ok_tools) else 2.5
                if started > 0:
                    priority -= 1.5

        if priority > 0.0:
            priorities.append((priority, stage))

    if not priorities:
        return [stage_specs[0]["stage"]]

    priorities.sort(key=lambda item: item[0], reverse=True)
    chosen = [priorities[0][1]]
    if len(priorities) > 1 and priorities[1][0] >= priorities[0][0] - 1.5:
        chosen.append(priorities[1][1])
    return chosen


def _stroke_paint_boxes(region_summary: dict[str, Any]) -> list[dict[str, int]]:
    boxes: list[dict[str, int]] = []
    for region in region_summary.get("top_regions", [])[:3]:
        width = int(region.get("width", 0))
        height = int(region.get("height", 0))
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        if width <= 0 or height <= 0:
            continue
        if width * height > 16000:
            continue
        pad = max(4, int(max(width, height) * 0.25))
        boxes.append({"x": max(0, x - pad), "y": max(0, y - pad), "width": width + pad * 2, "height": height + pad * 2})
    return boxes


def _step_from_recommendation(
    recommendation: ToolRecommendation,
    state: WorkflowState,
    region_summary: dict[str, Any],
    blocked_tools: set[str],
) -> PlanStep | None:
    tool = recommendation.tool
    if tool in blocked_tools:
        return None
    if tool in {"paint_strokes", "spot_healing_brush", "healing_brush", "clone_source_paint", "clone_stamp"}:
        return None
    params = dict(recommendation.params_hint)
    if tool == "deskew":
        diagnostics = (state["diagnostics"] or {}).get("current") or (state["diagnostics"] or {}).get("source", {})
        angle = float(diagnostics.get("skew_angle", 0.0))
        if abs(angle) < 0.5:
            return None
        params.setdefault("angle", angle)
        return PlanStep(tool=tool, params=params, reason="Correct noticeable skew before other edits.")
    if tool == "stroke_paint":
        boxes = _stroke_paint_boxes(region_summary)
        if not boxes:
            return None
        params.update(
            {
                "mask_boxes": boxes,
                "min_size": 3,
                "max_size": 10,
                "opacity": 0.5 if "natural" in state["prompt"].lower() else 0.62,
            }
        )
        return PlanStep(tool=tool, params=params, reason="Use iterative local repair strokes on the most concentrated damaged regions.")
    if tool == "texture_softness_bias_from_reference" and not state["reference_image_path"]:
        return None
    if tool == "bounded_histogram_match_to_reference" and not state["reference_image_path"]:
        return None
    reason = recommendation.rationale or f"Selected by tool ranking for {tool}"
    if tool == "shadow_highlight_balance":
        reason = "Recover faded tonal separation without aggressive clipping."
    elif tool == "clahe_contrast":
        reason = "Gently restore local contrast after structure cleanup."
    elif tool == "auto_white_balance":
        reason = "Reduce color cast conservatively."
    elif tool == "dust_cleanup":
        reason = "Remove isolated scan dust and speckles."
    elif tool == "scratch_candidate_cleanup":
        reason = "Reduce small scratch-like artifacts where safe."
    elif tool == "small_defect_heal":
        reason = "Use local healing for small detected defect regions after coarse cleanup."
    elif tool == "non_local_means_denoise":
        reason = "Reduce visible scan or sensor noise."
    elif tool == "bilateral_denoise":
        reason = "Use a gentler edge-preserving denoise after prior rollback or review warning."
    elif tool == "bounded_histogram_match_to_reference":
        reason = "Bias tonal feel toward the reference without copying content."
    elif tool == "texture_softness_bias_from_reference":
        reason = "Match the reference softness or crispness feel at a low level."
    elif tool == "unsharp_mask":
        reason = "Add light sharpening only after cleanup is stable."
    return PlanStep(tool=tool, params=params, reason=reason)


def _acceptance_for_tool(tool: str) -> list[str]:
    mapping = {
        "deskew": ["scan alignment improved"],
        "auto_white_balance": ["yellow cast reduced"],
        "shadow_highlight_balance": ["contrast improved without clipping"],
        "clahe_contrast": ["contrast improved without clipping"],
        "dust_cleanup": ["dust visibly reduced"],
        "scratch_candidate_cleanup": ["scratch defects reduced"],
        "small_defect_heal": ["targeted defect regions repaired naturally"],
        "stroke_paint": ["localized damaged regions repaired without broad overpaint"],
        "non_local_means_denoise": ["noise reduced while retaining texture"],
        "bilateral_denoise": ["noise reduced while retaining texture"],
        "bounded_histogram_match_to_reference": ["reference style reflected only in tone and texture"],
        "texture_softness_bias_from_reference": ["reference style reflected only in tone and texture"],
        "unsharp_mask": ["detail remains natural"],
    }
    return mapping.get(tool, [])


def _ordered_unique_steps(steps: list[PlanStep]) -> list[PlanStep]:
    preferred_order = {
        "deskew": 10,
        "dust_cleanup": 20,
        "scratch_candidate_cleanup": 25,
        "small_defect_heal": 30,
        "stroke_paint": 35,
        "auto_white_balance": 40,
        "shadow_highlight_balance": 45,
        "clahe_contrast": 50,
        "histogram_balance": 55,
        "non_local_means_denoise": 60,
        "bilateral_denoise": 61,
        "bounded_histogram_match_to_reference": 70,
        "texture_softness_bias_from_reference": 75,
        "unsharp_mask": 80,
    }
    indexed = []
    seen: set[str] = set()
    for idx, step in enumerate(steps):
        if step.tool in seen:
            continue
        seen.add(step.tool)
        indexed.append((preferred_order.get(step.tool, 999), idx, step))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]
