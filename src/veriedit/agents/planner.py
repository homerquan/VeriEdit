from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from veriedit.io.writer import append_jsonl
from veriedit.llm import GeminiStructuredClient, has_gemini_support
from veriedit.observability import record_agent_handoff, record_node_event
from veriedit.schemas import AgentLog, EditPlan, PlanStep, WorkflowState
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
        record_agent_handoff(
            state,
            from_agent="planner",
            to_agent="executor",
            summary=f"Planned {len(plan.steps)} steps toward '{plan.objective}'.",
            key_points=[
                f"must_preserve={', '.join(plan.must_preserve[:3]) or 'n/a'}",
                f"must_avoid={', '.join(plan.must_avoid[:3]) or 'n/a'}",
                f"top_tools={', '.join(step.tool for step in plan.steps[:4]) or 'none'}",
            ],
            payload={
                "objective": plan.objective,
                "steps": [step.model_dump() for step in plan.steps],
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
        record_node_event(state, node="plan_edits", phase="end", summary={"step_count": len(plan.steps)})
        return state

    def _plan_with_gemini(self, state: WorkflowState) -> EditPlan | None:
        model = self.model or state["request"].get("llm_model")
        if not model or not has_gemini_support():
            return None
        system_prompt = (
            "You are the Planner Agent for a non-generative image editing workflow. "
            "Prefer the minimal effective edit sequence. Use only provided registry tools. "
            "Keep parameters conservative. Never propose generative fill. Respect policy constraints. "
            "Return JSON matching the EditPlan schema with keys objective, must_preserve, must_avoid, steps, acceptance."
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
        }
        try:  # pragma: no cover - depends on external API
            response = GeminiStructuredClient(model=model).generate_json(system_prompt, payload)
            return EditPlan(**response)
        except (RuntimeError, ValidationError, ValueError, KeyError):
            return None

    def _heuristic_plan(self, state: WorkflowState) -> EditPlan:
        diagnostics = state["diagnostics"]["source"]
        prompt = state["prompt"].lower()
        retry = state.get("retry_decision") or {}
        blocked_tools = _blocked_tools_from_history(state["executed_steps"])
        allowed_tools = _allowed_or_all_tools(state, self.registry.names())
        blocked_tools |= {tool for tool in self.registry.names() if tool not in allowed_tools}
        region_summary = state["diagnostics"].get("regions", {})
        recommendations = rank_tools(
            registry=self.registry,
            prompt=prompt,
            diagnostics=state["diagnostics"],
            region_summary=region_summary,
            retry_context=retry,
            blocked_tools=blocked_tools,
            has_reference=bool(state["reference_image_path"]),
        )
        steps: list[PlanStep] = []
        acceptance: list[str] = []
        must_avoid = ["oversmoothing", "halo artifacts", "semantic content changes"]
        should_deskew = any(word in prompt for word in ("deskew", "de-skew", "straighten", "rotate", "align scan"))
        plausible_skew = 0.5 <= abs(float(diagnostics["skew_angle"])) <= 12.0
        if should_deskew and plausible_skew:
            steps.append(PlanStep(tool="deskew", params={"angle": diagnostics["skew_angle"]}, reason="Correct noticeable skew"))
            acceptance.append("skew reduced")
        for recommendation in recommendations:
            candidate = _step_from_recommendation(recommendation, state, region_summary, blocked_tools)
            if candidate is None:
                continue
            if recommendation.score < 1.0:
                continue
            if any(step.tool == candidate.tool for step in steps):
                continue
            steps.append(candidate)
            acceptance.extend(_acceptance_for_tool(candidate.tool))
            if len(steps) >= 6:
                break
        if not steps:
            steps.append(PlanStep(tool="histogram_balance", params={"strength": 0.4}, reason="Apply a safe baseline tonal normalization"))
            acceptance.append("overall balance improved naturally")
        steps = _ordered_unique_steps(steps)
        objective = _objective_from_prompt(prompt)
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
            must_preserve=["identity", "facial structure", "overall composition"],
            must_avoid=must_avoid,
            steps=steps,
            acceptance=list(dict.fromkeys(acceptance)) or ["result stays realistic"],
            recommended_tools=recommendations,
        )

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _objective_from_prompt(prompt: str) -> str:
    if "restore" in prompt:
        return "restore the image naturally"
    if "clean" in prompt:
        return "clean the image conservatively"
    if "contrast" in prompt or "tone" in prompt:
        return "improve tone and contrast while preserving realism"
    return "improve the image with explicit non-generative edits"


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


def _should_use_stroke_paint(
    prompt: str,
    diagnostics: dict[str, Any],
    region_summary: dict[str, Any],
    blocked_tools: set[str],
) -> bool:
    if "stroke_paint" in blocked_tools:
        return False
    prompt_requests_local_repair = any(
        word in prompt for word in ("repair", "retouch", "paint", "patch", "restore damaged", "fix damaged")
    )
    concentrated_damage = (
        diagnostics["scratch_candidates"] > 24
        or diagnostics["dust_candidates"] > 260
        or float(region_summary.get("largest_defect_ratio", 0.0)) > 0.0008
    )
    return prompt_requests_local_repair or concentrated_damage


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
    recommendation,
    state: WorkflowState,
    region_summary: dict[str, Any],
    blocked_tools: set[str],
) -> PlanStep | None:
    tool = recommendation.tool
    if tool in blocked_tools:
        return None
    if tool in {"paint_strokes", "spot_healing_brush", "healing_brush", "clone_source_paint"}:
        return None
    params = dict(recommendation.params_hint)
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
        return PlanStep(tool=tool, params=params, reason="Use iterative local repair strokes on the most concentrated damaged regions")
    if tool == "texture_softness_bias_from_reference" and not state["reference_image_path"]:
        return None
    if tool == "bounded_histogram_match_to_reference" and not state["reference_image_path"]:
        return None
    reason = recommendation.rationale or f"Selected by tool ranking for {tool}"
    if tool == "shadow_highlight_balance":
        reason = "Recover faded tonal separation without aggressive clipping"
    elif tool == "clahe_contrast":
        reason = "Gently restore local contrast"
    elif tool == "auto_white_balance":
        reason = "Reduce color cast conservatively"
    elif tool == "dust_cleanup":
        reason = "Remove isolated scan dust and speckles"
    elif tool == "scratch_candidate_cleanup":
        reason = "Reduce small scratch-like artifacts where safe"
    elif tool == "small_defect_heal":
        reason = "Use local healing for small detected defect regions after coarse cleanup"
    elif tool == "non_local_means_denoise":
        reason = "Reduce visible scan or sensor noise"
    elif tool == "bilateral_denoise":
        reason = "Use a gentler edge-preserving denoise after prior rollback"
    elif tool == "bounded_histogram_match_to_reference":
        reason = "Bias tonal feel toward the reference without copying content"
    elif tool == "texture_softness_bias_from_reference":
        reason = "Match the reference softness or crispness feel at a low level"
    elif tool == "unsharp_mask":
        reason = "Add light sharpening after cleanup"
    return PlanStep(tool=tool, params=params, reason=reason)


def _acceptance_for_tool(tool: str) -> list[str]:
    mapping = {
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
        "auto_white_balance": 20,
        "shadow_highlight_balance": 30,
        "clahe_contrast": 35,
        "histogram_balance": 36,
        "dust_cleanup": 40,
        "scratch_candidate_cleanup": 45,
        "small_defect_heal": 50,
        "stroke_paint": 55,
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
