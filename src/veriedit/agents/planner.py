from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from veriedit.io.writer import append_jsonl
from veriedit.llm import GeminiStructuredClient, has_gemini_support
from veriedit.schemas import AgentLog, EditPlan, PlanStep, WorkflowState
from veriedit.tools import build_tool_registry


class PlannerAgent:
    def __init__(self, model: str | None = None) -> None:
        self.registry = build_tool_registry()
        self.model = model

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        plan = self._plan_with_gemini(state) or self._heuristic_plan(state)
        state["plan"] = plan.model_dump()
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
            "tool_registry": [spec.model_dump(exclude={"operation"}) for spec in self.registry.specs()],
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
        steps: list[PlanStep] = []
        acceptance: list[str] = []
        must_avoid = ["oversmoothing", "halo artifacts", "semantic content changes"]
        should_deskew = any(word in prompt for word in ("deskew", "de-skew", "straighten", "rotate", "align scan"))
        plausible_skew = 0.5 <= abs(float(diagnostics["skew_angle"])) <= 12.0
        if should_deskew and plausible_skew:
            steps.append(PlanStep(tool="deskew", params={"angle": diagnostics["skew_angle"]}, reason="Correct noticeable skew"))
            acceptance.append("skew reduced")
        if "yellow" in prompt or "white balance" in prompt or diagnostics["yellow_cast"] > 0.58:
            steps.append(
                PlanStep(
                    tool="auto_white_balance",
                    params={"strength": 0.7 if "natural" not in prompt else 0.55},
                    reason="Reduce color cast conservatively",
                )
            )
            acceptance.append("yellow cast reduced")
        if "faded" in prompt or "contrast" in prompt or diagnostics["contrast_score"] < 0.52 or diagnostics["fade_score"] > 0.42:
            if "shadow_highlight_balance" not in blocked_tools:
                steps.append(
                    PlanStep(
                        tool="shadow_highlight_balance",
                        params={"shadow_lift": 0.16, "highlight_compress": 0.1, "blur_sigma": 16.0},
                        reason="Recover faded tonal separation without aggressive clipping",
                    )
                )
            steps.append(PlanStep(tool="clahe_contrast", params={"clip_limit": 1.8}, reason="Gently restore local contrast"))
            acceptance.append("contrast improved without clipping")
        if "dust" in prompt or "speck" in prompt or diagnostics["dust_candidates"] > 24:
            steps.append(
                PlanStep(
                    tool="dust_cleanup",
                    params={"max_area": 20, "sensitivity": 0.45},
                    reason="Remove isolated scan dust and speckles",
                )
            )
            acceptance.append("dust visibly reduced")
        if "scratch" in prompt or diagnostics["dust_candidates"] > 80 or diagnostics["scratch_candidates"] > 12:
            steps.append(
                PlanStep(
                    tool="scratch_candidate_cleanup",
                    params={"max_area": 72, "sensitivity": 0.35},
                    reason="Reduce small scratch-like artifacts where safe",
                )
            )
        if diagnostics["dust_candidates"] > 180 or diagnostics["scratch_candidates"] > 20 or diagnostics["edge_damage_ratio"] > 0.08:
            if "small_defect_heal" not in blocked_tools:
                steps.append(
                    PlanStep(
                        tool="small_defect_heal",
                        params={"max_area": 28, "sensitivity": 0.4, "radius": 2.0},
                        reason="Use local healing for small defects and pits after coarse cleanup",
                    )
                )
        if "noise" in prompt or "grain" in prompt or diagnostics["noise_score"] > 0.08:
            if "non_local_means_denoise" in blocked_tools and "bilateral_denoise" not in blocked_tools:
                steps.append(
                    PlanStep(
                        tool="bilateral_denoise",
                        params={"diameter": 7, "sigma_color": 24.0, "sigma_space": 7.0},
                        reason="Use a gentler edge-preserving denoise after prior rollback",
                    )
                )
            else:
                steps.append(
                    PlanStep(
                        tool="non_local_means_denoise",
                        params={"h": 5.0 if "natural" in prompt else 6.0},
                        reason="Reduce visible scan or sensor noise",
                    )
                )
            acceptance.append("noise reduced while retaining texture")
        if state["reference_image_path"]:
            steps.append(
                PlanStep(
                    tool="bounded_histogram_match_to_reference",
                    params={"strength": 0.28},
                    reason="Bias tonal feel toward the reference without copying content",
                )
            )
            steps.append(
                PlanStep(
                    tool="texture_softness_bias_from_reference",
                    params={},
                    reason="Match the reference softness or crispness feel at a low level",
                )
            )
            acceptance.append("reference style reflected only in tone and texture")
        sharpen_requested = any(word in prompt for word in ("sharp", "clarity", "detail"))
        if sharpen_requested or ("blur" in prompt and diagnostics["blur_score"] < 120):
            amount = 0.25 if "natural" in prompt else 0.35
            if retry.get("strategy") and "reduce sharpen" in retry["strategy"].lower():
                amount *= 0.7
            steps.append(
                PlanStep(
                    tool="unsharp_mask",
                    params={"radius": 1.0, "amount": round(amount, 2)},
                    reason="Add light sharpening after cleanup",
                )
            )
            acceptance.append("detail remains natural")
        if not steps:
            steps.append(PlanStep(tool="histogram_balance", params={"strength": 0.4}, reason="Apply a safe baseline tonal normalization"))
            acceptance.append("overall balance improved naturally")
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
