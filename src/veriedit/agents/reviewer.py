from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from veriedit.io.loader import load_image
from veriedit.io.writer import append_jsonl
from veriedit.llm import GeminiStructuredClient, has_gemini_support
from veriedit.metrics.iq_metrics import style_profile_from_image, summarize_image_quality
from veriedit.metrics.similarity import compare_images
from veriedit.schemas import AgentLog, ReviewResult, WorkflowState


class ReviewerAgent:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def run(self, state: WorkflowState) -> WorkflowState:
        start = time.perf_counter()
        review = self._review_with_gemini(state) or self._heuristic_review(state)
        state["review"] = review.model_dump()
        self._log(
            state,
            AgentLog(
                run_id=state["run_id"],
                agent_name="reviewer",
                iteration=state["iteration"],
                input_summary={"prompt": state["prompt"][:200], "step_count": len(state["executed_steps"])},
                decision=review.status,
                output_summary=review.model_dump(),
                latency_ms=(time.perf_counter() - start) * 1000,
            ),
        )
        return state

    def _review_with_gemini(self, state: WorkflowState) -> ReviewResult | None:
        model = self.model or state["request"].get("llm_model")
        if not model or not has_gemini_support():
            return None
        payload = {
            "prompt": state["prompt"],
            "plan": state["plan"],
            "policy_status": state["policy_status"],
            "diagnostics": state["diagnostics"],
            "executed_steps": state["executed_steps"][-8:],
            "reference_style_profile": state["style_profile"],
        }
        system_prompt = (
            "You are the Reviewer Agent for a non-generative image editing workflow. "
            "Judge realism over aggression. Penalize over-sharpening and over-smoothing. "
            "Distinguish style influence from content copying. Flag suspicious semantic changes. "
            "Return JSON matching the ReviewResult schema."
        )
        try:  # pragma: no cover - depends on external API
            response = GeminiStructuredClient(model=model).generate_json(system_prompt, payload)
            return ReviewResult(**response)
        except (RuntimeError, ValidationError, ValueError, KeyError):
            return None

    def _heuristic_review(self, state: WorkflowState) -> ReviewResult:
        source_image, source_meta = load_image(state["source_image_path"])
        current_image, current_meta = load_image(state["current_image_path"])
        source_summary = summarize_image_quality(source_image, source_meta)
        current_summary = summarize_image_quality(current_image, current_meta)
        deltas = compare_images(source_image, current_image)
        reference_alignment = 0.0
        if state["reference_image_path"]:
            reference_image, _ = load_image(state["reference_image_path"])
            reference_style = style_profile_from_image(reference_image)
            current_style = style_profile_from_image(current_image)
            style_delta = sum(abs(current_style[key] - reference_style[key]) for key in reference_style) / len(reference_style)
            reference_alignment = max(0.0, 1.0 - style_delta)
        prompt_score = _estimate_prompt_score(state["prompt"], state["diagnostics"]["source"], source_summary, current_summary, reference_alignment)
        artifact_risk = _estimate_artifact_risk(state["prompt"], state["plan"], source_summary, current_summary, deltas)
        semantic_fabrication_risk = _estimate_semantic_fabrication_risk(state["prompt"], state["plan"], deltas)
        findings = _build_findings(state["prompt"], state["plan"], source_summary, current_summary, deltas)
        recommendations = _build_recommendations(state["plan"], source_summary, current_summary, artifact_risk)
        status = "accept" if prompt_score >= 0.72 and artifact_risk <= 0.35 and semantic_fabrication_risk <= 0.55 else "revise"
        if state["iteration"] >= state["max_iterations"] and status != "accept":
            status = "stop"
        return ReviewResult(
            status=status,
            prompt_score=prompt_score,
            artifact_risk=artifact_risk,
            naturalness_score=max(0.0, 1.0 - artifact_risk),
            semantic_fabrication_risk=semantic_fabrication_risk,
            findings=findings,
            recommendations=recommendations,
            confidence=0.7 if state["reference_image_path"] else 0.8,
        )

    def _log(self, state: WorkflowState, record: AgentLog) -> None:
        state["logs"].append(record.model_dump())
        append_jsonl(record.model_dump(), f'{state["run_dir"]}/agent_logs.jsonl')


def _estimate_prompt_score(
    prompt: str,
    diagnostics: dict[str, Any],
    source_summary: dict[str, Any],
    current_summary: dict[str, Any],
    reference_alignment: float,
) -> float:
    prompt_lower = prompt.lower()
    score = 0.45
    if "yellow" in prompt_lower or "white balance" in prompt_lower:
        improvement = max(0.0, diagnostics["yellow_cast"] - current_summary["yellow_cast"])
        score += min(0.18, improvement * 0.8)
    if "dust" in prompt_lower or "speck" in prompt_lower:
        dust_improvement = max(0.0, diagnostics["dust_candidates"] - current_summary["dust_candidates"])
        score += min(0.18, dust_improvement / 160.0)
    if "noise" in prompt_lower or "grain" in prompt_lower:
        noise_improvement = max(0.0, source_summary["noise_score"] - current_summary["noise_score"])
        score += min(0.12, noise_improvement * 3.0)
    if "contrast" in prompt_lower or "faded" in prompt_lower:
        contrast_improvement = max(0.0, current_summary["contrast_score"] - source_summary["contrast_score"])
        score += min(0.12, contrast_improvement * 0.5)
        fade_improvement = max(0.0, source_summary["fade_score"] - current_summary["fade_score"])
        score += min(0.08, fade_improvement * 0.5)
    if "sharp" in prompt_lower or "detail" in prompt_lower:
        sharpness_improvement = max(0.0, current_summary["blur_score"] - source_summary["blur_score"])
        score += min(0.12, sharpness_improvement / 200.0)
    if reference_alignment:
        score += min(0.08, reference_alignment * 0.08)
    if "natural" in prompt_lower or "realistic" in prompt_lower:
        score += 0.05
    return float(min(0.99, score))


def _estimate_artifact_risk(
    prompt: str,
    plan: dict[str, Any] | None,
    source_summary: dict[str, Any],
    current_summary: dict[str, Any],
    deltas: dict[str, float],
) -> float:
    prompt_lower = prompt.lower()
    plan_tools = [step["tool"] for step in (plan or {}).get("steps", [])]
    global_tone_edit = any(word in prompt_lower for word in ("contrast", "tone", "faded", "exposure")) or any(
        tool in plan_tools for tool in ("histogram_balance", "clahe_contrast", "shadow_highlight_balance", "gamma_adjust")
    )
    risk = 0.1
    if current_summary["noise_score"] > source_summary["noise_score"] + 0.04:
        risk += 0.18
    if current_summary["blur_score"] < source_summary["blur_score"] * 0.5:
        risk += 0.2
    risk += max(0.0, current_summary["clipping_highlights"] - source_summary["clipping_highlights"]) * 1.4
    risk += max(0.0, current_summary["clipping_shadows"] - source_summary["clipping_shadows"]) * 1.4
    change_penalty_scale = 0.35 if global_tone_edit else 0.7
    change_penalty_offset = 0.5 if global_tone_edit else 0.35
    risk += max(0.0, deltas["change_area_ratio"] - change_penalty_offset) * change_penalty_scale
    risk += max(0.0, 0.82 - deltas["ssim"]) * 0.6
    risk += max(0.0, current_summary["edge_damage_ratio"] - source_summary["edge_damage_ratio"]) * 1.2
    return float(min(0.99, risk))


def _estimate_semantic_fabrication_risk(prompt: str, plan: dict[str, Any] | None, deltas: dict[str, float]) -> float:
    prompt_lower = prompt.lower()
    plan_tools = [step["tool"] for step in (plan or {}).get("steps", [])]
    global_tone_edit = any(word in prompt_lower for word in ("contrast", "tone", "faded", "exposure")) or any(
        tool in plan_tools for tool in ("histogram_balance", "clahe_contrast", "shadow_highlight_balance", "gamma_adjust")
    )
    risk = deltas["change_area_ratio"] * (0.7 if global_tone_edit else 1.4)
    risk += max(0.0, 0.8 - deltas["ssim"]) * 0.5
    return float(min(1.0, max(0.0, risk)))


def _build_findings(
    prompt: str,
    plan: dict[str, Any] | None,
    source_summary: dict[str, Any],
    current_summary: dict[str, Any],
    deltas: dict[str, float],
) -> list[str]:
    prompt_lower = prompt.lower()
    plan_tools = [step["tool"] for step in (plan or {}).get("steps", [])]
    global_tone_edit = any(word in prompt_lower for word in ("contrast", "tone", "faded", "exposure")) or any(
        tool in plan_tools for tool in ("histogram_balance", "clahe_contrast", "shadow_highlight_balance", "gamma_adjust")
    )
    findings: list[str] = []
    if current_summary["yellow_cast"] < source_summary["yellow_cast"] - 0.03:
        findings.append("yellow cast improved")
    if current_summary["dust_candidates"] < source_summary["dust_candidates"]:
        findings.append("small dust mostly reduced")
    if current_summary["noise_score"] < source_summary["noise_score"] - 0.01:
        findings.append("noise reduced")
    if current_summary["fade_score"] < source_summary["fade_score"] - 0.03:
        findings.append("faded tonal separation improved")
    if current_summary["scratch_candidates"] < source_summary["scratch_candidates"] - 4:
        findings.append("scratch-like defects reduced")
    if deltas["change_area_ratio"] > (0.6 if global_tone_edit else 0.45):
        findings.append("edit footprint is broader than ideal")
    if current_summary["clipping_highlights"] > source_summary["clipping_highlights"] + 0.03:
        findings.append("highlight clipping increased slightly")
    if current_summary["blur_score"] < source_summary["blur_score"] * 0.55:
        findings.append("some detail softened more than expected")
    if not findings:
        findings.append("result is stable with conservative changes")
    return findings


def _build_recommendations(plan: dict[str, Any] | None, source_summary: dict[str, Any], current_summary: dict[str, Any], artifact_risk: float) -> list[str]:
    recommendations: list[str] = []
    step_tools = [step["tool"] for step in (plan or {}).get("steps", [])]
    if artifact_risk > 0.35 and ("unsharp_mask" in step_tools or "edge_preserving_sharpen" in step_tools):
        recommendations.append("reduce sharpen amount by 30%")
    if current_summary["blur_score"] < source_summary["blur_score"] * 0.55 and any("denoise" in tool for tool in step_tools):
        recommendations.append("stop further denoise")
    if current_summary["clipping_highlights"] > source_summary["clipping_highlights"] + 0.03:
        recommendations.append("lower contrast or histogram intensity")
    if current_summary["edge_damage_ratio"] > source_summary["edge_damage_ratio"] + 0.02:
        recommendations.append("prefer local heal over broad tonal edits")
    if not recommendations:
        recommendations.append("no additional changes required")
    return recommendations
