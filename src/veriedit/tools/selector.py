from __future__ import annotations

from typing import Any

from veriedit.schemas import ToolRecommendation
from veriedit.tools.base import ToolRegistry, ToolSpec


def rank_tools(
    *,
    registry: ToolRegistry,
    prompt: str,
    diagnostics: dict[str, Any],
    region_summary: dict[str, Any],
    retry_context: dict[str, Any] | None,
    blocked_tools: set[str],
    has_reference: bool = False,
) -> list[ToolRecommendation]:
    prompt_lower = prompt.lower()
    retry_text = " ".join(
        str(value) for key, value in (retry_context or {}).items() if key in {"reason", "strategy"}
    ).lower()
    recommendations: list[ToolRecommendation] = []
    source = diagnostics.get("source", {})
    for spec in registry.specs():
        if spec.name in blocked_tools:
            continue
        score, rationale, params_hint = _score_tool(spec, prompt_lower, source, region_summary, retry_text, has_reference)
        if score <= 0.0:
            continue
        priority = "primary" if score >= 1.8 else "secondary" if score >= 1.0 else "fallback"
        recommendations.append(
            ToolRecommendation(
                tool=spec.name,
                score=round(score, 4),
                rationale=rationale,
                priority=priority,
                params_hint=params_hint,
                mode_hint="masked_local_repair" if spec.edit_scope == "local" else "reference_guided" if "reference" in spec.capability_tags else "global",
            )
        )
    recommendations.sort(key=lambda item: item.score, reverse=True)
    return recommendations[:10]


def _score_tool(
    spec: ToolSpec,
    prompt: str,
    source: dict[str, Any],
    region_summary: dict[str, Any],
    retry_text: str,
    has_reference: bool,
) -> tuple[float, str, dict[str, Any]]:
    score = 0.0
    reasons: list[str] = []
    params_hint: dict[str, Any] = {}
    tags = set(spec.capability_tags) | set(_infer_tags(spec))
    prompt_requests_local_repair = any(word in prompt for word in ("repair", "retouch", "patch", "local", "damaged", "fix"))
    if "reference" in tags and not has_reference:
        return 0.0, "reference tool skipped without reference image", {}
    if "yellow cast" in prompt or "white balance" in prompt or source.get("yellow_cast", 0.0) > 0.58:
        if "color_cast" in tags:
            score += 2.2
            reasons.append("match color-cast correction need")
    if "contrast" in prompt or "faded" in prompt or source.get("contrast_score", 1.0) < 0.52 or source.get("fade_score", 0.0) > 0.42:
        if "contrast" in tags or "tone" in tags:
            score += 1.8
            reasons.append("match tonal recovery need")
    if "dust" in prompt or "speck" in prompt or source.get("dust_candidates", 0) > 24:
        if "dust" in tags:
            score += 2.0
            reasons.append("match dust cleanup need")
    if "scratch" in prompt or source.get("scratch_candidates", 0) > 12:
        if "scratch" in tags:
            score += 2.0
            reasons.append("match scratch cleanup need")
    if source.get("noise_score", 0.0) > 0.08 or "noise" in prompt or "grain" in prompt:
        if "denoise" in tags:
            score += 1.5
            reasons.append("match noise reduction need")
    if "sharp" in prompt or "detail" in prompt or "clarity" in prompt:
        if "sharpen" in tags:
            score += 1.4
            reasons.append("match detail enhancement request")
    if source.get("edge_damage_ratio", 0.0) > 0.08 or float(region_summary.get("largest_defect_ratio", 0.0)) > 0.0008:
        if "local_repair" in tags:
            score += 1.6
            reasons.append("region summary favors localized repair")
    if "reference" in tags and ("reference" in prompt or "match" in prompt):
        score += 0.8
        reasons.append("reference-guided tone/texture may help")
    if prompt_requests_local_repair:
        if "local_repair" in tags:
            score += 2.4
            reasons.append("prompt explicitly requests local repair")
        elif spec.edit_scope == "global" and not any(tag in tags for tag in ("color_cast", "contrast", "tone")):
            score *= 0.4
            reasons.append("global non-repair tool deprioritized for local repair prompt")
    if prompt_requests_local_repair and spec.edit_scope == "global" and any(tag in tags for tag in ("contrast", "tone")) and not any(
        word in prompt for word in ("contrast", "tone", "yellow", "white balance", "noise", "grain", "sharp", "detail")
    ):
        score *= 0.45
        reasons.append("broad tonal edit deprioritized because prompt does not ask for it")

    if "reduce sharpen" in retry_text and "sharpen" in tags:
        score *= 0.45
        reasons.append("deprioritized after reviewer sharpen warning")
    if "stop further denoise" in retry_text and "denoise" in tags:
        score *= 0.4
        reasons.append("deprioritized after reviewer denoise warning")
    if "prefer local heal" in retry_text:
        if "local_repair" in tags:
            score += 1.2
            reasons.append("retry strategy prefers local repair")
        elif spec.edit_scope == "global":
            score *= 0.65
            reasons.append("global edit deprioritized by retry strategy")
    if spec.name == "stroke_paint" and prompt_requests_local_repair:
        score += 0.8
        reasons.append("closed-loop stroke repair preferred for autonomous local repair")

    if spec.name == "stroke_paint":
        params_hint = {"stroke_budget": 10 if "natural" in prompt else 14, "candidate_count": 14, "pen": "soft"}
    elif spec.name == "auto_white_balance":
        params_hint = {"strength": 0.55 if "natural" in prompt else 0.7}
    elif spec.name == "shadow_highlight_balance":
        params_hint = {"shadow_lift": 0.16, "highlight_compress": 0.1, "blur_sigma": 16.0}
    elif spec.name == "clahe_contrast":
        params_hint = {"clip_limit": 1.8}
    elif spec.name == "dust_cleanup":
        params_hint = {"max_area": 20, "sensitivity": 0.45}
    elif spec.name == "scratch_candidate_cleanup":
        params_hint = {"max_area": 72, "sensitivity": 0.35}
    elif spec.name == "small_defect_heal":
        params_hint = {"max_area": 28, "sensitivity": 0.4, "radius": 2.0}
    elif spec.name == "non_local_means_denoise":
        params_hint = {"h": 5.0 if "natural" in prompt else 6.0}
    elif spec.name == "bilateral_denoise":
        params_hint = {"diameter": 7, "sigma_color": 24.0, "sigma_space": 7.0}
    elif spec.name == "unsharp_mask":
        params_hint = {"radius": 1.0, "amount": 0.25 if "natural" in prompt else 0.35}
    elif spec.name == "bounded_histogram_match_to_reference":
        params_hint = {"strength": 0.28}

    return score, "; ".join(reasons) or spec.expected_effect, params_hint


def _infer_tags(spec: ToolSpec) -> list[str]:
    name = spec.name
    tags: list[str] = []
    if "white_balance" in name:
        tags.append("color_cast")
    if any(token in name for token in ("contrast", "gamma", "histogram", "shadow_highlight")):
        tags.extend(["tone", "contrast"])
    if "dust" in name:
        tags.extend(["dust", "local_repair"])
    if "scratch" in name:
        tags.extend(["scratch", "local_repair"])
    if any(token in name for token in ("heal", "clone", "stroke_paint")):
        tags.append("local_repair")
    if "denoise" in name or "cleanup" in name:
        tags.append("denoise")
    if "sharpen" in name or name == "unsharp_mask":
        tags.append("sharpen")
    if "reference" in name:
        tags.append("reference")
    return tags
