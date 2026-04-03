from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class EditRequest(BaseModel):
    source_image: str
    prompt: str
    reference_image: str | None = None
    output_path: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=3, ge=1, le=10)
    preserve_metadata: bool = False
    save_intermediates: bool = True
    llm_model: str = "gemini-3-flash"
    enable_human_approval: bool = True


class PolicyStatus(BaseModel):
    status: Literal["allow", "caution", "reject"]
    risk_level: Literal["low", "medium", "high"]
    constraints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason: str | None = None


class SourceDiagnostics(BaseModel):
    width: int
    height: int
    mode: str
    bit_depth: int
    blur_score: float
    noise_score: float
    yellow_cast: float
    contrast_score: float
    clipping_highlights: float
    clipping_shadows: float
    skew_angle: float
    dust_candidates: int
    scratch_candidates: int
    fade_score: float
    sepia_score: float
    edge_damage_ratio: float
    underexposed: bool


class StyleProfile(BaseModel):
    warmth: float = 0.5
    contrast: float = 0.5
    sharpness_feel: float = 0.5
    grain_level: float = 0.0
    saturation: float = 0.5


class DiagnosticsBundle(BaseModel):
    source: SourceDiagnostics
    reference: StyleProfile | None = None
    regions: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class PlanStep(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str


class ToolRecommendation(BaseModel):
    tool: str
    score: float = Field(ge=0.0)
    rationale: str
    priority: Literal["primary", "secondary", "fallback"] = "secondary"
    params_hint: dict[str, Any] = Field(default_factory=dict)
    mode_hint: Literal["global", "masked_local_repair", "reference_guided", "manual"] = "global"


class EditPlan(BaseModel):
    objective: str
    must_preserve: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    recommended_tools: list[ToolRecommendation] = Field(default_factory=list)


class AgentHandoff(BaseModel):
    from_agent: str
    to_agent: str
    iteration: int
    summary: str
    key_points: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(BaseModel):
    step_index: int
    tool: str
    params: dict[str, Any]
    execution_mode: str = "global"
    mask_name: str | None = None
    mask_coverage: float = 0.0
    before_metrics: dict[str, float] = Field(default_factory=dict)
    after_metrics: dict[str, float] = Field(default_factory=dict)
    output_path: str
    status: Literal["ok", "rolled_back", "failed"]
    notes: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    status: Literal["accept", "revise", "stop"]
    prompt_score: float = Field(ge=0.0, le=1.0)
    artifact_risk: float = Field(ge=0.0, le=1.0)
    naturalness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_fabrication_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_metrics: dict[str, float] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class RetryDecision(BaseModel):
    decision: Literal["accept", "retry", "stop"]
    reason: str
    strategy: str | None = None


class HumanReviewRequest(BaseModel):
    status: Literal["not_needed", "pending", "approved", "rejected"] = "not_needed"
    reason: str | None = None
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    manual_eval_md: str | None = None
    approval_json: str | None = None


class FinalResult(BaseModel):
    success: bool
    output_image: str | None = None
    report_json: str | None = None
    report_md: str | None = None
    summary_md: str | None = None
    observation_json: str | None = None
    observation_md: str | None = None
    iterations: int = 0
    applied_tools: list[str] = Field(default_factory=list)
    review_summary: str = ""
    stop_reason: str | None = None
    run_dir: str | None = None
    human_review_status: str | None = None
    human_review_reason: str | None = None
    manual_eval_md: str | None = None
    human_approval_json: str | None = None


class AgentLog(BaseModel):
    run_id: str
    agent_name: str
    iteration: int
    input_summary: dict[str, Any]
    decision: str
    output_summary: dict[str, Any]
    latency_ms: float


class EditResult(BaseModel):
    success: bool
    output_image: str | None = None
    report_json: str | None = None
    report_md: str | None = None
    summary_md: str | None = None
    observation_json: str | None = None
    observation_md: str | None = None
    iterations: int = 0
    applied_tools: list[str] = Field(default_factory=list)
    review_summary: str = ""
    stop_reason: str | None = None
    run_dir: str | None = None
    run_id: str | None = None
    human_review_status: str | None = None
    human_review_reason: str | None = None
    manual_eval_md: str | None = None
    human_approval_json: str | None = None


class WorkflowState(TypedDict):
    run_id: str
    request: dict[str, Any]
    source_image_path: str
    reference_image_path: str | None
    prompt: str
    output_path: str
    run_dir: str
    current_image_path: str
    policy_status: dict[str, Any]
    diagnostics: dict[str, Any]
    diagnostic_artifacts: dict[str, str]
    style_profile: dict[str, Any] | None
    plan: dict[str, Any] | None
    executed_steps: list[dict[str, Any]]
    agent_handoffs: list[dict[str, Any]]
    observation_trace: list[dict[str, Any]]
    intermediate_paths: list[str]
    review: dict[str, Any] | None
    human_review: dict[str, Any] | None
    retry_decision: dict[str, Any] | None
    final_result: dict[str, Any] | None
    logs: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    stop_reason: str | None


class RunArtifacts(BaseModel):
    run_id: str
    run_dir: Path
    source_copy: Path
    reference_copy: Path | None = None
    output_image: Path
    report_json: Path
    report_md: Path
    agent_logs: Path
