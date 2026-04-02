from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class WorkflowConfig(BaseModel):
    artifact_root: Path = Field(default=Path("runs"))
    prompt_satisfaction_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    artifact_risk_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    change_ratio_warning_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    default_llm_model: str = "gemini-3-flash"
    fallback_llm_model: str = "gemini-2.5-flash"
    keep_masks: bool = True
