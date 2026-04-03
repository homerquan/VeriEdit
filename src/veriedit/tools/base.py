from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


ToolCallable = Callable[[np.ndarray, dict[str, Any], Optional[np.ndarray]], tuple[np.ndarray, dict[str, Any]]]


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    safety_notes: list[str]
    edit_scope: str = "global"
    capability_tags: list[str] = Field(default_factory=list)
    parameter_bounds: dict[str, tuple[float, float] | tuple[int, int] | list[str]] = Field(default_factory=dict)
    expected_effect: str
    likely_failure_modes: list[str] = Field(default_factory=list)
    reversibility_notes: str
    operation: ToolCallable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name] for name in self.names()]


def sanitize_numeric_params(params: dict[str, Any], bounds: dict[str, tuple[float, float] | tuple[int, int] | list[str]]) -> dict[str, Any]:
    sanitized = dict(params)
    for name, rule in bounds.items():
        if name not in sanitized:
            continue
        value = sanitized[name]
        if isinstance(rule, list):
            if value not in rule:
                sanitized[name] = rule[0]
            continue
        lower, upper = rule
        if isinstance(lower, int) and isinstance(upper, int):
            sanitized[name] = int(min(max(int(value), lower), upper))
        else:
            sanitized[name] = float(min(max(float(value), float(lower)), float(upper)))
    return sanitized
