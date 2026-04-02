from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrokeAction:
    type: str
    points: list[tuple[float, float]]
    width: float
    opacity: float
    pressure: float
    color: float
    pen_down: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
