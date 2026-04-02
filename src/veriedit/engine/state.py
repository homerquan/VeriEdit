from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import StrokeAction


@dataclass
class PatchBBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    def clamp(self, shape: tuple[int, int]) -> "PatchBBox":
        height, width = shape
        x0 = int(np.clip(self.x0, 0, width))
        y0 = int(np.clip(self.y0, 0, height))
        x1 = int(np.clip(self.x1, x0 + 1, width))
        y1 = int(np.clip(self.y1, y0 + 1, height))
        return PatchBBox(x0=x0, y0=y0, x1=x1, y1=y1)


@dataclass
class EngineState:
    canvas: np.ndarray
    target: np.ndarray
    vector_history: list[StrokeAction] = field(default_factory=list)
    active_patch_bbox: PatchBBox | None = None
    canvas_patch: np.ndarray | None = None
    target_patch: np.ndarray | None = None
    pen_position: tuple[float, float] = (0.0, 0.0)
    pen_down: bool = False
    recent_strokes: list[StrokeAction] = field(default_factory=list)
    mode: str = "contour"
    step_index: int = 0
