from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .controller import ControllerStepRecord, MPCController
from .selector import PatchSelector
from .state import EngineState
from .utils import to_grayscale_array


@dataclass
class EngineConfig:
    patch_size: int = 48
    max_patches: int = 12
    candidates_per_step: int = 8
    commit_fraction: float = 0.25
    max_micro_steps: int = 10
    debug_dir: Path | None = None


@dataclass
class EngineResult:
    state: EngineState
    patch_records: list[dict[str, object]] = field(default_factory=list)


class ClosedLoopStrokeEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.selector = PatchSelector(patch_size=self.config.patch_size)
        self.controller = MPCController(
            commit_fraction=self.config.commit_fraction,
            max_micro_steps=self.config.max_micro_steps,
            debug_dir=self.config.debug_dir,
        )

    def run(self, target: np.ndarray, canvas: np.ndarray | None = None) -> EngineResult:
        target_gray = to_grayscale_array(target)
        if canvas is None:
            canvas_gray = np.ones_like(target_gray, dtype=np.float32)
        else:
            canvas_gray = to_grayscale_array(canvas)
        state = EngineState(canvas=canvas_gray, target=target_gray)
        patch_records: list[dict[str, object]] = []
        for _ in range(self.config.max_patches):
            bbox = self.selector.select_patch(state.canvas, state.target)
            state.pen_position = (bbox.width / 2.0, bbox.height / 2.0)
            state, records = self.controller.run_patch(state, bbox, k=self.config.candidates_per_step)
            patch_records.append(
                {
                    "bbox": bbox,
                    "steps": records,
                    "residual": float(np.mean(np.clip(state.target - (1.0 - state.canvas), 0.0, 1.0))),
                }
            )
            if self._done(state):
                break
        return EngineResult(state=state, patch_records=patch_records)

    def _done(self, state: EngineState) -> bool:
        residual = np.clip(state.target - (1.0 - state.canvas), 0.0, 1.0)
        return float(np.mean(residual)) <= 0.02
