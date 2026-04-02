from __future__ import annotations

import numpy as np

from .actions import StrokeAction
from .state import EngineState


class LocalCritic:
    def __init__(
        self,
        edge_alignment_weight: float = 1.6,
        coverage_weight: float = 1.8,
        overdraw_weight: float = 1.2,
        length_weight: float = 0.08,
        jitter_weight: float = 0.18,
    ) -> None:
        self.edge_alignment_weight = edge_alignment_weight
        self.coverage_weight = coverage_weight
        self.overdraw_weight = overdraw_weight
        self.length_weight = length_weight
        self.jitter_weight = jitter_weight

    def score(
        self,
        prev_patch: np.ndarray,
        next_patch: np.ndarray,
        target_patch: np.ndarray,
        state: EngineState,
        action: StrokeAction,
    ) -> float:
        target_edge = target_patch > 0.4
        prev_dark = 1.0 - prev_patch
        next_dark = 1.0 - next_patch
        missing_before = np.clip(target_patch - prev_dark, 0.0, 1.0)
        missing_after = np.clip(target_patch - next_dark, 0.0, 1.0)
        edge_alignment_error = float(np.mean(missing_after[target_edge])) if np.any(target_edge) else float(np.mean(missing_after))
        coverage_penalty = float(np.mean(missing_after))
        overdraw_penalty = float(np.mean(np.clip(next_dark - target_patch, 0.0, 1.0)))
        stroke_length_penalty = self._stroke_length(action) / max(8.0, float(max(target_patch.shape)))
        jitter_penalty = self._jitter_penalty(state, action)
        improvement_bonus = float(np.mean(missing_before) - np.mean(missing_after))
        total = (
            self.edge_alignment_weight * edge_alignment_error
            + self.coverage_weight * coverage_penalty
            + self.overdraw_weight * overdraw_penalty
            + self.length_weight * stroke_length_penalty
            + self.jitter_weight * jitter_penalty
            - 1.5 * improvement_bonus
        )
        return float(total)

    def _stroke_length(self, action: StrokeAction) -> float:
        if len(action.points) < 2:
            return 0.0
        return float(
            sum(
                np.hypot(action.points[idx + 1][0] - action.points[idx][0], action.points[idx + 1][1] - action.points[idx][1])
                for idx in range(len(action.points) - 1)
            )
        )

    def _jitter_penalty(self, state: EngineState, action: StrokeAction) -> float:
        if not state.recent_strokes or len(action.points) < 2:
            return 0.0
        previous = state.recent_strokes[-1]
        if len(previous.points) < 2:
            return 0.0
        prev_vec = np.array(previous.points[-1], dtype=np.float32) - np.array(previous.points[-2], dtype=np.float32)
        curr_vec = np.array(action.points[1], dtype=np.float32) - np.array(action.points[0], dtype=np.float32)
        prev_norm = float(np.linalg.norm(prev_vec))
        curr_norm = float(np.linalg.norm(curr_vec))
        if prev_norm < 1e-6 or curr_norm < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(prev_vec, curr_vec) / (prev_norm * curr_norm), -1.0, 1.0))
        return 1.0 - ((cosine + 1.0) / 2.0)
