from __future__ import annotations

import numpy as np

from .actions import StrokeAction
from .state import EngineState


class StrokeProposer:
    def propose(self, state: EngineState, k: int = 8) -> list[StrokeAction]:
        if state.canvas_patch is None or state.target_patch is None:
            return []
        residual = np.clip(state.target_patch - (1.0 - state.canvas_patch), 0.0, 1.0)
        if not np.any(residual > 0.05):
            return []
        gy, gx = np.gradient(state.target_patch.astype(np.float32))
        cutoff = float(np.percentile(residual[residual > 0], 75))
        ys, xs = np.where(residual >= cutoff)
        if len(xs) == 0:
            return []
        scores = residual[ys, xs]
        order = np.argsort(scores)[::-1][: max(1, k // 2)]
        actions: list[StrokeAction] = []
        prev_direction = self._recent_direction(state)
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            tangent = np.array([-gy[y, x], gx[y, x]], dtype=np.float32)
            tangent = self._normalize(tangent, fallback=prev_direction)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            length = float(np.clip(6.0 + residual[y, x] * 16.0, 6.0, 16.0))
            curvature = float(np.clip(np.hypot(gx[y, x], gy[y, x]) * 2.5, 0.8, 4.0))
            actions.extend(
                [
                    self._bezier_from_anchor(x, y, tangent, normal, length, 0.0, width=2.2),
                    self._bezier_from_anchor(x, y, tangent, normal, length, curvature, width=2.4),
                    self._bezier_from_anchor(x, y, tangent, normal, length, -curvature, width=2.4),
                    self._correction_stroke(state, x, y, width=1.8),
                ]
            )
        unique: list[StrokeAction] = []
        seen: set[tuple[int, ...]] = set()
        for action in actions:
            key = tuple(int(round(value)) for point in action.points for value in point)
            if key in seen:
                continue
            seen.add(key)
            unique.append(action)
            if len(unique) >= k:
                break
        return unique

    def _recent_direction(self, state: EngineState) -> np.ndarray:
        if not state.recent_strokes:
            return np.array([1.0, 0.0], dtype=np.float32)
        points = state.recent_strokes[-1].points
        if len(points) < 2:
            return np.array([1.0, 0.0], dtype=np.float32)
        delta = np.array(points[-1], dtype=np.float32) - np.array(points[-2], dtype=np.float32)
        return self._normalize(delta, fallback=np.array([1.0, 0.0], dtype=np.float32))

    def _bezier_from_anchor(
        self,
        x: int,
        y: int,
        tangent: np.ndarray,
        normal: np.ndarray,
        length: float,
        bend: float,
        width: float,
    ) -> StrokeAction:
        anchor = np.array([x, y], dtype=np.float32)
        p0 = anchor - tangent * (length * 0.5)
        p1 = anchor - tangent * (length * 0.15) + normal * bend
        p2 = anchor + tangent * (length * 0.15) + normal * bend
        p3 = anchor + tangent * (length * 0.5)
        return StrokeAction(
            type="bezier",
            points=[tuple(map(float, p)) for p in (p0, p1, p2, p3)],
            width=width,
            opacity=0.9,
            pressure=0.8,
            color=0.0,
            pen_down=True,
            metadata={"family": "contour_arc"},
        )

    def _correction_stroke(self, state: EngineState, x: int, y: int, width: float) -> StrokeAction:
        current = np.array(state.pen_position, dtype=np.float32)
        target = np.array([x, y], dtype=np.float32)
        if not np.isfinite(current).all():
            current = target
        delta = target - current
        if float(np.linalg.norm(delta)) < 1.0:
            p0 = target + np.array([-2.0, 0.0], dtype=np.float32)
        else:
            p0 = current
        tangent = self._normalize(target - p0, fallback=np.array([1.0, 0.0], dtype=np.float32))
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        mid = (p0 + target) * 0.5
        p1 = mid + normal * 1.5
        p2 = mid - normal * 1.5
        return StrokeAction(
            type="bezier",
            points=[tuple(map(float, p)) for p in (p0, p1, p2, target)],
            width=width,
            opacity=0.8,
            pressure=0.7,
            color=0.0,
            pen_down=True,
            metadata={"family": "correction"},
        )

    def _normalize(self, vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return fallback.astype(np.float32)
        return (vector / norm).astype(np.float32)
