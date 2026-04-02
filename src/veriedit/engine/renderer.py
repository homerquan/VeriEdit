from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .actions import StrokeAction
from .state import PatchBBox


@dataclass
class LocalRenderer:
    bbox: PatchBBox

    def simulate(self, canvas_patch: np.ndarray, action: StrokeAction, fraction: float = 1.0) -> np.ndarray:
        return self._render_on_patch(canvas_patch, action, fraction=fraction)

    def commit(self, canvas: np.ndarray, action: StrokeAction, fraction: float = 1.0) -> np.ndarray:
        patch = canvas[self.bbox.y0 : self.bbox.y1, self.bbox.x0 : self.bbox.x1].copy()
        rendered = self._render_on_patch(patch, action, fraction=fraction)
        updated = canvas.copy()
        updated[self.bbox.y0 : self.bbox.y1, self.bbox.x0 : self.bbox.x1] = rendered
        return updated

    def _render_on_patch(self, patch: np.ndarray, action: StrokeAction, fraction: float) -> np.ndarray:
        image = Image.fromarray(np.clip(patch * 255.0, 0, 255).astype(np.uint8))
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        path = self._partial_path(action, fraction=fraction)
        if len(path) == 1:
            x, y = path[0]
            radius = max(1, int(round(action.width / 2.0)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(action.opacity * 255))
        elif len(path) >= 2:
            draw.line(path, fill=int(action.opacity * 255), width=max(1, int(round(action.width))))
            radius = max(1, int(round(action.width / 2.0)))
            for x, y in (path[0], path[-1]):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(action.opacity * 255))
        blurred = mask.filter(ImageFilter.GaussianBlur(radius=max(0.6, action.width * 0.2)))
        alpha = np.asarray(blurred, dtype=np.float32) / 255.0
        tone = float(np.clip(action.color, 0.0, 1.0))
        rendered = (patch * (1.0 - alpha)) + (tone * alpha)
        return np.clip(rendered, 0.0, 1.0).astype(np.float32)

    def _partial_path(self, action: StrokeAction, fraction: float) -> list[tuple[int, int]]:
        samples = self._sample_action_points(action)
        if len(samples) <= 1:
            return samples
        fraction = float(np.clip(fraction, 0.05, 1.0))
        seg_lengths = [
            float(np.hypot(samples[idx + 1][0] - samples[idx][0], samples[idx + 1][1] - samples[idx][1]))
            for idx in range(len(samples) - 1)
        ]
        total = max(1e-6, float(sum(seg_lengths)))
        target = total * fraction
        path = [samples[0]]
        covered = 0.0
        for idx, length in enumerate(seg_lengths):
            if covered + length < target:
                path.append(samples[idx + 1])
                covered += length
                continue
            remain = max(0.0, target - covered)
            if length <= 1e-6:
                path.append(samples[idx + 1])
            else:
                t = remain / length
                x0, y0 = samples[idx]
                x1, y1 = samples[idx + 1]
                point = (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
                if point != path[-1]:
                    path.append(point)
            break
        if len(path) == 1 and len(samples) > 1:
            path.append(samples[1])
        return path

    def _sample_action_points(self, action: StrokeAction, samples: int = 24) -> list[tuple[int, int]]:
        if action.type == "bezier" and len(action.points) >= 4:
            p0, p1, p2, p3 = action.points[:4]
            curve: list[tuple[int, int]] = []
            for t in np.linspace(0.0, 1.0, samples):
                x = ((1 - t) ** 3) * p0[0] + 3 * ((1 - t) ** 2) * t * p1[0] + 3 * (1 - t) * (t**2) * p2[0] + (t**3) * p3[0]
                y = ((1 - t) ** 3) * p0[1] + 3 * ((1 - t) ** 2) * t * p1[1] + 3 * (1 - t) * (t**2) * p2[1] + (t**3) * p3[1]
                point = (int(round(x)), int(round(y)))
                if not curve or curve[-1] != point:
                    curve.append(point)
            return curve or [tuple(map(int, map(round, action.points[0])))]
        return [(int(round(x)), int(round(y))) for x, y in action.points]
