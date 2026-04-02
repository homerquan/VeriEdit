from __future__ import annotations

import numpy as np

from .state import PatchBBox


class PatchSelector:
    def __init__(self, patch_size: int = 48) -> None:
        self.patch_size = patch_size

    def select_patch(self, canvas: np.ndarray, target: np.ndarray) -> PatchBBox:
        residual = np.clip(target - (1.0 - canvas), 0.0, 1.0)
        if not np.any(residual > 0.05):
            return PatchBBox(0, 0, min(canvas.shape[1], self.patch_size), min(canvas.shape[0], self.patch_size))
        y, x = np.unravel_index(int(np.argmax(residual)), residual.shape)
        half = self.patch_size // 2
        bbox = PatchBBox(x0=x - half, y0=y - half, x1=x + half, y1=y + half)
        return bbox.clamp(canvas.shape)
