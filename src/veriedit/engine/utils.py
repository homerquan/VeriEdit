from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .state import PatchBBox


def to_grayscale_array(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def crop_patch(image: np.ndarray, bbox: PatchBBox) -> np.ndarray:
    return image[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1].copy()


def patch_center(bbox: PatchBBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1 - 1) / 2.0, (bbox.y0 + bbox.y1 - 1) / 2.0)


def save_grayscale_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(data).save(path)
