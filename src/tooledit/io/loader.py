from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(normalized).astype(np.uint8)
        bit_depth = int(array.dtype.itemsize * 8)
        metadata = {
            "path": str(image_path),
            "width": normalized.width,
            "height": normalized.height,
            "mode": normalized.mode,
            "bit_depth": bit_depth,
        }
    return array, metadata
