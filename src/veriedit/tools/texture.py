from __future__ import annotations

from typing import Any

import numpy as np

from veriedit.tools.denoise import wavelet_denoise
from veriedit.tools.sharpen import unsharp_mask


def texture_softness_bias_from_reference(
    image: np.ndarray,
    params: dict[str, Any],
    reference: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if reference is None:
        return image.copy(), {"applied": False}
    target_noise = float(np.std(reference.astype(np.float32) - reference.astype(np.float32).mean()) / 255.0)
    current_noise = float(np.std(image.astype(np.float32) - image.astype(np.float32).mean()) / 255.0)
    if current_noise > target_noise + 0.02:
        softened, details = wavelet_denoise(image, {"strength": 0.06}, None)
        details["applied"] = True
        return softened, details
    sharpened, details = unsharp_mask(image, {"radius": 0.8, "amount": 0.2}, None)
    details["applied"] = True
    return sharpened, details
