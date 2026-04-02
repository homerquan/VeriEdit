from __future__ import annotations

from typing import Any

import numpy as np

from tooledit._compat import exposure


def auto_white_balance(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(params.get("strength", 0.7))
    means = image.astype(np.float32).mean(axis=(0, 1))
    gray_target = float(np.mean(means))
    gains = gray_target / np.maximum(means, 1.0)
    gains = 1.0 + ((gains - 1.0) * strength)
    balanced = image.astype(np.float32) * gains.reshape(1, 1, 3)
    return balanced.clip(0, 255).astype(np.uint8), {"strength": strength, "gains": gains.tolist()}


def bounded_histogram_match_to_reference(
    image: np.ndarray,
    params: dict[str, Any],
    reference: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if reference is None:
        return image.copy(), {"applied": False}
    strength = float(params.get("strength", 0.35))
    if exposure is not None:
        matched = exposure.match_histograms(image, reference, channel_axis=-1)
    else:
        source_mean = image.mean(axis=(0, 1), keepdims=True)
        source_std = image.std(axis=(0, 1), keepdims=True) + 1e-6
        reference_mean = reference.mean(axis=(0, 1), keepdims=True)
        reference_std = reference.std(axis=(0, 1), keepdims=True) + 1e-6
        matched = ((image - source_mean) / source_std) * reference_std + reference_mean
    output = ((1.0 - strength) * image.astype(np.float32)) + (strength * matched.astype(np.float32))
    return output.clip(0, 255).astype(np.uint8), {"applied": True, "strength": strength}
