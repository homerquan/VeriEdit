from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from tooledit._compat import cv2, restoration


def non_local_means_denoise(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(params.get("h", 6.0))
    if cv2 is not None:
        output = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        return output, {"h": strength}
    radius = max(1, int(round(strength / 4.0)))
    output = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(output).astype(np.uint8), {"h": strength}


def wavelet_denoise(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(params.get("strength", 0.08))
    if restoration is not None:
        output = restoration.denoise_wavelet(
            image.astype(np.float32) / 255.0,
            channel_axis=-1,
            convert2ycbcr=True,
            rescale_sigma=True,
            sigma=strength,
        )
        return (output * 255.0).clip(0, 255).astype(np.uint8), {"strength": strength}
    radius = max(1, int(round(strength * 20)))
    output = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(output).astype(np.uint8), {"strength": strength}


def median_cleanup(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = int(params.get("kernel_size", 3))
    kernel_size = max(3, kernel_size | 1)
    if cv2 is not None:
        return cv2.medianBlur(image, kernel_size), {"kernel_size": kernel_size}
    output = Image.fromarray(image).filter(ImageFilter.MedianFilter(size=kernel_size))
    return np.asarray(output).astype(np.uint8), {"kernel_size": kernel_size}
