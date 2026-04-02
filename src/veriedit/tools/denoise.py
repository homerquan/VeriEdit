from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from veriedit._compat import cv2, restoration


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
        try:
            output = restoration.denoise_wavelet(
                image.astype(np.float32) / 255.0,
                channel_axis=-1,
                convert2ycbcr=True,
                rescale_sigma=True,
                sigma=strength,
            )
            return (output * 255.0).clip(0, 255).astype(np.uint8), {"strength": strength, "backend": "wavelet"}
        except ImportError:
            pass
    radius = max(1, int(round(strength * 20)))
    output = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(output).astype(np.uint8), {"strength": strength, "backend": "gaussian_fallback"}


def median_cleanup(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = int(params.get("kernel_size", 3))
    kernel_size = max(3, kernel_size | 1)
    if cv2 is not None:
        return cv2.medianBlur(image, kernel_size), {"kernel_size": kernel_size}
    output = Image.fromarray(image).filter(ImageFilter.MedianFilter(size=kernel_size))
    return np.asarray(output).astype(np.uint8), {"kernel_size": kernel_size}


def bilateral_denoise(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    sigma_color = float(params.get("sigma_color", 28.0))
    sigma_space = float(params.get("sigma_space", 7.0))
    diameter = int(params.get("diameter", 7))
    if cv2 is not None:
        output = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        return output.astype(np.uint8), {"diameter": diameter, "sigma_color": sigma_color, "sigma_space": sigma_space}
    if restoration is not None:
        output = restoration.denoise_bilateral(
            image.astype(np.float32) / 255.0,
            channel_axis=-1,
            sigma_color=max(0.02, sigma_color / 255.0),
            sigma_spatial=max(1.0, sigma_space),
        )
        return (output * 255.0).clip(0, 255).astype(np.uint8), {"diameter": diameter, "sigma_color": sigma_color, "sigma_space": sigma_space}
    output = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=max(1.0, sigma_space / 6.0)))
    return np.asarray(output).astype(np.uint8), {"diameter": diameter, "sigma_color": sigma_color, "sigma_space": sigma_space}
