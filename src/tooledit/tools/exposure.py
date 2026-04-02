from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from tooledit._compat import cv2


def histogram_balance(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(params.get("strength", 0.6))
    pil_image = Image.fromarray(image)
    balanced = ImageOps.autocontrast(pil_image, cutoff=int((1.0 - strength) * 4))
    output = np.asarray(balanced).astype(np.uint8)
    return output, {"strength": strength}


def clahe_contrast(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    clip_limit = float(params.get("clip_limit", 2.0))
    if cv2 is not None:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        merged = cv2.merge((clahe.apply(l), a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB), {"clip_limit": clip_limit}
    enhancer = ImageEnhance.Contrast(Image.fromarray(image))
    output = enhancer.enhance(1.0 + min(clip_limit, 4.0) * 0.08)
    return np.asarray(output).astype(np.uint8), {"clip_limit": clip_limit}


def gamma_adjust(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    gamma = float(params.get("gamma", 1.0))
    gamma = max(0.3, min(gamma, 2.5))
    corrected = np.power(image.astype(np.float32) / 255.0, 1.0 / gamma)
    return (corrected * 255.0).clip(0, 255).astype(np.uint8), {"gamma": gamma}
