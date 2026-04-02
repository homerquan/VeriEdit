from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from tooledit._compat import cv2
from tooledit.metrics.iq_metrics import estimate_skew_angle


def deskew(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    angle = float(params.get("angle", estimate_skew_angle(image)))
    if abs(angle) < 0.05:
        return image.copy(), {"angle": 0.0, "applied": False}
    if cv2 is not None:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        output = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return output, {"angle": angle, "applied": True}
    rotated = Image.fromarray(image).rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=tuple(image[0, 0]))
    return np.asarray(rotated).astype(np.uint8), {"angle": angle, "applied": True}


def crop(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    top = int(params.get("top", 0))
    left = int(params.get("left", 0))
    height = int(params.get("height", image.shape[0] - top))
    width = int(params.get("width", image.shape[1] - left))
    output = image[top : top + height, left : left + width]
    return output.copy(), {"top": top, "left": left, "height": height, "width": width}


def resize(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    width = int(params.get("width", image.shape[1]))
    height = int(params.get("height", image.shape[0]))
    if cv2 is not None:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC), {"width": width, "height": height}
    output = Image.fromarray(image).resize((width, height), resample=Image.Resampling.BICUBIC)
    return np.asarray(output).astype(np.uint8), {"width": width, "height": height}
