from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from veriedit._compat import cv2


def unsharp_mask(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    radius = float(params.get("radius", 1.0))
    amount = float(params.get("amount", 0.4))
    if cv2 is not None:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(radius, 0.1))
        output = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return output.clip(0, 255).astype(np.uint8), {"radius": radius, "amount": amount}
    pil_image = Image.fromarray(image).filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 150), threshold=2))
    return np.asarray(pil_image).astype(np.uint8), {"radius": radius, "amount": amount}


def edge_preserving_sharpen(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    amount = float(params.get("amount", 0.3))
    if cv2 is not None and hasattr(cv2, "detailEnhance"):
        enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=max(0.05, min(0.3, amount)))
        output = cv2.addWeighted(image, 1.0 - amount, enhanced, amount, 0.0)
        return output.clip(0, 255).astype(np.uint8), {"amount": amount}
    sharpened = ImageEnhance.Sharpness(Image.fromarray(image)).enhance(1.0 + amount)
    return np.asarray(sharpened).astype(np.uint8), {"amount": amount}
