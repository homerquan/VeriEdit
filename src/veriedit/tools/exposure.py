from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from veriedit._compat import cv2, ndimage


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


def shadow_highlight_balance(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    shadow_lift = float(params.get("shadow_lift", 0.18))
    highlight_compress = float(params.get("highlight_compress", 0.12))
    blur_sigma = float(params.get("blur_sigma", 18.0))
    working = image.astype(np.float32) / 255.0
    luminance = working.mean(axis=2)
    if cv2 is not None:
        illumination = cv2.GaussianBlur(luminance, (0, 0), sigmaX=blur_sigma)
    elif ndimage is not None:
        illumination = ndimage.gaussian_filter(luminance, sigma=blur_sigma)
    else:
        illumination = luminance
    shadow_mask = np.clip(1.0 - illumination, 0.0, 1.0)[..., None]
    highlight_mask = np.clip(illumination - 0.5, 0.0, 1.0)[..., None] * 2.0
    balanced = working + shadow_mask * shadow_lift - highlight_mask * highlight_compress
    return (np.clip(balanced, 0.0, 1.0) * 255.0).astype(np.uint8), {
        "shadow_lift": shadow_lift,
        "highlight_compress": highlight_compress,
        "blur_sigma": blur_sigma,
    }


def masked_curves_adjustment(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    curve_points = params.get("curve_points") or [[0, 0], [128, 128], [255, 255]]
    mask_boxes = params.get("mask_boxes") or []
    feather_sigma = float(params.get("feather_sigma", 4.0))
    motion_blur_length = int(params.get("motion_blur_length", 0))
    motion_blur_angle = float(params.get("motion_blur_angle", 0.0))
    opacity = float(params.get("opacity", 1.0))
    if not mask_boxes:
        return image.copy(), {"applied": False, "mask_pixels": 0}
    mask = _boxes_mask(image.shape[:2], mask_boxes)
    mask = _soften_mask(mask, feather_sigma, motion_blur_length, motion_blur_angle)
    lut = _curve_lut(curve_points)
    adjusted = lut[image]
    alpha = np.clip(mask[..., None] * opacity, 0.0, 1.0)
    output = image.astype(np.float32) * (1.0 - alpha) + adjusted.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8), {
        "applied": True,
        "mask_pixels": int(np.sum(mask > 0.01)),
        "feather_sigma": feather_sigma,
        "motion_blur_length": motion_blur_length,
        "motion_blur_angle": motion_blur_angle,
        "opacity": opacity,
    }


def _boxes_mask(shape: tuple[int, int], mask_boxes: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for box in mask_boxes:
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        width = int(box.get("width", 0))
        height = int(box.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(shape[1], x + width)
        y1 = min(shape[0], y + height)
        mask[y0:y1, x0:x1] = 1.0
    return mask


def _soften_mask(mask: np.ndarray, feather_sigma: float, motion_blur_length: int, motion_blur_angle: float) -> np.ndarray:
    softened = mask.astype(np.float32)
    if feather_sigma > 0:
        if cv2 is not None:
            softened = cv2.GaussianBlur(softened, (0, 0), sigmaX=feather_sigma)
        elif ndimage is not None:
            softened = ndimage.gaussian_filter(softened, sigma=feather_sigma)
    if motion_blur_length > 1:
        kernel = _motion_blur_kernel(motion_blur_length, motion_blur_angle)
        if cv2 is not None:
            softened = cv2.filter2D(softened, -1, kernel)
        elif ndimage is not None:
            softened = ndimage.convolve(softened, kernel, mode="nearest")
    return np.clip(softened, 0.0, 1.0)


def _motion_blur_kernel(length: int, angle: float) -> np.ndarray:
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, :] = 1.0
    if cv2 is not None and abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel_sum = float(kernel.sum()) or 1.0
    return kernel / kernel_sum


def _curve_lut(points: list[list[int]]) -> np.ndarray:
    normalized: list[tuple[int, int]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = point
        normalized.append((max(0, min(255, int(x))), max(0, min(255, int(y)))))
    normalized = sorted(normalized)
    if not normalized:
        normalized = [(0, 0), (255, 255)]
    xp = np.array([point[0] for point in normalized], dtype=np.float32)
    fp = np.array([point[1] for point in normalized], dtype=np.float32)
    if xp[0] != 0:
        xp = np.insert(xp, 0, 0.0)
        fp = np.insert(fp, 0, 0.0)
    if xp[-1] != 255:
        xp = np.append(xp, 255.0)
        fp = np.append(fp, 255.0)
    lut = np.interp(np.arange(256, dtype=np.float32), xp, fp)
    return np.clip(lut, 0, 255).astype(np.uint8)
