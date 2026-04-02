from __future__ import annotations

import math

import numpy as np

from tooledit._compat import cv2, filters


def _to_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def blur_score(image: np.ndarray) -> float:
    gray = _to_gray(image)
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    response = _convolve(gray, kernel)
    return float(np.var(response))


def noise_score(image: np.ndarray) -> float:
    gray = _to_gray(image)
    if cv2 is not None:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        blurred = _box_blur(gray, 5)
    residual = gray - blurred
    return float(np.std(residual) / 255.0)


def yellow_cast_score(image: np.ndarray) -> float:
    mean_rgb = image.mean(axis=(0, 1)) / 255.0
    yellow_bias = ((mean_rgb[0] + mean_rgb[1]) / 2.0) - mean_rgb[2]
    return float(np.clip((yellow_bias + 1.0) / 2.0, 0.0, 1.0))


def contrast_score(image: np.ndarray) -> float:
    gray = _to_gray(image)
    return float(np.clip(np.std(gray) / 64.0, 0.0, 1.5))


def clipping_stats(image: np.ndarray) -> tuple[float, float]:
    gray = _to_gray(image)
    highlights = float(np.mean(gray >= 250.0))
    shadows = float(np.mean(gray <= 5.0))
    return highlights, shadows


def estimate_skew_angle(image: np.ndarray) -> float:
    gray = _to_gray(image)
    threshold = gray < np.percentile(gray, 20)
    points = np.argwhere(threshold)
    if len(points) < 16:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.degrees(math.atan2(principal[0], principal[1]))
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    return float(angle)


def estimate_dust_candidates(image: np.ndarray) -> int:
    gray = _to_gray(image)
    if cv2 is not None:
        median = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    else:
        median = _box_blur(gray, 5)
    residual = np.abs(gray - median)
    mask = residual > max(8.0, residual.mean() + residual.std())
    if not mask.any():
        return 0
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        small = stats[1:, cv2.CC_STAT_AREA]
        return int(np.sum((small >= 1) & (small <= 32)))
    labeled = _label_connected(mask)
    return sum(1 for area in labeled if 1 <= area <= 32)


def style_profile_from_image(image: np.ndarray) -> dict[str, float]:
    mean_rgb = image.mean(axis=(0, 1)) / 255.0
    return {
        "warmth": float(np.clip((((mean_rgb[0] + mean_rgb[1]) / 2.0) - mean_rgb[2] + 1.0) / 2.0, 0.0, 1.0)),
        "contrast": float(np.clip(contrast_score(image) / 1.5, 0.0, 1.0)),
        "sharpness_feel": float(np.clip(blur_score(image) / 250.0, 0.0, 1.0)),
        "grain_level": float(np.clip(noise_score(image) * 4.0, 0.0, 1.0)),
        "saturation": float(np.clip(np.std(image.astype(np.float32) / 255.0, axis=2).mean() * 3.0, 0.0, 1.0)),
    }


def summarize_image_quality(image: np.ndarray, metadata: dict[str, object]) -> dict[str, object]:
    clipping_highlights, clipping_shadows = clipping_stats(image)
    return {
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "mode": str(metadata["mode"]),
        "bit_depth": int(metadata["bit_depth"]),
        "blur_score": blur_score(image),
        "noise_score": noise_score(image),
        "yellow_cast": yellow_cast_score(image),
        "contrast_score": contrast_score(image),
        "clipping_highlights": clipping_highlights,
        "clipping_shadows": clipping_shadows,
        "skew_angle": estimate_skew_angle(image),
        "dust_candidates": estimate_dust_candidates(image),
        "underexposed": float(np.mean(_to_gray(image))) < 90.0,
    }


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row : row + kh, col : col + kw]
            output[row, col] = float(np.sum(window * kernel))
    return output


def _box_blur(image: np.ndarray, size: int) -> np.ndarray:
    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)
    return _convolve(image, kernel)


def _label_connected(mask: np.ndarray) -> list[int]:
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    height, width = mask.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(area)
    return areas
