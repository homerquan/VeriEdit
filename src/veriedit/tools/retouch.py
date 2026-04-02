from __future__ import annotations

from typing import Any

import numpy as np

from veriedit._compat import cv2, ndimage


def dust_cleanup(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    sensitivity = float(params.get("sensitivity", 0.45))
    max_area = int(params.get("max_area", 20))
    gray = _gray(image)
    median = _median_color(image, 5)
    residual = np.abs(gray - _gray(median))
    threshold = residual.mean() + residual.std() * (0.5 + sensitivity)
    mask = residual > threshold
    output = _replace_small_regions(image, median, mask, max_area=max_area, min_aspect=0.2)
    return output, {"sensitivity": sensitivity, "max_area": max_area}


def scratch_candidate_cleanup(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    sensitivity = float(params.get("sensitivity", 0.4))
    max_area = int(params.get("max_area", 80))
    gray = _gray(image)
    median = _median_color(image, 7)
    residual = np.abs(gray - _gray(median))
    threshold = residual.mean() + residual.std() * (0.25 + sensitivity)
    mask = residual > threshold
    output = _replace_small_regions(image, median, mask, max_area=max_area, min_aspect=2.0)
    return output, {"sensitivity": sensitivity, "max_area": max_area}


def small_defect_heal(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    max_area = int(params.get("max_area", 36))
    sensitivity = float(params.get("sensitivity", 0.42))
    radius = float(params.get("radius", 2.0))
    gray = _gray(image)
    median = _median_color(image, 5)
    residual = np.abs(gray - _gray(median))
    threshold = residual.mean() + residual.std() * (0.45 + sensitivity)
    mask = residual > threshold
    refined_mask = _small_component_mask(mask, max_area=max_area)
    if cv2 is not None and refined_mask.any():
        healed = cv2.inpaint(image, refined_mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        return healed.astype(np.uint8), {
            "max_area": max_area,
            "sensitivity": sensitivity,
            "radius": radius,
            "mask_pixels": int(refined_mask.sum()),
        }
    output = _replace_small_regions(image, median, refined_mask, max_area=max_area, min_aspect=0.2)
    return output, {"max_area": max_area, "sensitivity": sensitivity, "radius": radius, "mask_pixels": int(refined_mask.sum())}


def _gray(image: np.ndarray) -> np.ndarray:
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def _median_color(image: np.ndarray, kernel: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.medianBlur(image, kernel)
    padded = np.pad(image, ((kernel // 2, kernel // 2), (kernel // 2, kernel // 2), (0, 0)), mode="edge")
    output = np.zeros_like(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row : row + kernel, col : col + kernel]
            output[row, col] = np.median(window.reshape(-1, 3), axis=0)
    return output.astype(np.uint8)


def _replace_small_regions(
    image: np.ndarray,
    replacement: np.ndarray,
    mask: np.ndarray,
    max_area: int,
    min_aspect: float,
) -> np.ndarray:
    output = image.copy()
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            aspect = max(width, height) / max(1, min(width, height))
            if area <= max_area and aspect >= min_aspect:
                region = labels == label
                output[region] = replacement[region]
        return output
    labels = _flood_fill(mask)
    for pixels in labels:
        rows = [pixel[0] for pixel in pixels]
        cols = [pixel[1] for pixel in pixels]
        width = max(cols) - min(cols) + 1
        height = max(rows) - min(rows) + 1
        area = len(pixels)
        aspect = max(width, height) / max(1, min(width, height))
        if area <= max_area and aspect >= min_aspect:
            for row, col in pixels:
                output[row, col] = replacement[row, col]
    return output


def _small_component_mask(mask: np.ndarray, max_area: int) -> np.ndarray:
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        refined = np.zeros_like(mask, dtype=bool)
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= max_area:
                refined |= labels == label
        return refined
    if ndimage is not None:
        labels, count = ndimage.label(mask)
        refined = np.zeros_like(mask, dtype=bool)
        for label in range(1, count + 1):
            area = int(np.sum(labels == label))
            if area <= max_area:
                refined |= labels == label
        return refined
    refined = np.zeros_like(mask, dtype=bool)
    for region in _flood_fill(mask):
        if len(region) <= max_area:
            for row, col in region:
                refined[row, col] = True
    return refined


def _flood_fill(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if visited[row, col] or not mask[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            region: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                region.append((y, x))
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            regions.append(region)
    return regions
