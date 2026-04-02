from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

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


def spot_healing_brush(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    points = _point_list(params.get("points", []))
    radius = int(params.get("radius", 10))
    if not points:
        return image.copy(), {"applied": False, "point_count": 0}
    mask = np.zeros(image.shape[:2], dtype=bool)
    for point in points:
        mask |= _circle_mask(image.shape[:2], point, radius)
    if cv2 is not None:
        healed = cv2.inpaint(image, mask.astype(np.uint8) * 255, max(1.0, radius / 3.0), cv2.INPAINT_TELEA)
        return healed.astype(np.uint8), {
            "applied": True,
            "point_count": len(points),
            "radius": radius,
            "mask_pixels": int(mask.sum()),
        }
    median = _median_color(image, 5)
    output = image.copy()
    output[mask] = median[mask]
    return output, {"applied": True, "point_count": len(points), "radius": radius, "mask_pixels": int(mask.sum())}


def healing_brush(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    return _source_retouch(image, params, default_mode="normal")


def clone_source_paint(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    return _source_retouch(image, params, default_mode="replace")


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


def _source_retouch(image: np.ndarray, params: dict[str, Any], default_mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    source_point = params.get("source_point")
    target_points = _point_list(params.get("target_points", []))
    radius = int(params.get("radius", 10))
    opacity = float(params.get("opacity", 0.95))
    mode = str(params.get("mode", default_mode)).lower()
    rotation = float(params.get("rotation", 0.0))
    flip_horizontal = bool(params.get("flip_horizontal", False))
    flip_vertical = bool(params.get("flip_vertical", False))
    feather = float(params.get("feather", max(1.0, radius * 0.3)))
    if not source_point or not target_points:
        return image.copy(), {"applied": False, "target_count": 0}
    source_xy = (int(source_point[0]), int(source_point[1]))
    output = image.astype(np.float32).copy()
    source_patch = _extract_patch(image, source_xy, radius).astype(np.float32)
    source_patch = _transform_patch(source_patch, rotation, flip_horizontal, flip_vertical)
    alpha = _patch_alpha(source_patch.shape[:2], radius, feather) * float(np.clip(opacity, 0.0, 1.0))

    for point in target_points:
        target_patch = _extract_patch(output.astype(np.uint8), point, radius).astype(np.float32)
        patch_to_apply = source_patch.copy()
        if mode != "replace":
            patch_to_apply = _match_patch_tone(patch_to_apply, target_patch)
        output = _composite_patch(output, patch_to_apply, alpha, point, radius)

    return np.clip(output, 0, 255).astype(np.uint8), {
        "applied": True,
        "target_count": len(target_points),
        "radius": radius,
        "mode": mode,
        "rotation": rotation,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "opacity": opacity,
    }


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


def _point_list(raw_points: Any) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    if not isinstance(raw_points, list):
        return points
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            points.append((int(point[0]), int(point[1])))
    return points


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    cx, cy = int(center[0]), int(center[1])
    return ((x - cx) ** 2 + (y - cy) ** 2) <= max(1, radius) ** 2


def _extract_patch(image: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    size = (radius * 2) + 1
    pad = radius + 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cx, cy = int(center[0]) + pad, int(center[1]) + pad
    return padded[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1].copy()


def _transform_patch(patch: np.ndarray, rotation: float, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    transformed = patch
    if flip_horizontal:
        transformed = transformed[:, ::-1]
    if flip_vertical:
        transformed = transformed[::-1, :]
    if abs(rotation) < 0.05:
        return transformed
    if cv2 is not None:
        height, width = transformed.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        rotated = cv2.warpAffine(
            transformed,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        return rotated.reshape(transformed.shape)
    rotated = Image.fromarray(transformed.astype(np.uint8)).rotate(rotation, resample=Image.Resampling.BICUBIC)
    return np.asarray(rotated).astype(np.float32)


def _patch_alpha(shape: tuple[int, int], radius: int, feather: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy = shape[0] // 2
    cx = shape[1] // 2
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core_radius = max(1.0, float(radius) - float(feather))
    alpha = np.clip((float(radius) - distance) / max(float(feather), 1.0), 0.0, 1.0)
    alpha[distance <= core_radius] = 1.0
    return alpha[..., None]


def _match_patch_tone(source_patch: np.ndarray, target_patch: np.ndarray) -> np.ndarray:
    src_mean = source_patch.mean(axis=(0, 1), keepdims=True)
    src_std = source_patch.std(axis=(0, 1), keepdims=True) + 1e-6
    tgt_mean = target_patch.mean(axis=(0, 1), keepdims=True)
    tgt_std = target_patch.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (source_patch - src_mean) / src_std
    matched = normalized * tgt_std + tgt_mean
    return np.clip(matched, 0, 255)


def _composite_patch(
    image: np.ndarray,
    patch: np.ndarray,
    alpha: np.ndarray,
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    output = image.copy()
    height, width = image.shape[:2]
    x0 = max(0, int(center[0]) - radius)
    y0 = max(0, int(center[1]) - radius)
    x1 = min(width, int(center[0]) + radius + 1)
    y1 = min(height, int(center[1]) + radius + 1)
    patch_x0 = x0 - (int(center[0]) - radius)
    patch_y0 = y0 - (int(center[1]) - radius)
    patch_x1 = patch_x0 + (x1 - x0)
    patch_y1 = patch_y0 + (y1 - y0)
    window = output[y0:y1, x0:x1]
    patch_window = patch[patch_y0:patch_y1, patch_x0:patch_x1]
    alpha_window = alpha[patch_y0:patch_y1, patch_x0:patch_x1]
    output[y0:y1, x0:x1] = window * (1.0 - alpha_window) + patch_window * alpha_window
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
