from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from veriedit._compat import cv2, ndimage


def defect_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = _to_gray(image)
    median = _median_gray(gray, 5)
    residual = np.abs(gray - median)
    dust_mask = residual > max(8.0, residual.mean() + residual.std() * 0.9)

    if cv2 is not None:
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        scratch_residual = np.abs(gray - background)
        scratch_mask = scratch_residual > (scratch_residual.mean() + scratch_residual.std() * 1.25)
        scratch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        scratch_mask = cv2.morphologyEx(scratch_mask.astype(np.uint8), cv2.MORPH_OPEN, scratch_kernel).astype(bool)
    else:
        scratch_residual = np.abs(gray - _median_gray(gray, 9))
        scratch_mask = scratch_residual > (scratch_residual.mean() + scratch_residual.std() * 1.1)

    border = max(8, int(min(gray.shape[:2]) * 0.08))
    edge_mask = np.zeros_like(gray, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    edge_damage = edge_mask & ((gray >= 245.0) | (gray <= 15.0))
    defect_union = dust_mask | scratch_mask | edge_damage
    return {
        "dust_mask": dust_mask,
        "scratch_mask": scratch_mask,
        "edge_damage_mask": edge_damage,
        "defect_union": defect_union,
    }


def region_summary(masks: dict[str, np.ndarray]) -> dict[str, object]:
    union = masks["defect_union"]
    boxes = _connected_boxes(union)
    total_pixels = union.shape[0] * union.shape[1]
    largest_area = max((box["area"] for box in boxes), default=0)
    return {
        "defect_region_count": len(boxes),
        "largest_defect_ratio": float(largest_area / max(1, total_pixels)),
        "top_regions": boxes[:8],
        "dust_ratio": float(np.mean(masks["dust_mask"])),
        "scratch_ratio": float(np.mean(masks["scratch_mask"])),
        "edge_damage_ratio_mask": float(np.mean(masks["edge_damage_mask"])),
    }


def save_mask_artifacts(image: np.ndarray, masks: dict[str, np.ndarray], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, mask in masks.items():
        path = output / f"{name}.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
        paths[name] = str(path)
    board_path = output / "diagnostic_regions_board.png"
    _save_overlay_board(image, masks, board_path)
    paths["regions_board"] = str(board_path)
    return paths


def _save_overlay_board(image: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> None:
    base = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "dust_mask": (255, 180, 0, 90),
        "scratch_mask": (255, 64, 64, 100),
        "edge_damage_mask": (64, 160, 255, 90),
    }
    for name, mask in masks.items():
        if name == "defect_union":
            continue
        coords = np.argwhere(mask)
        color = colors.get(name, (120, 255, 120, 80))
        for y, x in coords[:: max(1, len(coords) // 10000 or 1)]:
            draw.point((int(x), int(y)), fill=color)
    composite = Image.alpha_composite(base, overlay).convert("RGB")
    composite.save(path)


def _connected_boxes(mask: np.ndarray) -> list[dict[str, int | float]]:
    if cv2 is not None:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        boxes: list[dict[str, int | float]] = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            boxes.append({"x": int(x), "y": int(y), "width": int(width), "height": int(height), "area": int(area)})
        return sorted(boxes, key=lambda item: int(item["area"]), reverse=True)
    if ndimage is not None:
        labels, count = ndimage.label(mask)
        boxes = []
        for label in range(1, count + 1):
            ys, xs = np.where(labels == label)
            if len(xs) == 0:
                continue
            boxes.append(
                {
                    "x": int(xs.min()),
                    "y": int(ys.min()),
                    "width": int(xs.max() - xs.min() + 1),
                    "height": int(ys.max() - ys.min() + 1),
                    "area": int(len(xs)),
                }
            )
        return sorted(boxes, key=lambda item: int(item["area"]), reverse=True)
    return []


def _to_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def _median_gray(image: np.ndarray, kernel: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.medianBlur(image.astype(np.uint8), kernel).astype(np.float32)
    padded = np.pad(image, ((kernel // 2, kernel // 2), (kernel // 2, kernel // 2)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output[row, col] = float(np.median(padded[row : row + kernel, col : col + kernel]))
    return output
