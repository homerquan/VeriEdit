from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def paint_strokes(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strokes = params.get("strokes", [])
    if not isinstance(strokes, list) or not strokes:
        return image.copy(), {"applied": False, "stroke_count": 0}

    base = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    applied = 0
    pen_types: list[str] = []

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        points = _normalize_points(stroke.get("points", []))
        if len(points) < 1:
            continue
        color = _normalize_color(stroke.get("color", params.get("color", [255, 0, 0])))
        pen = str(stroke.get("pen", params.get("pen", "soft"))).lower()
        size = max(1, int(stroke.get("size", params.get("size", 6))))
        opacity = float(np.clip(float(stroke.get("opacity", params.get("opacity", 0.75))), 0.0, 1.0))
        _draw_stroke(overlay, points, color, pen, size, opacity)
        applied += 1
        pen_types.append(pen)

    if not applied:
        return image.copy(), {"applied": False, "stroke_count": 0}

    output = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(output).astype(np.uint8), {
        "applied": True,
        "stroke_count": applied,
        "pen_types": sorted(set(pen_types)),
    }


def _draw_stroke(overlay: Image.Image, points: list[tuple[int, int]], color: tuple[int, int, int], pen: str, size: int, opacity: float) -> None:
    alpha = int(round(opacity * 255))
    if pen == "soft":
        _draw_soft_stroke(overlay, points, color, size, alpha)
        return

    draw = ImageDraw.Draw(overlay, "RGBA")
    rgba = color + (alpha,)
    if len(points) == 1:
        _draw_point(draw, points[0], pen, size, rgba)
        return
    draw.line(points, fill=rgba, width=size)
    for point in points:
        _draw_point(draw, point, pen, size, rgba)


def _draw_soft_stroke(overlay: Image.Image, points: list[tuple[int, int]], color: tuple[int, int, int], size: int, alpha: int) -> None:
    mask = Image.new("L", overlay.size, 0)
    draw = ImageDraw.Draw(mask)
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - size, y - size, x + size, y + size), fill=alpha)
    else:
        draw.line(points, fill=alpha, width=size)
        radius = max(1, size // 2)
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=alpha)
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, size * 0.45)))
    color_layer = Image.new("RGBA", overlay.size, color + (0,))
    color_layer.putalpha(blurred)
    merged = Image.alpha_composite(overlay, color_layer)
    overlay.paste(merged)


def _draw_point(draw: ImageDraw.ImageDraw, point: tuple[int, int], pen: str, size: int, rgba: tuple[int, int, int, int]) -> None:
    x, y = point
    radius = max(1, size // 2)
    if pen == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=rgba)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba)


def _normalize_points(raw_points: Any) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    if not isinstance(raw_points, list):
        return points
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x, y = point
        points.append((int(x), int(y)))
    return points


def _normalize_color(raw_color: Any) -> tuple[int, int, int]:
    if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
        return tuple(int(np.clip(int(channel), 0, 255)) for channel in raw_color[:3])
    return (255, 0, 0)
