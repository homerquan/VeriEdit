from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from veriedit.engine import ClosedLoopStrokeEngine, EngineConfig
from veriedit._compat import cv2


def paint_strokes(image: np.ndarray, params: dict[str, Any], _reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    strokes = params.get("strokes", [])
    if not isinstance(strokes, list) or not strokes:
        return image.copy(), {"applied": False, "stroke_count": 0}

    base = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    applied = 0
    pen_types: list[str] = []
    primitives: list[str] = []

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        points = _normalize_points(stroke.get("points", []))
        if len(points) < 1:
            continue
        color = _normalize_color(stroke.get("color", params.get("color", [255, 0, 0])))
        pen = str(stroke.get("pen", params.get("pen", "soft"))).lower()
        primitive = str(stroke.get("primitive", _default_primitive(points))).lower()
        size = max(1, int(stroke.get("size", params.get("size", 6))))
        opacity = float(np.clip(float(stroke.get("opacity", params.get("opacity", 0.75))), 0.0, 1.0))
        _draw_stroke(overlay, points, color, pen, primitive, size, opacity)
        applied += 1
        pen_types.append(pen)
        primitives.append(primitive)

    if not applied:
        return image.copy(), {"applied": False, "stroke_count": 0}

    output = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(output).astype(np.uint8), {
        "applied": True,
        "stroke_count": applied,
        "pen_types": sorted(set(pen_types)),
        "primitives": sorted(set(primitives)),
    }


def stroke_paint(image: np.ndarray, params: dict[str, Any], reference: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
    stroke_budget = max(1, int(params.get("stroke_budget", 12)))
    candidate_count = max(4, int(params.get("candidate_count", 18)))
    min_size = max(1, int(params.get("min_size", 3)))
    max_size = max(min_size, int(params.get("max_size", 14)))
    opacity = float(np.clip(float(params.get("opacity", 0.6)), 0.05, 1.0))
    pen = str(params.get("pen", "soft")).lower()
    prompt = str(params.get("prompt", "")).strip()
    mask = _active_mask(
        shape=image.shape[:2],
        mask_boxes=params.get("mask_boxes"),
        points=params.get("points"),
        radius=int(params.get("radius", max_size)),
    )
    if not mask.any():
        return image.copy(), {"applied": False, "stroke_count": 0, "reason": "No active mask or points supplied."}

    target, target_source = _stroke_target(image, mask, reference)
    bbox = _mask_bbox(mask, padding=max_size)
    source_patch = image[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    target_patch = target[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    local_mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    engine_target = _engine_target_map(source_patch, target_patch, local_mask, prompt)
    if not np.any(engine_target > 0.01):
        return image.copy(), {
            "applied": False,
            "stroke_count": 0,
            "target_source": target_source,
            "reason": "Engine target map had no drawable residual.",
        }

    max_micro_steps = int(np.clip(params.get("max_micro_steps", min(8, stroke_budget)), 1, stroke_budget))
    max_patches = max(1, int(np.ceil(stroke_budget / max_micro_steps)))
    patch_size = int(
        np.clip(
            int(params.get("patch_size", min(max(source_patch.shape[:2]), max(24, max_size * 4)))),
            16,
            max(source_patch.shape[:2]),
        )
    )
    debug_dir_raw = params.get("debug_dir")
    debug_dir = Path(debug_dir_raw) if debug_dir_raw else None
    engine = ClosedLoopStrokeEngine(
        EngineConfig(
            patch_size=patch_size,
            max_patches=max_patches,
            candidates_per_step=candidate_count,
            commit_fraction=float(np.clip(float(params.get("commit_fraction", 0.25)), 0.05, 1.0)),
            max_micro_steps=max_micro_steps,
            debug_dir=debug_dir,
        )
    )
    engine_result = engine.run(target=engine_target)
    stroke_alpha = np.clip(1.0 - engine_result.state.canvas, 0.0, 1.0)
    stroke_alpha = np.where(local_mask, stroke_alpha, 0.0)
    if pen == "soft":
        stroke_alpha = _blur_mask(stroke_alpha, sigma=max(0.8, min_size * 0.35))
    elif pen == "round":
        stroke_alpha = _blur_mask(stroke_alpha, sigma=max(0.4, min_size * 0.18))
    stroke_alpha = np.clip(stroke_alpha * opacity, 0.0, 1.0)

    composed = image.astype(np.float32).copy()
    source_float = source_patch.astype(np.float32)
    target_float = target_patch.astype(np.float32)
    alpha3 = stroke_alpha[..., None]
    composed_patch = (source_float * (1.0 - alpha3)) + (target_float * alpha3)
    composed[bbox[1] : bbox[3], bbox[0] : bbox[2]] = composed_patch
    output = np.clip(composed, 0.0, 255.0).astype(np.uint8)
    mse_before = _masked_mse(image, target, mask)
    mse_after = _masked_mse(output, target, mask)
    stroke_records = _engine_step_records(engine_result.patch_records)

    return output, {
        "applied": bool(stroke_records),
        "backend": "closed_loop_stroke_engine",
        "stroke_count": len(stroke_records),
        "patch_count": sum(1 for record in engine_result.patch_records if record["steps"]),
        "target_source": target_source,
        "engine_target": "difference_contours",
        "prompt": prompt,
        "mse_before": round(float(mse_before), 6),
        "mse_after": round(float(mse_after), 6),
        "improvement": round(float(max(0.0, mse_before - mse_after)), 6),
        "debug_dir": str(debug_dir) if debug_dir else None,
        "strokes": stroke_records,
    }


def _draw_stroke(
    overlay: Image.Image,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    pen: str,
    primitive: str,
    size: int,
    opacity: float,
) -> None:
    alpha = int(round(opacity * 255))
    if pen == "soft":
        _draw_soft_stroke(overlay, points, color, primitive, size, alpha)
        return

    draw = ImageDraw.Draw(overlay, "RGBA")
    rgba = color + (alpha,)
    if primitive == "dot" or len(points) == 1:
        _draw_point(draw, points[0], pen, size, rgba)
        return
    path_points = _stroke_path_points(points, primitive)
    draw.line(path_points, fill=rgba, width=size)
    for point in path_points:
        _draw_point(draw, point, pen, size, rgba)


def _draw_soft_stroke(
    overlay: Image.Image,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    primitive: str,
    size: int,
    alpha: int,
) -> None:
    mask = Image.new("L", overlay.size, 0)
    draw = ImageDraw.Draw(mask)
    if primitive == "dot" or len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - size, y - size, x + size, y + size), fill=alpha)
    else:
        path_points = _stroke_path_points(points, primitive)
        draw.line(path_points, fill=alpha, width=size)
        radius = max(1, size // 2)
        for x, y in path_points:
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


def _default_primitive(points: list[tuple[int, int]]) -> str:
    if len(points) <= 1:
        return "dot"
    if len(points) >= 3:
        return "curve"
    return "line"


def _active_mask(shape: tuple[int, int], mask_boxes: Any, points: Any, radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if isinstance(mask_boxes, list):
        for box in mask_boxes:
            if not isinstance(box, dict):
                continue
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
            mask[y0:y1, x0:x1] = True
    normalized_points = _normalize_points(points)
    if normalized_points:
        for point in normalized_points:
            mask |= _circle_mask(shape, point, radius)
    return mask


def _stroke_target(image: np.ndarray, mask: np.ndarray, reference: np.ndarray | None) -> tuple[np.ndarray, str]:
    if reference is not None and reference.shape[:2] == image.shape[:2]:
        return reference.copy().astype(np.uint8), "reference"
    if cv2 is not None and mask.any():
        inpaint_radius = max(1.0, min(8.0, np.sqrt(float(mask.sum())) / 20.0))
        target = cv2.inpaint(image.astype(np.uint8), mask.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA)
        return target.astype(np.uint8), "inpainted"
    blurred = np.asarray(Image.fromarray(image.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.5))).astype(np.uint8)
    target = image.copy()
    target[mask] = blurred[mask]
    return target.astype(np.uint8), "blur_fill"


def _propose_strokes(
    canvas: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    candidate_count: int,
    min_size: int,
    max_size: int,
    opacity: float,
    pen: str,
) -> list[dict[str, Any]]:
    error_map = np.abs(target.astype(np.float32) - canvas.astype(np.float32)).mean(axis=2)
    error_map = np.where(mask, error_map, 0.0)
    if not np.any(error_map > 0):
        return []
    ys, xs = np.where(mask)
    flat_scores = error_map[ys, xs]
    order = np.argsort(flat_scores)[::-1]
    selected = order[: max(1, candidate_count // 3)]
    gradients = _gradient_fields(target)
    strokes: list[dict[str, Any]] = []
    for index in selected:
        x = int(xs[index])
        y = int(ys[index])
        strength = float(error_map[y, x] / 255.0)
        base_size = int(round(min_size + (max_size - min_size) * np.clip(strength * 2.0, 0.0, 1.0)))
        colors = _sample_candidate_colors(canvas, target, x, y, radius=max(1, base_size // 2))
        primitive_specs = _stroke_geometry_candidates(x, y, gradients, base_size, target.shape[:2])
        for primitive, points, size_scale, opacity_scale in primitive_specs:
            scaled_size = max(min_size, min(max_size, int(round(base_size * size_scale))))
            for color in colors:
                strokes.append(
                    {
                        "points": [[int(px), int(py)] for px, py in points],
                        "color": [int(channel) for channel in color],
                        "pen": pen,
                        "primitive": primitive,
                        "size": scaled_size,
                        "opacity": round(float(np.clip(opacity * opacity_scale, 0.05, 1.0)), 4),
                    }
                )
    return strokes


def _apply_stroke_with_mask(canvas: np.ndarray, stroke: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    base = Image.fromarray(canvas.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    points = _normalize_points(stroke.get("points", []))
    if not points:
        return canvas
    color = _normalize_color(stroke.get("color", [255, 0, 0]))
    pen = str(stroke.get("pen", "soft"))
    primitive = str(stroke.get("primitive", _default_primitive(points)))
    size = int(stroke.get("size", 6))
    opacity = float(stroke.get("opacity", 0.6))
    _draw_stroke(overlay, points, color, pen, primitive, size, opacity)
    if np.any(~mask):
        alpha = np.asarray(overlay.getchannel("A"), dtype=np.float32)
        alpha *= mask.astype(np.float32)
        overlay.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(composited).astype(np.uint8)


def _masked_mse(canvas: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    delta = (canvas.astype(np.float32) - target.astype(np.float32)) ** 2
    if not mask.any():
        return float(np.mean(delta))
    return float(np.mean(delta[mask]))


def _mask_bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(mask.shape[1], int(xs.max()) + padding + 1)
    y1 = min(mask.shape[0], int(ys.max()) + padding + 1)
    return (x0, y0, x1, y1)


def _engine_target_map(source_patch: np.ndarray, target_patch: np.ndarray, local_mask: np.ndarray, prompt: str) -> np.ndarray:
    source_gray = _grayscale_float(source_patch)
    target_gray = _grayscale_float(target_patch)
    diff = np.abs(target_gray - source_gray)
    if cv2 is not None:
        edges = cv2.Canny((target_gray * 255.0).astype(np.uint8), 40, 140).astype(np.float32) / 255.0
    else:
        gy, gx = np.gradient(target_gray)
        edges = np.hypot(gx, gy).astype(np.float32)
        edge_max = float(edges.max())
        if edge_max > 1e-6:
            edges /= edge_max
    prompt_lower = prompt.lower()
    edge_weight = 0.7 if any(token in prompt_lower for token in {"draw", "stroke", "outline", "sketch", "paint"}) else 0.45
    diff_weight = 1.0
    target_map = np.clip((diff * diff_weight) + (edges * edge_weight), 0.0, 1.0)
    target_map = np.where(local_mask, target_map, 0.0)
    return _blur_mask(target_map.astype(np.float32), sigma=0.8)


def _grayscale_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def _blur_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask.astype(np.float32)
    if cv2 is not None:
        ksize = max(3, int(round(sigma * 4)) * 2 + 1)
        return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigmaX=sigma).astype(np.float32)
    blurred = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=sigma))
    return (np.asarray(blurred, dtype=np.float32) / 255.0).astype(np.float32)


def _engine_step_records(patch_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strokes: list[dict[str, Any]] = []
    for patch_index, record in enumerate(patch_records, start=1):
        for step_index, step in enumerate(record.get("steps", []), start=1):
            action = step.chosen_action
            strokes.append(
                {
                    "patch_index": patch_index,
                    "step_index": step_index,
                    "type": action.type,
                    "primitive": "curve" if action.type == "bezier" else action.type,
                    "points": [[round(float(x), 3), round(float(y), 3)] for x, y in action.points],
                    "width": round(float(action.width), 3),
                    "opacity": round(float(action.opacity), 3),
                    "pressure": round(float(action.pressure), 3),
                    "commit_fraction": round(float(step.commit_fraction), 3),
                    "score": round(float(step.score), 6),
                    "residual_before": round(float(step.residual_before), 6),
                    "residual_after": round(float(step.residual_after), 6),
                    "debug_paths": step.debug_paths,
                    "family": action.metadata.get("family"),
                }
            )
    return strokes


def _gradient_fields(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = np.dot(target[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    if cv2 is not None:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return grad_x, grad_y
    grad_y, grad_x = np.gradient(gray)
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def _stroke_points_from_gradient(
    x: int,
    y: int,
    gradients: tuple[np.ndarray, np.ndarray],
    size: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    grad_x, grad_y = gradients
    gx = float(grad_x[y, x])
    gy = float(grad_y[y, x])
    tangent = np.array([-gy, gx], dtype=np.float32)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-5:
        tangent = np.array([1.0, 0.0], dtype=np.float32)
        norm = 1.0
    tangent /= norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    length = max(2.0, float(size) * 1.8)
    bend = float(size) * 0.35
    start = np.array([x, y], dtype=np.float32) - tangent * length * 0.6
    control = np.array([x, y], dtype=np.float32) + normal * bend
    end = np.array([x, y], dtype=np.float32) + tangent * length * 0.6
    return [_clip_point(start, shape), _clip_point(control, shape), _clip_point(end, shape)]


def _stroke_geometry_candidates(
    x: int,
    y: int,
    gradients: tuple[np.ndarray, np.ndarray],
    size: int,
    shape: tuple[int, int],
) -> list[tuple[str, list[tuple[int, int]], float, float]]:
    curve_points = _stroke_points_from_gradient(x, y, gradients, size, shape)
    start, control, end = curve_points
    line_points = [start, end]
    dot_points = [(x, y)]
    return [
        ("curve", curve_points, 1.0, 0.9),
        ("line", line_points, 0.9, 0.8),
        ("dot", dot_points, 0.7, 1.0),
    ]


def _clip_point(point: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    x = int(np.clip(round(float(point[0])), 0, shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, shape[0] - 1))
    return (x, y)


def _sample_patch_color(target: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int, int]:
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(target.shape[1], x + radius + 1)
    y1 = min(target.shape[0], y + radius + 1)
    patch = target[y0:y1, x0:x1]
    color = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in color)


def _sample_candidate_colors(
    canvas: np.ndarray,
    target: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> list[tuple[int, int, int]]:
    target_patch = _patch_pixels(target, x, y, radius)
    canvas_patch = _patch_pixels(canvas, x, y, radius)
    mean_color = tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in target_patch.mean(axis=0))
    median_color = tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in np.median(target_patch, axis=0))
    blended = tuple(
        int(
            np.clip(
                round(float((target_patch.mean(axis=0)[idx] * 0.75) + (canvas_patch.mean(axis=0)[idx] * 0.25))),
                0,
                255,
            )
        )
        for idx in range(3)
    )
    colors = [mean_color, median_color, blended]
    unique: list[tuple[int, int, int]] = []
    for color in colors:
        if color not in unique:
            unique.append(color)
    return unique


def _patch_pixels(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(image.shape[1], x + radius + 1)
    y1 = min(image.shape[0], y + radius + 1)
    patch = image[y0:y1, x0:x1].reshape(-1, 3)
    if len(patch) == 0:
        return image.reshape(-1, 3)
    return patch


def _stroke_path_points(points: list[tuple[int, int]], primitive: str) -> list[tuple[int, int]]:
    if primitive == "curve" and len(points) >= 3:
        return _quadratic_curve_points(points[0], points[1], points[2])
    if primitive == "line" and len(points) >= 2:
        return [points[0], points[-1]]
    return points


def _quadratic_curve_points(
    start: tuple[int, int],
    control: tuple[int, int],
    end: tuple[int, int],
    samples: int = 16,
) -> list[tuple[int, int]]:
    curve: list[tuple[int, int]] = []
    for t in np.linspace(0.0, 1.0, samples):
        x = ((1 - t) ** 2) * start[0] + (2 * (1 - t) * t * control[0]) + (t**2) * end[0]
        y = ((1 - t) ** 2) * start[1] + (2 * (1 - t) * t * control[1]) + (t**2) * end[1]
        point = (int(round(x)), int(round(y)))
        if not curve or curve[-1] != point:
            curve.append(point)
    return curve or [start, end]


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - int(center[0])) ** 2 + (yy - int(center[1])) ** 2 <= max(1, radius) ** 2
