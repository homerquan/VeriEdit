from pathlib import Path

import numpy as np

from veriedit.cli import _build_paint_payload, _save_clone_preview_overlay, _suggest_clone_source_point


def test_suggest_clone_source_point_avoids_bright_damage_region() -> None:
    image = np.full((96, 96, 3), 140, dtype=np.uint8)
    image[28:68, 20:54] = 245
    image[:, 60:] = 120
    strokes = [
        {"points": [[24, 32], [32, 32], [40, 32]]},
        {"points": [[24, 48], [32, 48], [40, 48]]},
    ]
    suggested = _suggest_clone_source_point(image, strokes, size=8)
    assert suggested is not None
    sx, sy = suggested
    assert not (20 <= sx <= 54 and 28 <= sy <= 68)


def test_build_paint_payload_uses_auto_source_and_spacing_for_clone_stamp() -> None:
    image = np.full((96, 96, 3), 140, dtype=np.uint8)
    image[28:68, 20:54] = 245
    strokes = [{"points": [[24, 32], [32, 32], [40, 32]]}]
    payload, _ = _build_paint_payload(
        image=image,
        strokes=strokes,
        mask_boxes=[],
        tool_name="clone_stamp",
        color=None,
        sample_color=None,
        source_point=None,
        pen="soft",
        blend_mode="replace",
        size=10,
        opacity=0.9,
        feather=3.0,
        spacing=6.0,
        rotation=0.0,
        stroke_budget=12,
        candidate_count=18,
        prompt="",
        engine_debug_dir=None,
        flip_horizontal=False,
        flip_vertical=False,
    )
    assert payload["source_point"]
    assert payload["spacing"] == 6.0


def test_save_clone_preview_overlay_creates_preview_image(tmp_path: Path) -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    payload = {
        "source_point": [44, 20],
        "strokes": [{"points": [[16, 24], [28, 24], [40, 24]]}],
        "target_points": [[16, 24], [28, 24], [40, 24]],
        "radius": 8,
        "aligned": True,
    }
    output = tmp_path / "clone.png"
    preview_path = _save_clone_preview_overlay(image, payload, output)
    assert preview_path.exists()
    assert preview_path.name == "clone_preview.png"
