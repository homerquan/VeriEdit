import numpy as np

from veriedit.tools import build_tool_registry


def test_registry_contains_expected_tools() -> None:
    registry = build_tool_registry()
    assert "auto_white_balance" in registry.names()
    assert "dust_cleanup" in registry.names()
    assert "paint_strokes" in registry.names()
    assert "spot_healing_brush" in registry.names()
    assert "healing_brush" in registry.names()
    assert "clone_source_paint" in registry.names()
    assert "masked_curves_adjustment" in registry.names()
    assert "stroke_paint" in registry.names()


def test_white_balance_changes_blue_channel_bias() -> None:
    registry = build_tool_registry()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 180
    image[..., 1] = 170
    image[..., 2] = 80
    output, _ = registry.get("auto_white_balance").operation(image, {"strength": 0.8}, None)
    assert output[..., 2].mean() > image[..., 2].mean()


def test_small_defect_heal_reduces_isolated_defect() -> None:
    registry = build_tool_registry()
    image = np.full((48, 48, 3), 140, dtype=np.uint8)
    image[20:23, 20:23] = 255
    output, details = registry.get("small_defect_heal").operation(image, {"max_area": 16, "sensitivity": 0.2, "radius": 2.0}, None)
    assert int(details["mask_pixels"]) > 0
    assert output[21, 21, 0] < 220


def test_paint_strokes_draws_visible_soft_brush_mark() -> None:
    registry = build_tool_registry()
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    output, details = registry.get("paint_strokes").operation(
        image,
        {
            "strokes": [
                {
                    "points": [[5, 5], [20, 20], [32, 20]],
                    "color": [200, 80, 60],
                    "pen": "soft",
                    "size": 6,
                    "opacity": 0.8,
                }
            ]
        },
        None,
    )
    assert details["applied"] is True
    assert details["stroke_count"] == 1
    assert not np.array_equal(output, image)
    assert output[20, 20, 0] > image[20, 20, 0]


def test_paint_strokes_supports_curve_and_dot_primitives() -> None:
    registry = build_tool_registry()
    image = np.full((48, 48, 3), 110, dtype=np.uint8)
    output, details = registry.get("paint_strokes").operation(
        image,
        {
            "strokes": [
                {
                    "points": [[6, 30], [20, 10], [34, 28]],
                    "color": [180, 70, 60],
                    "primitive": "curve",
                    "pen": "soft",
                    "size": 5,
                    "opacity": 0.75,
                },
                {
                    "points": [[38, 38]],
                    "color": [60, 180, 90],
                    "primitive": "dot",
                    "pen": "round",
                    "size": 6,
                    "opacity": 0.9,
                },
            ]
        },
        None,
    )
    assert details["applied"] is True
    assert "curve" in details["primitives"]
    assert "dot" in details["primitives"]
    assert not np.array_equal(output, image)
    assert output[38, 38, 1] > image[38, 38, 1]


def test_spot_healing_brush_repairs_marked_spot() -> None:
    registry = build_tool_registry()
    image = np.full((40, 40, 3), 150, dtype=np.uint8)
    image[20, 20] = [255, 255, 255]
    output, details = registry.get("spot_healing_brush").operation(image, {"points": [[20, 20]], "radius": 4}, None)
    assert details["applied"] is True
    assert details["point_count"] == 1
    assert int(output[20, 20, 0]) < 240


def test_healing_brush_copies_texture_to_target_region() -> None:
    registry = build_tool_registry()
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    image[16:24, 16:24] = [200, 110, 90]
    image[44:52, 44:52] = [40, 40, 40]
    output, details = registry.get("healing_brush").operation(
        image,
        {
            "source_point": [20, 20],
            "target_points": [[48, 48]],
            "radius": 5,
            "mode": "replace",
            "opacity": 1.0,
            "feather": 1.0,
        },
        None,
    )
    assert details["applied"] is True
    assert details["target_count"] == 1
    assert output[48, 48, 0] > image[48, 48, 0]


def test_clone_source_paint_supports_flipped_source() -> None:
    registry = build_tool_registry()
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image[12:20, 12:20] = [180, 80, 80]
    output, details = registry.get("clone_source_paint").operation(
        image,
        {
            "source_point": [16, 16],
            "target_points": [[48, 48]],
            "radius": 4,
            "rotation": 0.0,
            "flip_horizontal": True,
            "opacity": 1.0,
            "feather": 1.0,
        },
        None,
    )
    assert details["applied"] is True
    assert details["flip_horizontal"] is True
    assert output[48, 48, 0] > image[48, 48, 0]


def test_masked_curves_adjustment_changes_only_masked_area() -> None:
    registry = build_tool_registry()
    image = np.full((48, 48, 3), 120, dtype=np.uint8)
    output, details = registry.get("masked_curves_adjustment").operation(
        image,
        {
            "curve_points": [[0, 0], [128, 170], [255, 255]],
            "mask_boxes": [{"x": 0, "y": 0, "width": 18, "height": 48}],
            "feather_sigma": 2.0,
            "motion_blur_length": 9,
            "motion_blur_angle": 0.0,
            "opacity": 1.0,
        },
        None,
    )
    assert details["applied"] is True
    assert details["mask_pixels"] > 0
    assert output[:, 6, 0].mean() > image[:, 6, 0].mean()
    assert abs(float(output[:, 42, 0].mean()) - float(image[:, 42, 0].mean())) < 5.0


def test_stroke_paint_improves_masked_damage_region() -> None:
    registry = build_tool_registry()
    image = np.full((64, 64, 3), 135, dtype=np.uint8)
    image[20:32, 24:40] = [240, 240, 240]
    output, details = registry.get("stroke_paint").operation(
        image,
        {
            "mask_boxes": [{"x": 22, "y": 18, "width": 22, "height": 18}],
            "stroke_budget": 10,
            "candidate_count": 12,
            "min_size": 3,
            "max_size": 10,
            "opacity": 0.7,
            "pen": "soft",
        },
        None,
    )
    assert details["applied"] is True
    assert details["stroke_count"] > 0
    assert details["mse_after"] < details["mse_before"]
    assert any(stroke["primitive"] in {"curve", "line", "dot"} for stroke in details["strokes"])
    repaired_mean = float(output[22:30, 26:38, 0].mean())
    original_mean = float(image[22:30, 26:38, 0].mean())
    assert repaired_mean < original_mean
