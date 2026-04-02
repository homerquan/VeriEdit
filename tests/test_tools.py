import numpy as np

from veriedit.tools import build_tool_registry


def test_registry_contains_expected_tools() -> None:
    registry = build_tool_registry()
    assert "auto_white_balance" in registry.names()
    assert "dust_cleanup" in registry.names()


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
