import numpy as np

from tooledit.tools import build_tool_registry


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
