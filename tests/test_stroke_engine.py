from pathlib import Path

import numpy as np

from veriedit.engine import ClosedLoopStrokeEngine, EngineConfig


def _circle_target(size: int = 96, radius: float = 24.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cx = cy = (size - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = np.abs(distance - radius) <= 1.5
    image = np.zeros((size, size), dtype=np.float32)
    image[ring] = 1.0
    return image


def _residual(target: np.ndarray, canvas: np.ndarray) -> float:
    return float(np.mean(np.clip(target - (1.0 - canvas), 0.0, 1.0)))


def test_closed_loop_engine_reduces_contour_residual() -> None:
    target = _circle_target()
    initial_canvas = np.ones_like(target, dtype=np.float32)
    engine = ClosedLoopStrokeEngine(
        EngineConfig(
            patch_size=40,
            max_patches=8,
            candidates_per_step=8,
            commit_fraction=0.3,
            max_micro_steps=8,
        )
    )
    result = engine.run(target=target, canvas=initial_canvas)
    assert result.state.vector_history
    assert _residual(target, result.state.canvas) < _residual(target, initial_canvas)


def test_engine_emits_debug_artifacts(tmp_path: Path) -> None:
    target = _circle_target(size=72, radius=18.0)
    engine = ClosedLoopStrokeEngine(
        EngineConfig(
            patch_size=32,
            max_patches=4,
            candidates_per_step=6,
            commit_fraction=0.25,
            max_micro_steps=4,
            debug_dir=tmp_path,
        )
    )
    result = engine.run(target=target)
    assert result.patch_records
    assert any(step.debug_paths for record in result.patch_records for step in record["steps"])
    assert any(path.name == "canvas_patch.png" for path in tmp_path.rglob("*.png"))
