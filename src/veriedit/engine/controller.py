from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .actions import StrokeAction
from .critic import LocalCritic
from .proposer import StrokeProposer
from .renderer import LocalRenderer
from .state import EngineState, PatchBBox
from .utils import crop_patch, save_grayscale_image


@dataclass
class ControllerStepRecord:
    patch_bbox: PatchBBox
    commit_fraction: float
    chosen_action: StrokeAction
    score: float
    residual_before: float
    residual_after: float
    candidate_count: int
    debug_paths: dict[str, str] = field(default_factory=dict)


class MPCController:
    def __init__(
        self,
        proposer: StrokeProposer | None = None,
        critic: LocalCritic | None = None,
        commit_fraction: float = 0.25,
        max_micro_steps: int = 12,
        debug_dir: Path | None = None,
    ) -> None:
        self.proposer = proposer or StrokeProposer()
        self.critic = critic or LocalCritic()
        self.commit_fraction = commit_fraction
        self.max_micro_steps = max_micro_steps
        self.debug_dir = debug_dir

    def run_patch(self, state: EngineState, bbox: PatchBBox, k: int = 8) -> tuple[EngineState, list[ControllerStepRecord]]:
        records: list[ControllerStepRecord] = []
        for local_step in range(self.max_micro_steps):
            state.active_patch_bbox = bbox
            state.canvas_patch = crop_patch(state.canvas, bbox)
            state.target_patch = crop_patch(state.target, bbox)
            residual_before = float(np.mean(np.clip(state.target_patch - (1.0 - state.canvas_patch), 0.0, 1.0)))
            if residual_before <= 0.02:
                break
            renderer = LocalRenderer(bbox=bbox)
            candidates = self.proposer.propose(state, k=k)
            if not candidates:
                break
            chosen = None
            chosen_score = None
            chosen_patch = None
            for action in candidates:
                predicted_patch = renderer.simulate(state.canvas_patch, action, fraction=self.commit_fraction)
                score = self.critic.score(state.canvas_patch, predicted_patch, state.target_patch, state, action)
                if chosen is None or score < float(chosen_score):
                    chosen = action
                    chosen_score = score
                    chosen_patch = predicted_patch
            if chosen is None or chosen_patch is None or chosen_score is None:
                break
            new_canvas = renderer.commit(state.canvas, chosen, fraction=self.commit_fraction)
            residual_after = float(
                np.mean(np.clip(state.target_patch - (1.0 - crop_patch(new_canvas, bbox)), 0.0, 1.0))
            )
            if residual_after > residual_before + 1e-4:
                break
            state.canvas = new_canvas
            state.vector_history.append(self._to_global_action(chosen, bbox))
            state.recent_strokes = (state.recent_strokes + [chosen])[-6:]
            partial_path = renderer._partial_path(chosen, fraction=self.commit_fraction)
            if partial_path:
                state.pen_position = tuple(map(float, partial_path[-1]))
            state.pen_down = True
            state.step_index += 1
            record = ControllerStepRecord(
                patch_bbox=bbox,
                commit_fraction=self.commit_fraction,
                chosen_action=chosen,
                score=float(chosen_score),
                residual_before=residual_before,
                residual_after=residual_after,
                candidate_count=len(candidates),
            )
            record.debug_paths = self._save_debug_artifacts(state, bbox, local_step, chosen_patch, chosen, residual_after)
            records.append(record)
        return state, records

    def _to_global_action(self, action: StrokeAction, bbox: PatchBBox) -> StrokeAction:
        return StrokeAction(
            type=action.type,
            points=[(point[0] + bbox.x0, point[1] + bbox.y0) for point in action.points],
            width=action.width,
            opacity=action.opacity,
            pressure=action.pressure,
            color=action.color,
            pen_down=action.pen_down,
            metadata={**action.metadata, "patch_bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1)},
        )

    def _save_debug_artifacts(
        self,
        state: EngineState,
        bbox: PatchBBox,
        local_step: int,
        predicted_patch: np.ndarray,
        action: StrokeAction,
        residual_after: float,
    ) -> dict[str, str]:
        if self.debug_dir is None:
            return {}
        step_dir = self.debug_dir / f"step_{state.step_index:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        canvas_patch = crop_patch(state.canvas, bbox)
        target_patch = crop_patch(state.target, bbox)
        residual = np.clip(target_patch - (1.0 - canvas_patch), 0.0, 1.0)
        residual_path = step_dir / "residual.png"
        save_grayscale_image(residual, residual_path)
        canvas_path = step_dir / "canvas_patch.png"
        save_grayscale_image(canvas_patch, canvas_path)
        target_path = step_dir / "target_patch.png"
        save_grayscale_image(target_patch, target_path)
        predicted_path = step_dir / "predicted_patch.png"
        save_grayscale_image(predicted_patch, predicted_path)
        overlay_path = step_dir / "chosen_action_overlay.png"
        self._save_overlay(canvas_patch, action, overlay_path)
        meta_path = step_dir / "meta.txt"
        meta_path.write_text(
            f"local_step={local_step}\nresidual_after={residual_after:.6f}\naction_type={action.type}\n",
            encoding="utf-8",
        )
        return {
            "canvas_patch": str(canvas_path),
            "target_patch": str(target_path),
            "predicted_patch": str(predicted_path),
            "residual": str(residual_path),
            "chosen_action_overlay": str(overlay_path),
            "meta": str(meta_path),
        }

    def _save_overlay(self, canvas_patch: np.ndarray, action: StrokeAction, output_path: Path) -> None:
        image = Image.fromarray(np.clip(canvas_patch * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
        renderer = LocalRenderer(PatchBBox(0, 0, canvas_patch.shape[1], canvas_patch.shape[0]))
        path = renderer._sample_action_points(action)
        if len(path) >= 2:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(image)
            draw.line(path, fill=(255, 0, 0), width=max(1, int(round(action.width))))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
