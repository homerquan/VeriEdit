from __future__ import annotations

from veriedit.tools.base import ToolRegistry, ToolSpec
from veriedit.tools.color import auto_white_balance, bounded_histogram_match_to_reference
from veriedit.tools.denoise import bilateral_denoise, median_cleanup, non_local_means_denoise, wavelet_denoise
from veriedit.tools.exposure import clahe_contrast, gamma_adjust, histogram_balance, shadow_highlight_balance
from veriedit.tools.geometry import crop, deskew, resize
from veriedit.tools.retouch import dust_cleanup, scratch_candidate_cleanup, small_defect_heal
from veriedit.tools.sharpen import edge_preserving_sharpen, unsharp_mask
from veriedit.tools.texture import texture_softness_bias_from_reference


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="auto_white_balance",
            description="Apply gray-world style white balance correction.",
            input_schema={"strength": "float"},
            safety_notes=["Does not alter geometry or semantic content."],
            parameter_bounds={"strength": (0.0, 1.0)},
            expected_effect="Reduces color cast conservatively.",
            likely_failure_modes=["Can neutralize deliberate warm lighting if overused."],
            reversibility_notes="Rerunnable from the original image with lower strength.",
            operation=auto_white_balance,
        )
    )
    registry.register(
        ToolSpec(
            name="histogram_balance",
            description="Apply conservative histogram balancing.",
            input_schema={"strength": "float"},
            safety_notes=["Use conservatively to avoid clipped highlights."],
            parameter_bounds={"strength": (0.0, 1.0)},
            expected_effect="Expands tonal range.",
            likely_failure_modes=["Can flatten mood if pushed too far."],
            reversibility_notes="Re-run from a previous intermediate with lower strength.",
            operation=histogram_balance,
        )
    )
    registry.register(
        ToolSpec(
            name="clahe_contrast",
            description="Local contrast improvement with CLAHE.",
            input_schema={"clip_limit": "float"},
            safety_notes=["Avoid aggressive clip limits to prevent haloing."],
            parameter_bounds={"clip_limit": (1.0, 4.0)},
            expected_effect="Lifts local contrast in faded images.",
            likely_failure_modes=["Can exaggerate dust or noise."],
            reversibility_notes="Use a lower clip limit or skip on retry.",
            operation=clahe_contrast,
        )
    )
    registry.register(
        ToolSpec(
            name="gamma_adjust",
            description="Global gamma correction.",
            input_schema={"gamma": "float"},
            safety_notes=["Prefer mild adjustments only."],
            parameter_bounds={"gamma": (0.4, 2.2)},
            expected_effect="Lightens or darkens midtones.",
            likely_failure_modes=["Can compress highlights or shadows."],
            reversibility_notes="Invert with reciprocal gamma from an earlier intermediate.",
            operation=gamma_adjust,
        )
    )
    registry.register(
        ToolSpec(
            name="shadow_highlight_balance",
            description="Lift shadows and compress highlights using low-frequency illumination estimation.",
            input_schema={"shadow_lift": "float", "highlight_compress": "float", "blur_sigma": "float"},
            safety_notes=["Prefer small values to avoid flat-looking tones."],
            parameter_bounds={"shadow_lift": (0.0, 0.4), "highlight_compress": (0.0, 0.3), "blur_sigma": (4.0, 40.0)},
            expected_effect="Recovers faded or unevenly lit tonal balance.",
            likely_failure_modes=["Can flatten contrast if pushed too hard."],
            reversibility_notes="Retry with smaller lift/compression values.",
            operation=shadow_highlight_balance,
        )
    )
    registry.register(
        ToolSpec(
            name="non_local_means_denoise",
            description="Colored non-local means denoise.",
            input_schema={"h": "float"},
            safety_notes=["Use low strengths to preserve texture."],
            parameter_bounds={"h": (1.0, 12.0)},
            expected_effect="Reduces fine noise while keeping edges.",
            likely_failure_modes=["Can oversmooth faces and paper texture."],
            reversibility_notes="Retry with reduced strength or remove from the plan.",
            operation=non_local_means_denoise,
        )
    )
    registry.register(
        ToolSpec(
            name="bilateral_denoise",
            description="Edge-preserving bilateral denoise.",
            input_schema={"diameter": "int", "sigma_color": "float", "sigma_space": "float"},
            safety_notes=["Prefer this before stronger denoisers for natural-looking cleanup."],
            parameter_bounds={"diameter": (3, 11), "sigma_color": (5.0, 80.0), "sigma_space": (2.0, 20.0)},
            expected_effect="Reduces noise while retaining edge structure.",
            likely_failure_modes=["Large sigmas can create painterly surfaces."],
            reversibility_notes="Lower sigma values or skip on retry.",
            operation=bilateral_denoise,
        )
    )
    registry.register(
        ToolSpec(
            name="wavelet_denoise",
            description="Wavelet-based denoise fallback or refinement.",
            input_schema={"strength": "float"},
            safety_notes=["Keep strength small for restoration workflows."],
            parameter_bounds={"strength": (0.01, 0.2)},
            expected_effect="Softly reduces sensor or scan noise.",
            likely_failure_modes=["Can reduce micro-detail if stacked."],
            reversibility_notes="Retry without the step or with a lower sigma.",
            operation=wavelet_denoise,
        )
    )
    registry.register(
        ToolSpec(
            name="median_cleanup",
            description="Median filtering for salt-and-pepper cleanup.",
            input_schema={"kernel_size": "int"},
            safety_notes=["Keep kernels small to avoid smearing detail."],
            parameter_bounds={"kernel_size": (3, 7)},
            expected_effect="Removes speckle noise and isolated pixels.",
            likely_failure_modes=["Can blur edges and text."],
            reversibility_notes="Run from a prior intermediate with a smaller kernel.",
            operation=median_cleanup,
        )
    )
    registry.register(
        ToolSpec(
            name="dust_cleanup",
            description="Replace isolated dust-like defects using local medians.",
            input_schema={"max_area": "int", "sensitivity": "float"},
            safety_notes=["Only operates on small connected regions."],
            parameter_bounds={"max_area": (3, 64), "sensitivity": (0.1, 0.8)},
            expected_effect="Removes scan dust and speckles.",
            likely_failure_modes=["May miss larger debris or flatten tiny real texture."],
            reversibility_notes="Retry with a lower sensitivity or max area.",
            operation=dust_cleanup,
        )
    )
    registry.register(
        ToolSpec(
            name="scratch_candidate_cleanup",
            description="Reduce thin scratch-like defects using nearby pixels.",
            input_schema={"max_area": "int", "sensitivity": "float"},
            safety_notes=["Designed for narrow defects only; not for missing content."],
            parameter_bounds={"max_area": (10, 200), "sensitivity": (0.1, 0.8)},
            expected_effect="Suppresses narrow scratch artifacts.",
            likely_failure_modes=["Can soften narrow line detail if overused."],
            reversibility_notes="Lower sensitivity or remove on retry if lines degrade.",
            operation=scratch_candidate_cleanup,
        )
    )
    registry.register(
        ToolSpec(
            name="small_defect_heal",
            description="Heal small isolated defects using local inpainting or nearby-pixel replacement.",
            input_schema={"max_area": "int", "sensitivity": "float", "radius": "float"},
            safety_notes=["Restricted to small masks only; not for large missing regions."],
            parameter_bounds={"max_area": (4, 64), "sensitivity": (0.1, 0.8), "radius": (1.0, 4.0)},
            expected_effect="Removes small dust pits and scratch fragments more naturally than flat median replacement.",
            likely_failure_modes=["May soften tiny authentic texture if the mask is too broad."],
            reversibility_notes="Lower sensitivity or reduce max area on retry.",
            operation=small_defect_heal,
        )
    )
    registry.register(
        ToolSpec(
            name="unsharp_mask",
            description="Conservative unsharp mask sharpening.",
            input_schema={"radius": "float", "amount": "float"},
            safety_notes=["Cap strength to avoid halos."],
            parameter_bounds={"radius": (0.3, 2.5), "amount": (0.0, 1.0)},
            expected_effect="Improves perceived detail lightly.",
            likely_failure_modes=["Haloing and noise amplification."],
            reversibility_notes="Retry with a smaller amount or remove entirely.",
            operation=unsharp_mask,
        )
    )
    registry.register(
        ToolSpec(
            name="edge_preserving_sharpen",
            description="Sharpen with more restraint around smooth regions.",
            input_schema={"amount": "float"},
            safety_notes=["Prefer this over strong global sharpening."],
            parameter_bounds={"amount": (0.0, 0.6)},
            expected_effect="Adds crispness with fewer artifacts.",
            likely_failure_modes=["Can still raise edge halos on hard contrast transitions."],
            reversibility_notes="Lower amount or replace with unsharp mask at lower strength.",
            operation=edge_preserving_sharpen,
        )
    )
    registry.register(
        ToolSpec(
            name="deskew",
            description="Rotate image to correct estimated skew.",
            input_schema={"angle": "float"},
            safety_notes=["Preserves content and uses replicated borders."],
            parameter_bounds={"angle": (-15.0, 15.0)},
            expected_effect="Straightens scanned documents or photos.",
            likely_failure_modes=["Wrong angle estimate can over-rotate."],
            reversibility_notes="Apply the inverse rotation from an earlier intermediate.",
            operation=deskew,
        )
    )
    registry.register(
        ToolSpec(
            name="crop",
            description="Crop by explicit pixel bounds.",
            input_schema={"top": "int", "left": "int", "height": "int", "width": "int"},
            safety_notes=["Only use when explicitly requested or on safe auto-crops."],
            parameter_bounds={},
            expected_effect="Removes scan borders or focuses framing.",
            likely_failure_modes=["May cut meaningful content."],
            reversibility_notes="Use the original image to restore lost edges.",
            operation=crop,
        )
    )
    registry.register(
        ToolSpec(
            name="resize",
            description="Resize image to explicit dimensions.",
            input_schema={"width": "int", "height": "int"},
            safety_notes=["Avoid repeated resampling."],
            parameter_bounds={"width": (16, 8192), "height": (16, 8192)},
            expected_effect="Normalizes size for downstream processing.",
            likely_failure_modes=["Can soften details on downsample/upsample."],
            reversibility_notes="Use the source image for pristine dimensions.",
            operation=resize,
        )
    )
    registry.register(
        ToolSpec(
            name="bounded_histogram_match_to_reference",
            description="Blend toward reference tone characteristics without copying content.",
            input_schema={"strength": "float"},
            safety_notes=["Uses only low-level histogram tendencies."],
            parameter_bounds={"strength": (0.0, 0.5)},
            expected_effect="Moves tone and color balance toward the reference feel.",
            likely_failure_modes=["Can drift too far from source mood if overused."],
            reversibility_notes="Retry at lower strength or omit entirely.",
            operation=bounded_histogram_match_to_reference,
        )
    )
    registry.register(
        ToolSpec(
            name="texture_softness_bias_from_reference",
            description="Bias texture softness or crispness toward reference feel.",
            input_schema={},
            safety_notes=["Only affects low-level texture character."],
            parameter_bounds={},
            expected_effect="Aligns softness or crispness with the reference.",
            likely_failure_modes=["May slightly soften or sharpen more than desired."],
            reversibility_notes="Skip on retry if the reviewer flags over-editing.",
            operation=texture_softness_bias_from_reference,
        )
    )
    return registry
