from __future__ import annotations

from typing import Any

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    cv2 = None

try:
    from skimage import exposure, filters, measure, metrics, morphology, restoration, transform, util
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    exposure = filters = measure = metrics = morphology = restoration = transform = util = None

try:
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    ndimage = None


def require_cv2() -> Any:
    if cv2 is None:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python-headless is required for this operation.")
    return cv2


def require_skimage() -> dict[str, Any]:
    if any(module is None for module in (exposure, filters, measure, metrics, morphology, restoration, transform, util)):
        raise RuntimeError("scikit-image is required for this operation.")
    return {
        "exposure": exposure,
        "filters": filters,
        "measure": measure,
        "metrics": metrics,
        "morphology": morphology,
        "restoration": restoration,
        "transform": transform,
        "util": util,
    }


def require_scipy_ndimage() -> Any:
    if ndimage is None:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required for this operation.")
    return ndimage
