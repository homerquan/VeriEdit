from __future__ import annotations

import numpy as np

from tooledit._compat import metrics
from tooledit.metrics.iq_metrics import _to_gray


def change_area_ratio(before: np.ndarray, after: np.ndarray, threshold: float = 12.0) -> float:
    delta = np.abs(before.astype(np.float32) - after.astype(np.float32)).mean(axis=2)
    return float(np.mean(delta > threshold))


def mean_absolute_delta(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.mean(np.abs(before.astype(np.float32) - after.astype(np.float32))) / 255.0)


def structural_similarity(before: np.ndarray, after: np.ndarray) -> float:
    gray_before = _to_gray(before)
    gray_after = _to_gray(after)
    if metrics is not None:
        return float(metrics.structural_similarity(gray_before, gray_after, data_range=255.0))
    mse = float(np.mean((gray_before - gray_after) ** 2))
    return float(max(0.0, 1.0 - (mse / (255.0**2))))


def psnr(before: np.ndarray, after: np.ndarray) -> float:
    if metrics is not None:
        return float(metrics.peak_signal_noise_ratio(before, after, data_range=255))
    mse = float(np.mean((before.astype(np.float32) - after.astype(np.float32)) ** 2))
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compare_images(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    return {
        "ssim": structural_similarity(before, after),
        "psnr": psnr(before, after),
        "change_area_ratio": change_area_ratio(before, after),
        "mean_absolute_delta": mean_absolute_delta(before, after),
    }
