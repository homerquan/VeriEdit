from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def add_speckles(image: np.ndarray, seed: int, density: float = 0.012) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = image.astype(np.float32).copy()
    mask = rng.random(image.shape[:2]) < density
    colors = rng.choice([0.0, 255.0], size=(image.shape[0], image.shape[1], 1), p=[0.3, 0.7])
    output[mask] = np.repeat(colors[mask], 3, axis=1)
    return output.clip(0, 255).astype(np.uint8)


def add_scratches(image: np.ndarray, seed: int, count: int = 14) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = Image.fromarray(image.astype(np.uint8))
    for _ in range(count):
        x = int(rng.integers(0, image.shape[1]))
        y0 = int(rng.integers(0, image.shape[0] - 40))
        height = int(rng.integers(25, 90))
        width = int(rng.integers(1, 3))
        patch = Image.new("RGB", (width, height), color=(245, 245, 240))
        output.paste(patch, (x, y0))
    return np.asarray(output, dtype=np.uint8)


def sepia_wash(image: np.ndarray, warmth: float = 0.18, fade: float = 0.16) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    sepia = np.stack(
        [
            arr[..., 0] * (1.0 + warmth) + fade,
            arr[..., 1] * (1.0 + warmth * 0.5) + fade,
            arr[..., 2] * (1.0 - warmth * 0.8) + fade * 0.4,
        ],
        axis=2,
    )
    return (np.clip(sepia, 0.0, 1.0) * 255.0).astype(np.uint8)


def low_contrast(image: np.ndarray, factor: float = 0.72) -> np.ndarray:
    return np.asarray(ImageEnhance.Contrast(Image.fromarray(image)).enhance(factor), dtype=np.uint8)


def soft_focus(image: np.ndarray, radius: float = 1.4) -> np.ndarray:
    return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)


def rotate_and_crop(image: np.ndarray, angle: float, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    rotated = Image.fromarray(image).rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    return np.asarray(rotated.crop(crop_box), dtype=np.uint8)


def build_pairs() -> list[dict[str, str]]:
    input_image = load_rgb(DATA_DIR / "sample_input.png")
    reference_image = load_rgb(DATA_DIR / "sample_reference.png")
    crop_box = (60, 40, 520, 430)

    pairs: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        (
            "portrait_crop",
            add_speckles(rotate_and_crop(input_image, -2.2, crop_box), seed=7, density=0.009),
            soft_focus(rotate_and_crop(reference_image, 0.0, crop_box), radius=0.8),
            "Clean scan dust and gently align with the softer reference mood.",
        ),
        (
            "sepia_fade",
            add_speckles(low_contrast(sepia_wash(input_image, warmth=0.22, fade=0.14), factor=0.66), seed=11, density=0.014),
            sepia_wash(reference_image, warmth=0.12, fade=0.05),
            "Restore faded tonal range, reduce dust, and keep the sepia look natural.",
        ),
        (
            "scratch_heavy",
            add_scratches(add_speckles(input_image, seed=19, density=0.008), seed=21, count=16),
            soft_focus(reference_image, radius=0.9),
            "Reduce scratches and dust while preserving faces and composition.",
        ),
        (
            "low_contrast_soft",
            soft_focus(low_contrast(reference_image, factor=0.58), radius=1.8),
            soft_focus(reference_image, radius=0.9),
            "Recover contrast gently and keep the image realistic.",
        ),
        (
            "tight_crop_damage",
            add_scratches(add_speckles(rotate_and_crop(input_image, 1.4, (120, 80, 470, 430)), seed=33, density=0.01), seed=34, count=10),
            rotate_and_crop(reference_image, 0.0, (120, 80, 470, 430)),
            "Clean localized damage and softly match the reference tone.",
        ),
    ]

    manifest: list[dict[str, str]] = []
    for name, source, reference, prompt in pairs:
        source_path = DATA_DIR / f"{name}_source.png"
        reference_path = DATA_DIR / f"{name}_reference.png"
        save_rgb(source, source_path)
        save_rgb(reference, reference_path)
        manifest.append(
            {
                "name": name,
                "source": str(source_path.relative_to(ROOT)),
                "reference": str(reference_path.relative_to(ROOT)),
                "prompt": prompt,
            }
        )
    return manifest


def main() -> None:
    manifest = build_pairs()
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
