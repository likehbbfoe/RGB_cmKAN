#!/usr/bin/env python3
"""Report privacy-safe aggregate statistics for unpaired translation results."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import numpy as np


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print aggregate brightness, contrast, clipping, and RGB statistics "
            "without exposing images or filenames."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("CMKAN_DATA_ROOT", "/home/share/y50063074/data")
        ),
        help="Dataset root containing val/ or test/ source and target directories",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CMKAN_RESULTS_ROOT",
                str(
                    PROJECT_ROOT.parent
                    / "experiment"
                    / "results"
                    / "custom_unpaired"
                ),
            )
        ),
        help="Prediction root containing source_to_target and target_to_source",
    )
    parser.add_argument(
        "--split",
        choices=("auto", "val", "test"),
        default="auto",
        help="Dataset split to inspect; auto prefers test and falls back to val",
    )
    parser.add_argument(
        "--source-domain",
        default=os.environ.get("CMKAN_SOURCE_DOMAIN", "source"),
    )
    parser.add_argument(
        "--target-domain",
        default=os.environ.get("CMKAN_TARGET_DOMAIN", "target"),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of images sampled from each directory",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=20_000,
        help="Maximum number of pixels sampled from each image",
    )
    return parser.parse_args()


def _resolve_split(args: argparse.Namespace) -> str:
    if args.split != "auto":
        return args.split
    test_source = args.data_root / "test" / args.source_domain
    test_target = args.data_root / "test" / args.target_domain
    return "test" if test_source.is_dir() and test_target.is_dir() else "val"


def _find_images(root: Path, max_images: int) -> list[Path]:
    if not root.is_dir():
        return []
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[:max_images]


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], repeats=3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, repeats=3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected 1, 3, or 4 channels, got {image.shape[-1]}")
    return image


def _normalize(image: np.ndarray) -> np.ndarray:
    original_dtype = image.dtype
    image = image.astype(np.float32)

    if np.issubdtype(original_dtype, np.bool_):
        return image

    if np.issubdtype(original_dtype, np.unsignedinteger):
        return image / float(np.iinfo(original_dtype).max)

    if np.issubdtype(original_dtype, np.signedinteger):
        if image.min() >= 0 and image.max() <= 255:
            return image / 255.0
        if image.min() >= 0 and image.max() <= 65_535:
            return image / 65_535.0
        return image / float(np.iinfo(original_dtype).max)

    if image.min() >= 0 and image.max() > 1:
        if image.max() <= 255:
            return image / 255.0
        if image.max() <= 65_535:
            return image / 65_535.0
    return image


def collect_stats(
    root: Path,
    max_images: int,
    samples_per_image: int,
) -> dict[str, Any] | None:
    paths = _find_images(root, max_images)
    if not paths:
        return None

    samples: list[np.ndarray] = []
    dtype_counts: Counter[str] = Counter()
    failures = 0

    for path in paths:
        try:
            raw = imageio.imread(path)
            dtype_counts[str(raw.dtype)] += 1
            image = _normalize(_ensure_rgb(raw))
            pixels = image.reshape(-1, 3)
            step = max(1, len(pixels) // samples_per_image)
            samples.append(pixels[::step][:samples_per_image])
        except (OSError, ValueError, RuntimeError):
            failures += 1

    if not samples:
        return None

    rgb = np.concatenate(samples, axis=0)
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return {
        "images": len(paths) - failures,
        "failures": failures,
        "sampled_pixels": len(rgb),
        "dtypes": dict(sorted(dtype_counts.items())),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "p01": float(np.percentile(luma, 1)),
        "p50": float(np.percentile(luma, 50)),
        "p99": float(np.percentile(luma, 99)),
        "black_fraction": float((luma <= 1 / 255).mean()),
        "white_fraction": float((luma >= 254 / 255).mean()),
        "rgb_mean": [float(value) for value in rgb.mean(axis=0)],
        "rgb_std": [float(value) for value in rgb.std(axis=0)],
    }


def _format_rgb(values: list[float]) -> str:
    return "/".join(f"{value:.3f}" for value in values)


def _print_stats(name: str, root: Path, stats: dict[str, Any] | None) -> None:
    if stats is None:
        print(f"{name:18s} MISSING  ({root})")
        return
    print(
        f"{name:18s} "
        f"n={stats['images']:4d} "
        f"mean={stats['luma_mean']:.4f} "
        f"std={stats['luma_std']:.4f} "
        f"p01/50/99={stats['p01']:.3f}/{stats['p50']:.3f}/{stats['p99']:.3f} "
        f"black={stats['black_fraction']:.2%} "
        f"white={stats['white_fraction']:.2%} "
        f"RGB={_format_rgb(stats['rgb_mean'])} "
        f"dtype={stats['dtypes']} "
        f"failures={stats['failures']}"
    )


def _print_comparison(
    name: str,
    output: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    if output is None or expected is None:
        return

    mean_delta = output["luma_mean"] - expected["luma_mean"]
    contrast_ratio = output["luma_std"] / max(expected["luma_std"], 1e-8)
    black_delta = output["black_fraction"] - expected["black_fraction"]
    rgb_delta = [
        output_value - expected_value
        for output_value, expected_value in zip(
            output["rgb_mean"], expected["rgb_mean"]
        )
    ]

    print(
        f"{name}: mean_delta={mean_delta:+.4f}, "
        f"contrast_ratio={contrast_ratio:.3f}, "
        f"black_delta={black_delta:+.2%}, "
        f"RGB_delta={_format_rgb(rgb_delta)}"
    )
    if mean_delta < -0.08:
        print("  WARNING: output is substantially darker than the expected domain")
    if contrast_ratio < 0.7:
        print("  WARNING: output contrast is compressed; gray haze/collapse is likely")
    if black_delta > 0.05:
        print("  WARNING: output contains substantially more clipped black pixels")
    if max(abs(value) for value in rgb_delta) > 0.08:
        print("  WARNING: output has a strong channel/color distribution shift")


def main() -> None:
    args = parse_args()
    if args.max_images <= 0 or args.samples_per_image <= 0:
        raise ValueError("--max-images and --samples-per-image must be positive")

    split = _resolve_split(args)
    roots = {
        "source": args.data_root / split / args.source_domain,
        "target": args.data_root / split / args.target_domain,
        "source_to_target": (
            args.results_root / f"{args.source_domain}_to_{args.target_domain}"
        ),
        "target_to_source": (
            args.results_root / f"{args.target_domain}_to_{args.source_domain}"
        ),
    }
    stats = {
        name: collect_stats(root, args.max_images, args.samples_per_image)
        for name, root in roots.items()
    }

    print("Privacy-safe aggregate image statistics")
    print(f"split={split}; no filenames or image content are printed")
    for name, root in roots.items():
        _print_stats(name, root, stats[name])

    print("\nDomain comparisons")
    _print_comparison(
        "source_to_target vs target",
        stats["source_to_target"],
        stats["target"],
    )
    _print_comparison(
        "target_to_source vs source",
        stats["target_to_source"],
        stats["source"],
    )


if __name__ == "__main__":
    main()
