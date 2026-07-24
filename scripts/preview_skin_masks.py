#!/usr/bin/env python3
"""Preview the dependency-free skin-color masks used by the v5 objective."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "experiment"
    / "custom_one_to_one_reference_color_v5_skin"
    / "logs"
    / "figures"
    / "skin_mask_preview.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save six local source/target skin-mask previews. No image, "
            "filename, or per-image statistic is uploaded or printed."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CMKAN_DATA_ROOT", "/home/share/y50063074/data")),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--source-domain", default="source")
    parser.add_argument("--target-domain", default="target")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.as_posix())
    )


def _image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"No supported images found in: {root}")
    return sorted(paths, key=_natural_key)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _soft_interval(
    values: np.ndarray,
    lower: float,
    upper: float,
    softness: float,
) -> np.ndarray:
    return (
        _sigmoid((values - lower) / softness)
        * _sigmoid((upper - values) / softness)
    )


def soft_skin_mask(image: Image.Image) -> np.ndarray:
    """Mirror ``UnsupervisedPipeline._soft_skin_mask`` using NumPy."""
    srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(srgb, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    channel_max = srgb.max(axis=-1)
    channel_min = srgb.min(axis=-1)
    saturation = (
        (channel_max - channel_min) / np.maximum(channel_max, 1e-3)
    )
    raw_mask = (
        _soft_interval(luminance, 0.06, 0.95, 0.03)
        * _soft_interval(cb, 0.30, 0.52, 0.02)
        * _soft_interval(cr, 0.52, 0.70, 0.02)
        * _sigmoid((red - green + 0.02) / 0.03)
        * _sigmoid((0.78 - saturation) / 0.06)
    )
    return raw_mask * _sigmoid((raw_mask - 0.35) / 0.05)


def _fit_panel(image: Image.Image, size: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    contained = ImageOps.contain(image.convert("RGB"), (size, size), resampling)
    panel = Image.new("RGB", (size, size), "white")
    left = (size - contained.width) // 2
    top = (size - contained.height) // 2
    panel.paste(contained, (left, top))
    return panel


def _mask_panel(mask: np.ndarray, size: int) -> Image.Image:
    mask_image = Image.fromarray(
        np.clip(mask * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    panel = _fit_panel(mask_image, size)
    coverage = float((mask > 0.25).mean()) * 100
    status = "ok" if 0.5 <= coverage <= 50.0 else "skip"
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 142, 24), fill="black")
    draw.text(
        (6, 6),
        f"coverage {coverage:.1f}% {status}",
        fill="white",
    )
    return panel


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    count = min(length, count)
    if count <= 1:
        return [0]
    return [
        round(index * (length - 1) / (count - 1))
        for index in range(count)
    ]


def main() -> None:
    args = parse_args()
    if args.pairs < 1:
        raise SystemExit("ERROR: --pairs must be at least 1")
    if args.panel_size < 64:
        raise SystemExit("ERROR: --panel-size must be at least 64")

    source_root = args.data_root / args.split / args.source_domain
    target_root = args.data_root / args.split / args.target_domain
    try:
        source_paths = _image_paths(source_root)
        target_paths = _image_paths(target_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if len(source_paths) != len(target_paths):
        raise SystemExit(
            "ERROR: one_to_one mask preview requires equal source/target "
            f"counts, got {len(source_paths)} and {len(target_paths)}"
        )

    selected = _evenly_spaced_indices(len(source_paths), args.pairs)
    header_height = 42
    row_gap = 12
    column_gap = 12
    width = 4 * args.panel_size + 3 * column_gap
    height = (
        header_height
        + len(selected) * args.panel_size
        + max(0, len(selected) - 1) * row_gap
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headings = ("source", "source mask", "target", "target mask")
    for column, heading in enumerate(headings):
        x = column * (args.panel_size + column_gap)
        draw.text((x + 8, 12), heading, fill="black")

    for row, path_index in enumerate(selected):
        source = ImageOps.exif_transpose(Image.open(source_paths[path_index]))
        target = ImageOps.exif_transpose(Image.open(target_paths[path_index]))
        source_mask = soft_skin_mask(source)
        target_mask = soft_skin_mask(target)
        panels = (
            _fit_panel(source, args.panel_size),
            _mask_panel(source_mask, args.panel_size),
            _fit_panel(target, args.panel_size),
            _mask_panel(target_mask, args.panel_size),
        )
        y = header_height + row * (args.panel_size + row_gap)
        for column, panel in enumerate(panels):
            x = column * (args.panel_size + column_gap)
            canvas.paste(panel, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(
        f"Saved {len(selected)} private mask-preview pairs to {args.output}"
    )


if __name__ == "__main__":
    main()
