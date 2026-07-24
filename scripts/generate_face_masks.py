#!/usr/bin/env python3
"""Generate private, offline face ROI masks with OpenCV's Haar cascade."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re

import imageio.v3 as imageio
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
FaceBox = tuple[int, int, int, int]
FaceDetector = Callable[[np.ndarray], Sequence[Sequence[int]]]


@dataclass(frozen=True)
class FaceMaskRecord:
    """One generated mask and the privacy-safe status needed for summaries."""

    split: str
    domain: str
    image_path: Path
    mask_path: Path
    face_box: FaceBox | None
    reused: bool = False
    detected_override: bool | None = None

    @property
    def detected(self) -> bool:
        if self.detected_override is not None:
            return self.detected_override
        return self.face_box is not None


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.as_posix())
    )


def discover_images(root: Path) -> list[Path]:
    """Find supported images recursively in stable natural order."""
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


def mask_output_path(
    image_path: Path,
    image_root: Path,
    output_root: Path,
) -> Path:
    """Mirror an image's relative path and replace its suffix with PNG."""
    try:
        relative_path = image_path.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(
            f"Image '{image_path}' is not inside root '{image_root}'"
        ) from exc
    return output_root / relative_path.with_suffix(".png")


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an imageio array to contiguous HWC uint8 RGB."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(
            "Expected 1, 3, or 4 channels, "
            f"got shape {array.shape}"
        )

    if array.dtype == np.uint8:
        converted = array
    elif array.dtype == np.bool_:
        converted = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.unsignedinteger):
        maximum = np.iinfo(array.dtype).max
        converted = np.rint(
            array.astype(np.float32) * (255.0 / maximum)
        ).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        finite = np.nan_to_num(
            array.astype(np.float32),
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            finite = finite * 255.0
        converted = np.rint(np.clip(finite, 0.0, 255.0)).astype(np.uint8)
    else:
        converted = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(converted)


def select_largest_face(
    boxes: Iterable[Sequence[int]],
) -> FaceBox | None:
    """Return the valid face box with the largest area."""
    largest_box = None
    largest_area = -1
    for raw_box in boxes:
        if len(raw_box) != 4:
            raise ValueError(f"Face box must contain four values: {raw_box}")
        x, y, width, height = (int(value) for value in raw_box)
        if width <= 0 or height <= 0:
            continue
        area = width * height
        if area > largest_area:
            largest_box = (x, y, width, height)
            largest_area = area
    return largest_box


def ellipse_roi_mask(
    height: int,
    width: int,
    box: FaceBox | None,
) -> np.ndarray:
    """Create a slightly inset, filled ellipse inside one face box."""
    if height <= 0 or width <= 0:
        raise ValueError("Mask height and width must both be positive")
    mask = np.zeros((height, width), dtype=np.uint8)
    if box is None:
        return mask

    x, y, box_width, box_height = box
    left = max(0.0, float(x))
    top = max(0.0, float(y))
    right = min(float(width), float(x + box_width))
    bottom = min(float(height), float(y + box_height))
    clipped_width = right - left
    clipped_height = bottom - top
    if clipped_width <= 0 or clipped_height <= 0:
        return mask

    center_x = 0.5 * (left + right)
    # A slight downward shift excludes more hair while keeping the cheeks.
    center_y = top + 0.52 * clipped_height
    radius_x = max(0.5, 0.42 * clipped_width)
    radius_y = max(0.5, 0.46 * clipped_height)
    y_grid, x_grid = np.ogrid[:height, :width]
    normalized_distance = (
        ((x_grid + 0.5 - center_x) / radius_x) ** 2
        + ((y_grid + 0.5 - center_y) / radius_y) ** 2
    )
    mask[normalized_distance <= 1.0] = 255
    return mask


def soft_skin_mask(image: np.ndarray) -> np.ndarray:
    """Mirror the training-time v5 skin-color heuristic for previews."""
    srgb = to_uint8_rgb(image).astype(np.float32) / 255.0
    red, green, blue = np.moveaxis(srgb, -1, 0)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    channel_max = srgb.max(axis=-1)
    channel_min = srgb.min(axis=-1)
    saturation = (
        (channel_max - channel_min) / np.maximum(channel_max, 1e-3)
    )

    def sigmoid(values):
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def soft_interval(values, lower, upper, softness):
        return (
            sigmoid((values - lower) / softness)
            * sigmoid((upper - values) / softness)
        )

    raw_mask = (
        soft_interval(luminance, 0.06, 0.95, 0.03)
        * soft_interval(cb, 0.30, 0.52, 0.02)
        * soft_interval(cr, 0.52, 0.70, 0.02)
        * sigmoid((red - green + 0.02) / 0.03)
        * sigmoid((0.78 - saturation) / 0.06)
    )
    return raw_mask * sigmoid((raw_mask - 0.35) / 0.05)


def final_skin_mask(image: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Return the binary face-ROI × skin-color mask used for inspection."""
    return np.where(
        (face_mask > 0) & (soft_skin_mask(image) > 0.25),
        255,
        0,
    ).astype(np.uint8)


class OpenCVHaarFaceDetector:
    """Callable wrapper that keeps OpenCV out of the training package."""

    def __init__(
        self,
        cascade_path: Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 7,
        min_face_size: int = 24,
        min_face_ratio: float = 0.08,
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")
        if min_face_size < 1:
            raise ValueError("min_face_size must be at least 1")
        if not 0 < min_face_ratio <= 1:
            raise ValueError("min_face_ratio must be in the interval (0, 1]")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required by this offline script. Install the "
                "project requirements, which include opencv-python."
            ) from exc

        if cascade_path is None:
            cascade_path = (
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
        cascade_path = Path(cascade_path).expanduser()
        classifier = cv2.CascadeClassifier(str(cascade_path))
        if classifier.empty():
            raise FileNotFoundError(
                f"Could not load OpenCV Haar cascade: {cascade_path}"
            )

        self._cv2 = cv2
        self._classifier = classifier
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.min_face_ratio = min_face_ratio

    def __call__(self, image: np.ndarray) -> tuple[FaceBox, ...]:
        rgb = to_uint8_rgb(image)
        gray = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)
        gray = self._cv2.equalizeHist(gray)
        short_edge = min(gray.shape[0], gray.shape[1])
        minimum = min(
            short_edge,
            max(
                self.min_face_size,
                round(short_edge * self.min_face_ratio),
            ),
        )
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(minimum, minimum),
        )
        return tuple(
            tuple(int(value) for value in box)
            for box in boxes
        )


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _generation_plan(
    data_root: Path,
    output_root: Path,
    splits: Sequence[str],
    domains: Sequence[str],
) -> list[tuple[str, str, Path, Path]]:
    if not splits:
        raise ValueError("At least one split is required")
    if not domains:
        raise ValueError("At least one domain is required")

    data_root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root == data_root:
        raise ValueError(
            "Output root must be different from the data root so input PNG "
            "files cannot be overwritten"
        )
    plan = []
    output_paths = set()
    for split in splits:
        for domain in domains:
            domain_root = (data_root / split / domain).resolve()
            real_root = domain_root / "real"
            image_root = (
                real_root
                if split == "train" and real_root.is_dir()
                else domain_root
            )
            if output_root == image_root or _is_inside(output_root, image_root):
                raise ValueError(
                    "Output root cannot be inside an input image directory: "
                    f"{output_root}"
                )
            domain_output_root = output_root / split / domain
            for image_path in discover_images(image_root):
                destination = mask_output_path(
                    image_path,
                    image_root,
                    domain_output_root,
                )
                if destination in output_paths:
                    raise ValueError(
                        "Two input files map to the same PNG mask path: "
                        f"{destination}"
                    )
                output_paths.add(destination)
                plan.append((split, domain, image_path, destination))
    return plan


def _write_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.png")
    Image.fromarray(mask, mode="L").save(temporary_path, format="PNG")
    temporary_path.replace(path)


def generate_face_masks(
    data_root: Path,
    output_root: Path,
    detector: FaceDetector,
    splits: Sequence[str] = ("train", "val"),
    domains: Sequence[str] = ("source", "target"),
    overwrite: bool = False,
) -> list[FaceMaskRecord]:
    """Generate one full-resolution PNG mask for every discovered image."""
    plan = _generation_plan(data_root, output_root, splits, domains)
    records = []
    for split, domain, image_path, destination in plan:
        image = to_uint8_rgb(imageio.imread(image_path))
        if destination.is_file() and not overwrite:
            existing_mask = np.asarray(
                Image.open(destination).convert("L")
            )
            if existing_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Existing mask '{destination}' has shape "
                    f"{existing_mask.shape}, expected {image.shape[:2]}. "
                    "Delete it or rerun with --overwrite."
                )
            records.append(
                FaceMaskRecord(
                    split=split,
                    domain=domain,
                    image_path=image_path,
                    mask_path=destination,
                    face_box=None,
                    reused=True,
                    detected_override=bool(existing_mask.any()),
                )
            )
            continue
        face_box = select_largest_face(detector(image))
        mask = ellipse_roi_mask(
            height=image.shape[0],
            width=image.shape[1],
            box=face_box,
        )
        _write_mask(mask, destination)
        records.append(
            FaceMaskRecord(
                split=split,
                domain=domain,
                image_path=image_path,
                mask_path=destination,
                face_box=face_box,
            )
        )
    return records


def summarize_records(records: Sequence[FaceMaskRecord]) -> str:
    """Return aggregate detection counts without exposing image names."""
    total = len(records)
    detected = sum(record.detected for record in records)
    reused = sum(record.reused for record in records)
    lines = [
        "Face-mask generation summary",
        f"total={total} detected={detected} missed={total - detected} "
        f"reused={reused}",
    ]
    group_names = sorted(
        {(record.split, record.domain) for record in records}
    )
    for split, domain in group_names:
        group = [
            record
            for record in records
            if record.split == split and record.domain == domain
        ]
        group_detected = sum(record.detected for record in group)
        lines.append(
            f"{split}/{domain}: total={len(group)} "
            f"detected={group_detected} "
            f"missed={len(group) - group_detected}"
        )
    return "\n".join(lines)


def _evenly_spaced(items: Sequence[FaceMaskRecord], count: int):
    if count <= 0 or not items:
        return []
    count = min(count, len(items))
    if count == 1:
        return [items[0]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))]
        for index in range(count)
    ]


def _preview_records(
    records: Sequence[FaceMaskRecord],
    sample_count: int,
) -> list[FaceMaskRecord]:
    """Include misses in the preview while sampling both result classes."""
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    misses = [record for record in records if not record.detected]
    detections = [record for record in records if record.detected]
    if not misses or not detections:
        return _evenly_spaced(records, sample_count)

    miss_count = min(len(misses), max(1, sample_count // 3))
    detected_count = min(len(detections), sample_count - miss_count)
    risk_count = (
        min(len(detections), max(1, detected_count // 2))
        if detected_count > 0
        else 0
    )

    def risk_score(record):
        image = to_uint8_rgb(imageio.imread(record.image_path))
        face_mask = np.asarray(
            Image.open(record.mask_path).convert("L")
        )
        roi_fraction = float((face_mask > 0).mean())
        roi_count = max(int((face_mask > 0).sum()), 1)
        skin_density = float(
            (final_skin_mask(image, face_mask) > 0).sum() / roi_count
        )
        # Extremal density and atypical ROI area are the most useful cases
        # to inspect for beige-background false positives or wrong boxes.
        density_risk = abs(skin_density - 0.5)
        area_risk = abs(
            np.log(max(roi_fraction, 1e-4) / 0.12)
        )
        return density_risk + 0.25 * area_risk

    detections_by_risk = sorted(
        detections,
        key=risk_score,
        reverse=True,
    )
    risky = detections_by_risk[:risk_count]
    remaining_detections = detections_by_risk[risk_count:]
    ordinary = _evenly_spaced(
        remaining_detections,
        detected_count - len(risky),
    )
    selected = (
        _evenly_spaced(misses, miss_count)
        + risky
        + ordinary
    )
    remaining = sample_count - len(selected)
    if remaining > 0:
        selected_ids = {id(record) for record in selected}
        unused = [
            record for record in records if id(record) not in selected_ids
        ]
        selected.extend(_evenly_spaced(unused, remaining))
    return selected


def _fit_panel(
    image: Image.Image,
    size: int,
    resampling: int,
) -> Image.Image:
    contained = ImageOps.contain(
        image.convert("RGB"),
        (size, size),
        resampling,
    )
    panel = Image.new("RGB", (size, size), "white")
    left = (size - contained.width) // 2
    top = (size - contained.height) // 2
    panel.paste(contained, (left, top))
    return panel


def _roi_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    outside = np.rint(image.astype(np.float32) * 0.2).astype(np.uint8)
    selected = mask > 0
    overlay = np.where(selected[..., None], image, outside)
    return Image.fromarray(overlay, mode="RGB")


def write_preview(
    records: Sequence[FaceMaskRecord],
    output_path: Path,
    sample_count: int = 12,
    panel_size: int = 256,
) -> Path:
    """Save original, face ROI, final skin mask, and overlay rows."""
    if panel_size < 64:
        raise ValueError("panel_size must be at least 64")
    selected = _preview_records(records, sample_count)
    if not selected:
        raise ValueError("Cannot create a preview without generated masks")

    header_height = 42
    gap = 12
    row_gap = 10
    width = 4 * panel_size + 3 * gap
    height = (
        header_height
        + len(selected) * panel_size
        + (len(selected) - 1) * row_gap
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headings = (
        "original",
        "face ROI",
        "ROI x skin mask",
        "final overlay",
    )
    for column, heading in enumerate(headings):
        x = column * (panel_size + gap)
        draw.text((x + 8, 12), heading, fill="black")

    resampling = getattr(Image, "Resampling", Image)
    manifest_lines = [
        "row\tsplit\tdomain\tstatus\timage_path\tmask_path"
    ]
    for row, record in enumerate(selected):
        image = to_uint8_rgb(imageio.imread(record.image_path))
        mask = np.asarray(Image.open(record.mask_path).convert("L"))
        skin_mask = final_skin_mask(image, mask)
        panels = (
            _fit_panel(
                Image.fromarray(image, mode="RGB"),
                panel_size,
                resampling.LANCZOS,
            ),
            _fit_panel(
                Image.fromarray(mask, mode="L"),
                panel_size,
                resampling.NEAREST,
            ),
            _fit_panel(
                Image.fromarray(skin_mask, mode="L"),
                panel_size,
                resampling.NEAREST,
            ),
            _fit_panel(
                _roi_overlay(image, skin_mask),
                panel_size,
                resampling.LANCZOS,
            ),
        )
        y = header_height + row * (panel_size + row_gap)
        for column, panel in enumerate(panels):
            x = column * (panel_size + gap)
            canvas.paste(panel, (x, y))
        status = "DETECTED" if record.detected else "MISSED / BLACK MASK"
        row_label = f"#{row + 1:02d} {record.split}/{record.domain}"
        draw.rectangle(
            (3 * (panel_size + gap), y, width, y + 22),
            fill="black",
        )
        draw.text(
            (3 * (panel_size + gap) + 6, y + 5),
            f"{row_label} {status}",
            fill="white",
        )
        manifest_lines.append(
            "\t".join(
                (
                    str(row + 1),
                    record.split,
                    record.domain,
                    status,
                    str(record.image_path),
                    str(record.mask_path),
                )
            )
        )

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    manifest_path = output_path.with_suffix(".tsv")
    manifest_path.write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    default_output_root = os.environ.get(
        "CMKAN_FACE_MASK_ROOT",
        "/home/share/y50063074/data_face_masks",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Generate local face ROI masks for train/val source/target images. "
            "Images stay on this machine and are never uploaded."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CMKAN_DATA_ROOT",
                "/home/share/y50063074/data",
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(default_output_root),
        help=(
            "Mask root; may also be set with CMKAN_FACE_MASK_ROOT. The script "
            "creates <root>/<split>/<domain>/<relative-path>.png."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val"),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=("source", "target"),
    )
    parser.add_argument("--cascade-path", type=Path)
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=7)
    parser.add_argument("--min-face-size", type=int, default=24)
    parser.add_argument("--min-face-ratio", type=float, default=0.08)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-samples", type=int, default=30)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate existing sidecars. By default existing masks are "
            "preserved so manual corrections are not lost."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_samples < 0:
        raise SystemExit("ERROR: --preview-samples cannot be negative")
    if args.panel_size < 64:
        raise SystemExit("ERROR: --panel-size must be at least 64")

    try:
        detector = OpenCVHaarFaceDetector(
            cascade_path=args.cascade_path,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_face_size=args.min_face_size,
            min_face_ratio=args.min_face_ratio,
        )
        records = generate_face_masks(
            data_root=args.data_root,
            output_root=args.output_root,
            detector=detector,
            splits=args.splits,
            domains=args.domains,
            overwrite=args.overwrite,
        )
        if args.preview_samples > 0:
            preview_output = (
                args.preview_output
                if args.preview_output is not None
                else args.output_root / "face_mask_preview.png"
            )
            preview_manifest = write_preview(
                records,
                preview_output,
                sample_count=args.preview_samples,
                panel_size=args.panel_size,
            )
        else:
            preview_output = None
            preview_manifest = None
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(summarize_records(records))
    print(f"Masks saved under: {args.output_root}")
    if preview_output is not None:
        print(f"Preview saved to: {preview_output}")
        print(f"Preview row manifest saved to: {preview_manifest}")
        print(
            "Preview columns: original | face ROI | ROI x skin mask | "
            "final overlay. White pixels in the third column enter the "
            "skin statistics; black pixels are excluded."
        )


if __name__ == "__main__":
    main()
