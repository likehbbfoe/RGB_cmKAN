#!/usr/bin/env python3
"""Print the latest privacy-safe reference-guided validation statistics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_METRICS_PATH = (
    PROJECT_ROOT.parent
    / "experiment"
    / "custom_one_to_one_reference_color_v7_face_skin"
    / "logs"
    / "metrics.csv"
)
DEFAULT_METRICS_PATH = Path(
    os.environ.get(
        "CMKAN_METRICS_PATH",
        str(FALLBACK_METRICS_PATH),
    )
)

COMPACT_OUTPUT_METRICS = (
    ("ratio", "val_reference_style_ratio"),
    ("move", "val_source_fake_l1"),
    ("response", "val_reference_response_l1"),
    ("direct", "val_reference_direct_parameter_rms"),
    ("luma_ratio", "val_fake_target_luminance_ratio"),
    ("red_bad", "val_fake_target_local_red_bad_fraction"),
)

RED_OUTPUT_METRICS = (
    ("ratio", "val_reference_style_ratio"),
    ("move", "val_source_fake_l1"),
    ("response", "val_reference_response_l1"),
    ("luma_ratio", "val_fake_target_luminance_ratio"),
    ("rg_delta", "val_fake_target_red_green_delta"),
    ("bg_delta", "val_fake_target_blue_green_delta"),
    ("warm_bias", "val_fake_target_warm_bias"),
    ("warm_abs", "val_fake_target_warm_abs"),
    ("warm_positive", "val_fake_target_warm_positive_fraction"),
    ("tint_bias", "val_fake_target_tint_bias"),
    ("tint_abs", "val_fake_target_tint_abs"),
    ("source_warm", "val_source_target_warm_bias"),
    ("source_tint", "val_source_target_tint_bias"),
    ("red_tail", "val_fake_target_local_red_tail"),
    ("red_bad", "val_fake_target_local_red_bad_fraction"),
    ("red_overshoot", "val_fake_target_red_overshoot_loss"),
)

SKIN_OUTPUT_METRICS = (
    ("ratio", "val_reference_style_ratio"),
    ("move", "val_source_fake_l1"),
    ("response", "val_reference_response_l1"),
    ("skin_ratio", "val_fake_target_skin_tone_ratio"),
    ("skin_loss", "val_fake_target_skin_tone_loss"),
    ("skin_base", "val_source_target_skin_tone_loss"),
    ("skin_rg", "val_fake_target_skin_red_green_delta"),
    ("skin_bg", "val_fake_target_skin_blue_green_delta"),
    ("skin_warm", "val_fake_target_skin_warm_delta"),
    ("skin_tint", "val_fake_target_skin_tint_delta"),
    ("skin_luma", "val_fake_target_skin_luminance_ratio"),
    ("skin_red", "val_fake_target_skin_red_overshoot"),
    ("skin_red_tail", "val_fake_target_skin_local_red_tail"),
    (
        "skin_red_bad",
        "val_fake_target_skin_local_red_bad_fraction",
    ),
    ("skin_valid", "val_fake_target_skin_valid_fraction"),
    ("source_skin", "val_source_skin_fraction"),
    ("target_skin", "val_target_skin_fraction"),
)

FACE_OUTPUT_METRICS = (
    ("skin_valid", "val_fake_target_skin_valid_fraction"),
    ("source_face", "val_source_face_mask_fraction"),
    ("target_face", "val_target_face_mask_fraction"),
    ("source_face_skin", "val_source_skin_face_density"),
    ("target_face_skin", "val_target_skin_face_density"),
    ("face_area_ratio", "val_face_pair_area_ratio"),
    ("face_center_distance", "val_face_pair_center_distance"),
)

LEGACY_OUTPUT_METRICS = (
    ("source_ref", "val_source_reference_style_distance"),
    ("fake_ref", "val_fake_reference_style_distance"),
    ("ratio", "val_reference_style_ratio"),
    ("reference_loss", "val_reference_style_loss"),
    ("wb_loss", "val_reference_white_balance_loss"),
    ("local_chroma", "val_reference_local_chroma_loss"),
    ("rg_delta", "val_fake_target_red_green_delta"),
    ("bg_delta", "val_fake_target_blue_green_delta"),
    ("warm_bias", "val_fake_target_warm_bias"),
    ("warm_abs", "val_fake_target_warm_abs"),
    ("warm_positive", "val_fake_target_warm_positive_fraction"),
    ("tint_bias", "val_fake_target_tint_bias"),
    ("tint_abs", "val_fake_target_tint_abs"),
    ("source_warm", "val_source_target_warm_bias"),
    ("source_warm_abs", "val_source_target_warm_abs"),
    ("source_tint", "val_source_target_tint_bias"),
    ("source_tint_abs", "val_source_target_tint_abs"),
    ("fake_luma", "val_fake_target_luminance"),
    ("target_luma", "val_real_target_luminance"),
    ("range", "val_range_loss"),
)

SAFETY_OUTPUT_METRICS = (
    ("selection", "val_reference_selection_loss"),
    ("luma_ratio", "val_fake_target_luminance_ratio"),
    ("move", "val_source_fake_l1"),
    ("response", "val_reference_response_l1"),
    ("condition", "val_reference_condition_mean_abs"),
    ("direct_weight", "val_reference_direct_weight_rms"),
    ("direct", "val_reference_direct_parameter_rms"),
    ("affine_weight", "val_reference_affine_weight_rms"),
    (
        "condition_saturation",
        "val_reference_condition_saturation_fraction",
    ),
    ("local_mean", "val_fake_target_local_chroma_mean"),
    ("local_tail", "val_fake_target_local_chroma_tail"),
    ("local_bad", "val_fake_target_local_chroma_bad_fraction"),
    ("red_tail", "val_fake_target_local_red_tail"),
    ("red_bad", "val_fake_target_local_red_bad_fraction"),
    ("red_overshoot", "val_fake_target_red_overshoot_loss"),
    ("range_tail", "val_range_tail_loss"),
    ("out_of_range", "val_fake_target_out_of_range_fraction"),
)

SKIN_SAFETY_OUTPUT_METRICS = (
    ("skin_objective", "val_fake_target_skin_loss"),
    ("skin_loss", "val_fake_target_skin_tone_loss"),
    ("skin_base", "val_source_target_skin_tone_loss"),
    ("skin_ratio", "val_fake_target_skin_tone_ratio"),
    ("skin_chroma", "val_fake_target_skin_chroma_loss"),
    ("skin_spread", "val_fake_target_skin_spread_loss"),
    ("skin_luminance", "val_fake_target_skin_luminance_loss"),
    ("skin_uniformity", "val_fake_target_skin_uniformity_loss"),
    ("skin_rg", "val_fake_target_skin_red_green_delta"),
    ("skin_bg", "val_fake_target_skin_blue_green_delta"),
    ("skin_warm", "val_fake_target_skin_warm_delta"),
    ("skin_tint", "val_fake_target_skin_tint_delta"),
    ("skin_luma", "val_fake_target_skin_luminance_ratio"),
    ("skin_red", "val_fake_target_skin_red_overshoot"),
    ("skin_red_tail", "val_fake_target_skin_local_red_tail"),
    (
        "skin_red_bad",
        "val_fake_target_skin_local_red_bad_fraction",
    ),
    ("skin_valid", "val_fake_target_skin_valid_fraction"),
    ("source_skin", "val_source_skin_fraction"),
    ("target_skin", "val_target_skin_fraction"),
    ("source_face", "val_source_face_mask_fraction"),
    ("target_face", "val_target_face_mask_fraction"),
    ("source_face_skin", "val_source_skin_face_density"),
    ("target_face_skin", "val_target_skin_face_density"),
    ("face_area_ratio", "val_face_pair_area_ratio"),
    ("face_center_distance", "val_face_pair_center_distance"),
)

ALL_OUTPUT_METRICS = (
    LEGACY_OUTPUT_METRICS
    + SAFETY_OUTPUT_METRICS
    + SKIN_SAFETY_OUTPUT_METRICS
)

VALID_GATED_SKIN_METRICS = (
    "skin_objective",
    "skin_loss",
    "skin_base",
    "skin_chroma",
    "skin_spread",
    "skin_luminance",
    "skin_uniformity",
    "skin_rg",
    "skin_bg",
    "skin_warm",
    "skin_tint",
    "skin_luma",
    "skin_red",
    "skin_red_tail",
    "skin_red_bad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print one line of aggregate validation metrics without reading "
            "images, filenames, or dataset paths."
        )
    )
    parser.add_argument(
        "metrics",
        nargs="?",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the Lightning CSVLogger metrics.csv file",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Print all legacy and safety metrics instead of the compact "
            "six-field report"
        ),
    )
    output_group.add_argument(
        "--red",
        action="store_true",
        help=(
            "Print only the metrics needed to diagnose global and local "
            "red color casts"
        ),
    )
    output_group.add_argument(
        "--skin",
        action="store_true",
        help=(
            "Print the target-relative skin-tone and local-red diagnostics"
        ),
    )
    output_group.add_argument(
        "--face",
        action="store_true",
        help=(
            "Print only face-ROI validity diagnostics"
        ),
    )
    return parser.parse_args()


def _finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_epoch(row: Mapping[str, object]) -> int | None:
    value = _finite_float(row.get("epoch"))
    return int(value) if value is not None else None


def _mean_for_epoch(
    rows: Iterable[Mapping[str, object]],
    epoch: int,
    metric: str,
) -> float | None:
    values = [
        value
        for row in rows
        if _row_epoch(row) == epoch
        if (value := _finite_float(row.get(metric))) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_metrics(path: Path) -> tuple[int, dict[str, float | None]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    ratio_key = "val_reference_style_ratio"
    validation_epochs = [
        epoch
        for row in rows
        if (epoch := _row_epoch(row)) is not None
        if _finite_float(row.get(ratio_key)) is not None
    ]
    if not validation_epochs:
        raise ValueError(
            f"No '{ratio_key}' values were found in {path}. "
            "Check that reference-guided validation has completed."
        )

    latest_epoch = max(validation_epochs)
    summary = {
        output_name: _mean_for_epoch(rows, latest_epoch, csv_name)
        for output_name, csv_name in ALL_OUTPUT_METRICS
    }
    # The supplied v5 config validates one image per batch. Invalid mask pairs
    # contribute zero to gated Lightning metrics, so undo that dilution before
    # reporting the valid-sample mean. Derive skin_ratio from the two corrected
    # aggregate losses instead of averaging per-image ratios.
    valid_fraction = summary.get("skin_valid")
    if valid_fraction is not None:
        if valid_fraction > 1e-12:
            for metric_name in VALID_GATED_SKIN_METRICS:
                value = summary.get(metric_name)
                if value is not None:
                    summary[metric_name] = value / valid_fraction
        else:
            for metric_name in VALID_GATED_SKIN_METRICS:
                summary[metric_name] = None
    skin_loss = summary.get("skin_loss")
    skin_base = summary.get("skin_base")
    summary["skin_ratio"] = (
        skin_loss / skin_base
        if (
            skin_loss is not None
            and skin_base is not None
            and skin_base > 1e-12
        )
        else None
    )
    return latest_epoch, summary


def _format_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_summary(
    epoch: int,
    summary: Mapping[str, float | None],
    output_metrics: tuple[tuple[str, str], ...] = ALL_OUTPUT_METRICS,
) -> str:
    values = " ".join(
        f"{output_name}={_format_value(summary.get(output_name))}"
        for output_name, _ in output_metrics
    )
    return f"epoch={epoch} {values}"


def main() -> None:
    args = parse_args()
    try:
        epoch, summary = summarize_metrics(args.metrics)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if args.all:
        output_metrics = ALL_OUTPUT_METRICS
    elif args.red:
        output_metrics = RED_OUTPUT_METRICS
    elif args.skin:
        output_metrics = SKIN_OUTPUT_METRICS
    elif args.face:
        output_metrics = FACE_OUTPUT_METRICS
    else:
        output_metrics = COMPACT_OUTPUT_METRICS
    print(format_summary(epoch, summary, output_metrics))


if __name__ == "__main__":
    main()
