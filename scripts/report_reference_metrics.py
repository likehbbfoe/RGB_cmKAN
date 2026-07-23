#!/usr/bin/env python3
"""Print the latest privacy-safe reference-guided validation statistics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_METRICS_PATH = Path(
    os.environ.get(
        "CMKAN_METRICS_PATH",
        "experiments/custom_one_to_one_reference_color_v2/logs/metrics.csv",
    )
)

OUTPUT_METRICS = (
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
        for output_name, csv_name in OUTPUT_METRICS
    }
    return latest_epoch, summary


def _format_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_summary(epoch: int, summary: Mapping[str, float | None]) -> str:
    values = " ".join(
        f"{output_name}={_format_value(summary.get(output_name))}"
        for output_name, _ in OUTPUT_METRICS
    )
    return f"epoch={epoch} {values}"


def main() -> None:
    args = parse_args()
    try:
        epoch, summary = summarize_metrics(args.metrics)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(format_summary(epoch, summary))


if __name__ == "__main__":
    main()
