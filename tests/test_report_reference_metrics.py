import csv
from pathlib import Path

from scripts.report_reference_metrics import (
    COMPACT_OUTPUT_METRICS,
    format_summary,
    summarize_metrics,
)


def _write_metrics(path: Path) -> None:
    fieldnames = [
        "epoch",
        "val_source_reference_style_distance",
        "val_fake_reference_style_distance",
        "val_reference_style_ratio",
        "val_reference_style_loss",
        "val_reference_white_balance_loss",
        "val_reference_local_chroma_loss",
        "val_fake_target_red_green_delta",
        "val_fake_target_blue_green_delta",
        "val_fake_target_warm_bias",
        "val_fake_target_warm_abs",
        "val_fake_target_warm_positive_fraction",
        "val_fake_target_tint_bias",
        "val_fake_target_tint_abs",
        "val_source_target_warm_bias",
        "val_source_target_warm_abs",
        "val_source_target_tint_bias",
        "val_source_target_tint_abs",
        "val_fake_target_luminance",
        "val_real_target_luminance",
        "val_range_loss",
        "val_fake_target_local_chroma_tail",
        "val_fake_target_local_red_tail",
        "val_fake_target_local_red_bad_fraction",
        "val_fake_target_red_overshoot_loss",
        "val_fake_target_out_of_range_fraction",
    ]
    rows = [
        {
            "epoch": "135",
            "val_source_reference_style_distance": "0.04",
            "val_fake_reference_style_distance": "0.03",
            "val_reference_style_ratio": "0.75",
        },
        {
            "epoch": "136",
            "val_source_reference_style_distance": "0.03",
            "val_fake_reference_style_distance": "0.027",
            "val_reference_style_ratio": "0.90",
            "val_reference_style_loss": "0.05",
            "val_reference_white_balance_loss": "0.018",
            "val_reference_local_chroma_loss": "0.021",
            "val_fake_target_red_green_delta": "0.012",
            "val_fake_target_blue_green_delta": "-0.008",
            "val_fake_target_warm_bias": "0.010",
            "val_fake_target_warm_abs": "0.014",
            "val_fake_target_warm_positive_fraction": "0.70",
            "val_fake_target_tint_bias": "-0.011",
            "val_fake_target_tint_abs": "0.013",
            "val_source_target_warm_bias": "0.030",
            "val_source_target_warm_abs": "0.040",
            "val_source_target_tint_bias": "-0.020",
            "val_source_target_tint_abs": "0.025",
            "val_fake_target_luminance": "0.47",
            "val_real_target_luminance": "0.49",
            "val_range_loss": "0.001",
            "val_fake_target_local_chroma_tail": "0.031",
            "val_fake_target_local_red_tail": "0.012",
            "val_fake_target_local_red_bad_fraction": "0.004",
            "val_fake_target_red_overshoot_loss": "0.002",
            "val_fake_target_out_of_range_fraction": "0.0",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_metrics_uses_latest_validation_epoch(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)

    epoch, summary = summarize_metrics(metrics_path)

    assert epoch == 136
    assert summary["source_ref"] == 0.03
    assert summary["fake_ref"] == 0.027
    assert summary["ratio"] == 0.90
    assert summary["wb_loss"] == 0.018
    assert summary["local_chroma"] == 0.021
    assert summary["rg_delta"] == 0.012
    assert summary["bg_delta"] == -0.008
    assert summary["warm_bias"] == 0.010
    assert summary["warm_abs"] == 0.014
    assert summary["warm_positive"] == 0.70
    assert summary["tint_bias"] == -0.011
    assert summary["tint_abs"] == 0.013
    assert summary["source_warm"] == 0.030
    assert summary["source_warm_abs"] == 0.040
    assert summary["source_tint"] == -0.020
    assert summary["source_tint_abs"] == 0.025
    assert summary["target_luma"] == 0.49
    assert format_summary(epoch, summary).startswith(
        "epoch=136 source_ref=0.030000 fake_ref=0.027000 ratio=0.900000"
    )
    assert format_summary(
        epoch,
        summary,
        COMPACT_OUTPUT_METRICS,
    ) == (
        "epoch=136 ratio=0.900000 local_tail=0.031000 "
        "red_tail=0.012000 red_bad=0.004000 red_overshoot=0.002000 "
        "out_of_range=0.000000"
    )


def test_summarize_metrics_keeps_new_metrics_optional_for_old_csv(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "old_metrics.csv"
    fieldnames = [
        "epoch",
        "val_reference_style_ratio",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "157",
                "val_reference_style_ratio": "0.824462",
            }
        )

    epoch, summary = summarize_metrics(metrics_path)
    output = format_summary(epoch, summary)

    assert epoch == 157
    assert summary["wb_loss"] is None
    assert summary["local_chroma"] is None
    assert summary["warm_bias"] is None
    assert "wb_loss=NA" in output
    assert "local_chroma=NA" in output
    assert "rg_delta=NA" in output
    assert "bg_delta=NA" in output
    assert "warm_bias=NA" in output
    assert "warm_abs=NA" in output
    assert "warm_positive=NA" in output
    assert "tint_bias=NA" in output
    assert "tint_abs=NA" in output
    assert "source_tint=NA" in output
    assert "source_tint_abs=NA" in output
