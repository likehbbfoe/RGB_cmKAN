import csv
import math
from pathlib import Path

from scripts.report_reference_metrics import (
    COMPACT_OUTPUT_METRICS,
    RED_OUTPUT_METRICS,
    SKIN_OUTPUT_METRICS,
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
        "val_reference_selection_loss",
        "val_fake_target_luminance_ratio",
        "val_source_fake_l1",
        "val_reference_response_l1",
        "val_reference_condition_mean_abs",
        "val_reference_direct_weight_rms",
        "val_reference_direct_parameter_rms",
        "val_reference_affine_weight_rms",
        "val_reference_condition_saturation_fraction",
        "val_fake_target_skin_loss",
        "val_fake_target_skin_tone_loss",
        "val_source_target_skin_tone_loss",
        "val_fake_target_skin_tone_ratio",
        "val_fake_target_skin_chroma_loss",
        "val_fake_target_skin_spread_loss",
        "val_fake_target_skin_luminance_loss",
        "val_fake_target_skin_uniformity_loss",
        "val_fake_target_skin_red_green_delta",
        "val_fake_target_skin_blue_green_delta",
        "val_fake_target_skin_warm_delta",
        "val_fake_target_skin_tint_delta",
        "val_fake_target_skin_luminance_ratio",
        "val_fake_target_skin_red_overshoot",
        "val_fake_target_skin_local_red_tail",
        "val_fake_target_skin_local_red_bad_fraction",
        "val_fake_target_skin_valid_fraction",
        "val_source_skin_fraction",
        "val_target_skin_fraction",
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
            "val_reference_selection_loss": "0.61",
            "val_fake_target_luminance_ratio": "0.96",
            "val_source_fake_l1": "0.042",
            "val_reference_response_l1": "0.036",
            "val_reference_condition_mean_abs": "0.19",
            "val_reference_direct_weight_rms": "0.014",
            "val_reference_direct_parameter_rms": "0.021",
            "val_reference_affine_weight_rms": "0.009",
            "val_reference_condition_saturation_fraction": "0.01",
            # Lightning averages the zero values from invalid mask pairs.
            # These raw values use valid_fraction=0.85; the report restores
            # the valid-only means and derives ratio from loss/base.
            "val_fake_target_skin_loss": "0.02635",
            "val_fake_target_skin_tone_loss": "0.0204",
            "val_source_target_skin_tone_loss": "0.051",
            "val_fake_target_skin_tone_ratio": "0.49",
            "val_fake_target_skin_chroma_loss": "0.0153",
            "val_fake_target_skin_spread_loss": "0.0102",
            "val_fake_target_skin_luminance_loss": "0.017",
            "val_fake_target_skin_uniformity_loss": "0.00425",
            "val_fake_target_skin_red_green_delta": "0.0068",
            "val_fake_target_skin_blue_green_delta": "-0.0034",
            "val_fake_target_skin_warm_delta": "0.0051",
            "val_fake_target_skin_tint_delta": "0.0017",
            "val_fake_target_skin_luminance_ratio": "0.8585",
            "val_fake_target_skin_red_overshoot": "0.00085",
            "val_fake_target_skin_local_red_tail": "0.00255",
            "val_fake_target_skin_local_red_bad_fraction": "0.0017",
            "val_fake_target_skin_valid_fraction": "0.85",
            "val_source_skin_fraction": "0.12",
            "val_target_skin_fraction": "0.11",
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
    assert summary["move"] == 0.042
    assert summary["response"] == 0.036
    assert summary["direct_weight"] == 0.014
    assert summary["direct"] == 0.021
    assert summary["affine_weight"] == 0.009
    assert math.isclose(summary["skin_ratio"], 0.40)
    assert math.isclose(summary["skin_loss"], 0.024)
    assert summary["skin_valid"] == 0.85
    assert format_summary(epoch, summary).startswith(
        "epoch=136 source_ref=0.030000 fake_ref=0.027000 ratio=0.900000"
    )
    assert format_summary(
        epoch,
        summary,
        COMPACT_OUTPUT_METRICS,
    ) == (
        "epoch=136 ratio=0.900000 move=0.042000 "
        "response=0.036000 direct=0.021000 "
        "luma_ratio=0.960000 red_bad=0.004000"
    )
    assert format_summary(
        epoch,
        summary,
        RED_OUTPUT_METRICS,
    ) == (
        "epoch=136 ratio=0.900000 move=0.042000 "
        "response=0.036000 luma_ratio=0.960000 "
        "rg_delta=0.012000 bg_delta=-0.008000 "
        "warm_bias=0.010000 warm_abs=0.014000 "
        "warm_positive=0.700000 tint_bias=-0.011000 "
        "tint_abs=0.013000 source_warm=0.030000 "
        "source_tint=-0.020000 red_tail=0.012000 "
        "red_bad=0.004000 red_overshoot=0.002000"
    )
    assert format_summary(
        epoch,
        summary,
        SKIN_OUTPUT_METRICS,
    ) == (
        "epoch=136 ratio=0.900000 move=0.042000 "
        "response=0.036000 skin_ratio=0.400000 "
        "skin_loss=0.024000 skin_base=0.060000 "
        "skin_rg=0.008000 skin_bg=-0.004000 "
        "skin_warm=0.006000 skin_tint=0.002000 "
        "skin_luma=1.010000 skin_red=0.001000 "
        "skin_red_tail=0.003000 skin_red_bad=0.002000 "
        "skin_valid=0.850000 source_skin=0.120000 "
        "target_skin=0.110000"
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
    assert summary["move"] is None
    assert summary["response"] is None
    assert summary["direct_weight"] is None
    assert summary["direct"] is None
    assert summary["affine_weight"] is None
    assert summary["skin_ratio"] is None
    assert summary["skin_loss"] is None
    assert summary["skin_valid"] is None


def test_skin_report_does_not_treat_zero_valid_fraction_as_improvement(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "no_valid_skin.csv"
    fieldnames = [
        "epoch",
        "val_reference_style_ratio",
        "val_fake_target_skin_tone_loss",
        "val_source_target_skin_tone_loss",
        "val_fake_target_skin_valid_fraction",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "3",
                "val_reference_style_ratio": "0.9",
                "val_fake_target_skin_tone_loss": "0",
                "val_source_target_skin_tone_loss": "0",
                "val_fake_target_skin_valid_fraction": "0",
            }
        )

    epoch, summary = summarize_metrics(metrics_path)
    output = format_summary(epoch, summary, SKIN_OUTPUT_METRICS)

    assert summary["skin_valid"] == 0
    assert summary["skin_loss"] is None
    assert summary["skin_base"] is None
    assert summary["skin_ratio"] is None
    assert "skin_ratio=NA" in output
