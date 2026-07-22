import csv
from pathlib import Path

from scripts.report_reference_metrics import format_summary, summarize_metrics


def _write_metrics(path: Path) -> None:
    fieldnames = [
        "epoch",
        "val_source_reference_style_distance",
        "val_fake_reference_style_distance",
        "val_reference_style_ratio",
        "val_reference_style_loss",
        "val_fake_target_luminance",
        "val_real_target_luminance",
        "val_range_loss",
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
            "val_fake_target_luminance": "0.47",
            "val_real_target_luminance": "0.49",
            "val_range_loss": "0.001",
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
    assert summary["target_luma"] == 0.49
    assert format_summary(epoch, summary).startswith(
        "epoch=136 source_ref=0.030000 fake_ref=0.027000 ratio=0.900000"
    )
