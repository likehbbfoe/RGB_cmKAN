import torch

from cm_kan.ml.callbacks.generate import GenerateCallback


def test_color_statistics_use_display_range_and_normalized_histogram() -> None:
    image = torch.tensor(
        [[[-0.2, 1.2]], [[0.25, 0.75]], [[0.5, 0.5]]]
    )

    values = GenerateCallback._display_values(image)
    _, histogram = GenerateCallback._histogram(values, bins=4)

    assert torch.equal(
        values,
        torch.tensor([0.0, 1.0, 0.25, 0.75, 0.5, 0.5]),
    )
    assert abs(sum(histogram) - 1.0) < 1e-6


def test_distribution_tile_matches_preview_image_shape() -> None:
    source = torch.linspace(0, 1, steps=48).reshape(1, 3, 4, 4)
    translated = (source * 0.8 + 0.1).clamp(0, 1)
    target = (source * 0.6 + 0.2).clamp(0, 1)
    preview = GenerateCallback._four_column_preview(
        source,
        translated,
        target,
    )

    assert preview.shape == (4, 3, 4, 4)
    assert preview[3].min().item() >= 0
    assert preview[3].max().item() <= 1
