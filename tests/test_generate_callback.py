from types import SimpleNamespace

import torch

from cm_kan.ml.callbacks.generate import GenerateCallback


def test_srgb_primaries_convert_to_expected_cie_xy() -> None:
    primaries = torch.eye(3).reshape(3, 1, 3).permute(2, 0, 1)
    xy = GenerateCallback._rgb_to_xy(primaries)

    expected = torch.tensor(
        [
            [0.6400, 0.3300],
            [0.3000, 0.6000],
            [0.1500, 0.0600],
        ]
    )
    assert torch.allclose(xy, expected, atol=1e-4)


def test_black_pixels_are_excluded_from_cie_xy() -> None:
    black = torch.zeros(3, 2, 2)

    assert GenerateCallback._rgb_to_xy(black).shape == (0, 2)


def test_adaptive_xy_limits_zoom_without_log_transform() -> None:
    xy = torch.tensor(
        [
            [0.30, 0.31],
            [0.32, 0.33],
            [0.34, 0.35],
            [0.36, 0.37],
        ]
    )

    x_limits, y_limits = GenerateCallback._adaptive_xy_limits(xy)

    assert x_limits[1] - x_limits[0] < 0.2
    assert y_limits[1] - y_limits[0] < 0.2
    assert x_limits[0] < xy[:, 0].median() < x_limits[1]
    assert y_limits[0] < xy[:, 1].median() < y_limits[1]


def test_preview_selects_largest_chromaticity_gap() -> None:
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
    green = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
    callback = GenerateCallback(max_preview_candidates=2)
    dataloader = [
        {"source": red, "target": red},
        {"source": red, "target": green},
    ]

    callback._capture_most_distinct_batch(
        dataloader,
        SimpleNamespace(device=torch.device("cpu")),
    )

    assert torch.equal(callback.input_imgs, red)
    assert torch.equal(callback.target_imgs, green)
    assert callback.preview_pair_distance > 0.3


def test_scatter_tile_matches_preview_image_shape() -> None:
    source = torch.linspace(0, 1, steps=48).reshape(1, 3, 4, 4)
    translated = (source * 0.8 + 0.1).clamp(0, 1)
    target = (source * 0.6 + 0.2).clamp(0, 1)
    preview = GenerateCallback._four_column_preview(
        source,
        translated,
        target,
    )

    assert preview.shape == (4, 3, 8, 8)
    assert preview[3].min().item() >= 0
    assert preview[3].max().item() <= 1
