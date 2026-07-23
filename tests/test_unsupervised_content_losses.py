import torch

from cm_kan.ml.pipelines.unsupervised import UnsupervisedPipeline


def _encode_linear_srgb(images: torch.Tensor) -> torch.Tensor:
    return torch.where(
        images <= 0.0031308,
        images * 12.92,
        1.055 * images.pow(1 / 2.4) - 0.055,
    )


def test_chroma_loss_allows_multiplicative_brightness_change() -> None:
    image = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    image = image.expand(1, 3, 32, 32)

    loss = UnsupervisedPipeline._chroma_loss(image * 0.45, image)

    assert loss.item() < 1e-6


def test_chroma_loss_detects_hue_shift() -> None:
    image = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    image = image.expand(1, 3, 32, 32)
    shifted = image.clone()
    shifted[:, 0] = shifted[:, 0] * 0.55

    loss = UnsupervisedPipeline._chroma_loss(shifted, image)

    assert loss.item() > 0.05


def test_reflectance_loss_allows_global_brightness_change() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.rand((1, 3, 32, 32), generator=generator) * 0.7 + 0.2

    loss = UnsupervisedPipeline._reflectance_loss(image * 0.5, image)

    assert loss.item() < 1e-5


def test_reflectance_loss_detects_local_tone_change() -> None:
    image = torch.full((1, 3, 32, 32), 0.6)
    changed = image.clone()
    changed[:, :, 8:24, 8:24] *= 0.45

    loss = UnsupervisedPipeline._reflectance_loss(changed, image)

    assert loss.item() > 0.05


def test_patch_nce_prefers_matching_spatial_features() -> None:
    key = torch.eye(4).reshape(1, 4, 2, 2)
    matching_query = key.clone().requires_grad_(True)
    shifted_query = torch.roll(key, shifts=1, dims=-1)

    matching_loss = UnsupervisedPipeline._patch_nce_loss(
        [matching_query], [key], num_patches=4, temperature=0.07
    )
    shifted_loss = UnsupervisedPipeline._patch_nce_loss(
        [shifted_query], [key], num_patches=4, temperature=0.07
    )

    assert matching_loss.item() < shifted_loss.item()
    matching_loss.backward()
    assert matching_query.grad is not None
    assert torch.isfinite(matching_query.grad).all()


def test_white_balance_statistics_detect_warm_shift() -> None:
    neutral = torch.full((1, 3, 16, 16), 0.5)
    warm = neutral.clone()
    warm[:, 0] = 0.62
    warm[:, 2] = 0.38

    neutral_stats = UnsupervisedPipeline._white_balance_statistics(neutral)
    warm_stats = UnsupervisedPipeline._white_balance_statistics(warm)
    deltas = warm_stats - neutral_stats
    warm_bias = 0.5 * (deltas[:, 0] - deltas[:, 1])

    assert neutral_stats.abs().max().item() < 1e-6
    assert deltas[0, 0].item() > 0
    assert deltas[0, 1].item() < 0
    assert warm_bias.item() > 0


def test_white_balance_statistics_prefer_neutral_pixels_over_saturated_color() -> None:
    image = torch.tensor([0.90, 0.12, 0.10]).view(1, 3, 1, 1)
    image = image.expand(1, 3, 16, 16).clone()
    image[:, :, :, :4] = 0.5

    statistics = UnsupervisedPipeline._white_balance_statistics(image)

    assert statistics.abs().max().item() < 0.10


def test_local_chroma_loss_allows_global_channel_gains() -> None:
    generator = torch.Generator().manual_seed(11)
    source_linear = torch.rand((2, 3, 16, 16), generator=generator) * 0.35 + 0.15
    gains = torch.tensor([1.15, 0.90, 0.80]).view(1, 3, 1, 1)
    prediction_linear = source_linear * gains

    loss = UnsupervisedPipeline._local_chroma_loss(
        _encode_linear_srgb(prediction_linear),
        _encode_linear_srgb(source_linear),
    )

    assert loss.item() < 0.005


def test_local_chroma_loss_detects_spatially_inconsistent_color_shift() -> None:
    source = torch.full((1, 3, 16, 16), 0.5)
    prediction = source.clone()
    prediction[:, 0, 4:12, 4:12] = 0.75

    loss = UnsupervisedPipeline._local_chroma_loss(prediction, source)

    assert loss.item() > 0.05


def test_local_chroma_loss_has_finite_prediction_gradients() -> None:
    prediction = torch.rand((2, 3, 16, 16), requires_grad=True)
    source = torch.rand((2, 3, 16, 16))

    loss = UnsupervisedPipeline._local_chroma_loss(prediction, source)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_white_balance_statistics_have_finite_gradients() -> None:
    prediction = torch.rand((2, 3, 16, 16), requires_grad=True)
    reference = torch.rand((2, 3, 16, 16))
    deltas = UnsupervisedPipeline._white_balance_statistics(prediction) - (
        UnsupervisedPipeline._white_balance_statistics(reference).detach()
    )
    warm_delta = 0.5 * (deltas[:, 0] - deltas[:, 1])
    tint_delta = 0.5 * (deltas[:, 0] + deltas[:, 1])
    loss = (
        UnsupervisedPipeline._charbonnier(warm_delta)
        + 0.5 * UnsupervisedPipeline._charbonnier(tint_delta)
    ).mean()

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_white_balance_weight_ramps_by_absolute_epoch() -> None:
    weights = [
        UnsupervisedPipeline._ramped_weight(3.0, 5, epoch)
        for epoch in range(7)
    ]

    expected = [0.6, 1.2, 1.8, 2.4, 3.0, 3.0, 3.0]
    assert all(
        abs(actual - target) < 1e-9
        for actual, target in zip(weights, expected)
    )
    assert UnsupervisedPipeline._ramped_weight(3.0, 0, 157) == 3.0
