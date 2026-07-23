import math
from unittest.mock import Mock

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


def test_local_chroma_terms_allow_uniform_global_channel_gains() -> None:
    generator = torch.Generator().manual_seed(13)
    source_linear = torch.rand((2, 3, 32, 32), generator=generator) * 0.30 + 0.20
    gains = torch.tensor([1.18, 0.91, 0.82]).view(1, 3, 1, 1)
    prediction_linear = source_linear * gains

    terms = UnsupervisedPipeline._local_chroma_terms(
        _encode_linear_srgb(prediction_linear),
        _encode_linear_srgb(source_linear),
        chroma_tail_fraction=0.05,
        red_tail_fraction=0.02,
        chroma_threshold=0.25,
        red_threshold=math.log(1.2),
    )

    assert set(terms) == {
        "mean",
        "chroma_tail",
        "red_tail",
        "chroma_bad_fraction",
        "red_bad_fraction",
    }
    assert terms["mean"].item() < 0.005
    assert terms["chroma_tail"].item() < 0.005
    assert terms["red_tail"].item() < 0.005
    assert terms["chroma_bad_fraction"].item() == 0
    assert terms["red_bad_fraction"].item() == 0


def test_local_chroma_tail_exposes_a_small_severe_red_patch() -> None:
    source = torch.full((1, 3, 64, 64), 0.5)
    prediction = source.clone()
    prediction[:, 0, 24:32, 24:32] = 0.90

    terms = UnsupervisedPipeline._local_chroma_terms(
        prediction,
        source,
        chroma_tail_fraction=0.05,
        red_tail_fraction=0.02,
        chroma_threshold=0.25,
        red_threshold=math.log(1.2),
    )

    # A 64-pixel defect is only 1.56% of this image. The tail terms must keep
    # that region visible instead of diluting it into the whole-image mean.
    assert terms["chroma_tail"].item() > 3 * terms["mean"].item()
    assert terms["red_tail"].item() > 5 * terms["mean"].item()
    assert 0.005 < terms["chroma_bad_fraction"].item() < 0.05
    assert 0.005 < terms["red_bad_fraction"].item() < 0.05


def test_local_red_tail_is_one_sided_and_specific_to_red_overflow() -> None:
    source = torch.full((1, 3, 64, 64), 0.5)
    red_patch = source.clone()
    blue_patch = source.clone()
    red_patch[:, 0, 24:32, 24:32] = 0.90
    blue_patch[:, 2, 24:32, 24:32] = 0.90

    red_terms = UnsupervisedPipeline._local_chroma_terms(
        red_patch,
        source,
        red_threshold=math.log(1.2),
    )
    blue_terms = UnsupervisedPipeline._local_chroma_terms(
        blue_patch,
        source,
        red_threshold=math.log(1.2),
    )

    assert red_terms["red_tail"].item() > 0.10
    assert red_terms["red_tail"].item() > blue_terms["red_tail"].item() + 0.10
    assert red_terms["red_bad_fraction"].item() > 0
    assert blue_terms["red_bad_fraction"].item() == 0


def test_local_chroma_tail_terms_have_finite_prediction_gradients() -> None:
    generator = torch.Generator().manual_seed(17)
    prediction = torch.rand(
        (2, 3, 32, 32),
        generator=generator,
        requires_grad=True,
    )
    source = torch.rand((2, 3, 32, 32), generator=generator)

    terms = UnsupervisedPipeline._local_chroma_terms(prediction, source)
    loss = terms["mean"] + terms["chroma_tail"] + terms["red_tail"]
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_range_tail_exposes_sparse_values_that_would_be_clipped() -> None:
    prediction = torch.full((1, 3, 64, 64), 0.5)
    prediction[:, 0, 0, 0] = 4.0

    terms = UnsupervisedPipeline._range_terms(prediction)

    assert set(terms) == {"mean", "tail", "out_of_range_fraction"}
    assert terms["mean"].item() > 0
    assert terms["tail"].item() > 50 * terms["mean"].item()
    assert 0 < terms["out_of_range_fraction"].item() < 0.001


def test_range_terms_are_zero_for_bounded_predictions() -> None:
    prediction = torch.rand((2, 3, 32, 32))

    terms = UnsupervisedPipeline._range_terms(prediction)

    assert terms["mean"].item() == 0
    assert terms["tail"].item() == 0
    assert terms["out_of_range_fraction"].item() == 0


def _red_overshoot_pipeline() -> UnsupervisedPipeline:
    pipeline = UnsupervisedPipeline.__new__(UnsupervisedPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.reference_guided = True
    pipeline.reference_red_overshoot_margin = 0.0
    return pipeline


def test_reference_red_overshoot_penalizes_only_excess_red() -> None:
    pipeline = _red_overshoot_pipeline()
    reference = torch.full((1, 3, 32, 32), 0.5)
    redder = reference.clone()
    redder[:, 0] = 0.70
    less_red = reference.clone()
    less_red[:, 0] = 0.35

    redder_loss = pipeline._reference_red_overshoot_loss(redder, reference)
    less_red_loss = pipeline._reference_red_overshoot_loss(less_red, reference)

    assert redder_loss.item() > 0.10
    assert less_red_loss.item() < 1e-7


def test_reference_red_overshoot_has_finite_useful_gradients() -> None:
    pipeline = _red_overshoot_pipeline()
    reference = torch.full((1, 3, 32, 32), 0.5)
    prediction = reference.clone()
    prediction[:, 0] = 0.65
    prediction.requires_grad_(True)

    loss = pipeline._reference_red_overshoot_loss(prediction, reference)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 0].abs().sum().item() > 0


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


def test_adversarial_weight_waits_for_warmup_then_ramps() -> None:
    weights = [
        UnsupervisedPipeline._ramped_adversarial_weight(
            weight=1.0,
            warmup_epochs=5,
            ramp_epochs=10,
            current_epoch=epoch,
        )
        for epoch in (0, 4, 5, 9, 14, 20)
    ]

    assert weights == [0.0, 0.0, 0.1, 0.5, 1.0, 1.0]


def test_generator_warmup_uses_full_objective_without_adversarial_loss() -> None:
    pipeline = UnsupervisedPipeline.__new__(UnsupervisedPipeline)
    torch.nn.Module.__init__(pipeline)
    expected_loss = torch.tensor(1.25, requires_grad=True)
    pipeline.generator_training_step = Mock(return_value=expected_loss)
    pipeline._log_loss = Mock()
    source = torch.rand(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)

    loss = pipeline.generator_warmup_step(source, target)

    assert loss is expected_loss
    pipeline.generator_training_step.assert_called_once_with(
        source,
        target,
        adversarial_weight=0.0,
    )
