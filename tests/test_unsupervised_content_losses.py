import math
from unittest.mock import Mock, patch

import pytest
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


def test_soft_skin_mask_classifies_synthetic_colors() -> None:
    colors = torch.tensor(
        [
            [0.62, 0.41, 0.28],
            [0.25, 0.16, 0.10],
            [0.90, 0.72, 0.62],
            [0.50, 0.50, 0.50],
            [0.20, 0.30, 0.70],
            [0.80, 0.10, 0.10],
        ],
    ).reshape(6, 3, 1, 1)

    mask = UnsupervisedPipeline._soft_skin_mask(colors)
    values = mask[:, 0, 0, 0]

    assert mask.shape == (6, 1, 1, 1)
    assert torch.isfinite(mask).all()
    assert ((0 <= mask) & (mask <= 1)).all()
    assert values[0].item() > 0.8
    assert values[1].item() > 0.5
    assert values[2].item() > 0.8
    assert values[3].item() < 0.01
    assert values[4].item() < 1e-3
    assert values[5].item() < 1e-3


def _skin_patch_image(
    background: tuple[float, float, float],
    skin: tuple[float, float, float],
    patch: tuple[slice, slice],
) -> torch.Tensor:
    image = torch.tensor(background).view(1, 3, 1, 1)
    image = image.expand(1, 3, 24, 24).clone()
    image[:, :, patch[0], patch[1]] = torch.tensor(skin).view(1, 3, 1, 1)
    return image


def test_skin_loss_ignores_background_and_does_not_require_pixel_alignment() -> None:
    source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(4, 12), slice(4, 12)),
    )
    target = _skin_patch_image(
        (0.80, 0.10, 0.10),
        (0.72, 0.48, 0.35),
        (slice(12, 20), slice(12, 20)),
    )
    prediction = _skin_patch_image(
        (0.10, 0.70, 0.15),
        (0.72, 0.48, 0.35),
        (slice(4, 12), slice(4, 12)),
    )

    matched = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
    )
    red_prediction = prediction.clone()
    red_prediction[:, :, 4:12, 4:12] = torch.tensor(
        [0.90, 0.25, 0.18],
    ).view(1, 3, 1, 1)
    red = UnsupervisedPipeline._reference_skin_tone_terms(
        red_prediction,
        source,
        target,
    )

    assert matched['valid_fraction'].item() == 1
    assert matched['tone_loss'].item() < 0.01
    assert red['chroma_loss'].item() > matched['chroma_loss'].item() + 0.10
    assert red['red_overshoot'].item() > 0.10


def test_skin_loss_uses_source_mask_even_when_prediction_turns_blue() -> None:
    source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(4, 20), slice(4, 20)),
    )
    target = source.clone()
    prediction = source.clone()
    prediction[:, :, 4:20, 4:20] = torch.tensor(
        [0.15, 0.25, 0.75],
    ).view(1, 3, 1, 1)

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
    )

    assert terms['valid_fraction'].item() == 1
    assert terms['chroma_loss'].item() > 0.5


def test_skin_loss_has_finite_gradients_and_detaches_real_images() -> None:
    source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(4, 20), slice(4, 20)),
    ).requires_grad_(True)
    target = _skin_patch_image(
        (0.15, 0.65, 0.20),
        (0.68, 0.46, 0.34),
        (slice(4, 20), slice(4, 20)),
    ).requires_grad_(True)
    prediction = source.detach().clone()
    prediction[:, 0, 4:20, 4:20] = 0.85
    prediction.requires_grad_(True)

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
    )
    terms['loss'].backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, :, 4:20, 4:20].abs().sum().item() > 0
    assert source.grad is None
    assert target.grad is None


def test_empty_skin_masks_return_differentiable_zero() -> None:
    source = torch.tensor([0.20, 0.30, 0.70]).view(1, 3, 1, 1)
    source = source.expand(1, 3, 16, 16)
    target = torch.full((1, 3, 16, 16), 0.5)
    prediction = source.clone().requires_grad_(True)

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
    )
    terms['loss'].backward()

    assert terms['valid_fraction'].item() == 0
    assert terms['loss'].item() == 0
    assert prediction.grad is not None
    assert prediction.grad.count_nonzero().item() == 0


def test_excessive_skin_coverage_is_rejected_as_background_contamination() -> None:
    source = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    source = source.expand(1, 3, 16, 16)
    target = torch.tensor([0.70, 0.46, 0.32]).view(1, 3, 1, 1)
    target = target.expand(1, 3, 16, 16)

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        source,
        source,
        target,
        max_fraction=0.5,
    )

    assert terms['input_fraction'].item() > 0.5
    assert terms['reference_fraction'].item() > 0.5
    assert terms['valid_fraction'].item() == 0
    assert terms['loss'].item() == 0


def test_face_roi_excludes_skin_colored_background_from_tone_statistics() -> None:
    source = _skin_patch_image(
        (0.90, 0.72, 0.62),
        (0.62, 0.41, 0.28),
        (slice(4, 12), slice(4, 12)),
    )
    target = _skin_patch_image(
        (0.25, 0.16, 0.10),
        (0.72, 0.48, 0.35),
        (slice(12, 20), slice(12, 20)),
    )
    prediction = source.clone()
    prediction[:, :, 4:12, 4:12] = torch.tensor(
        [0.72, 0.48, 0.35],
    ).view(1, 3, 1, 1)
    source_face_mask = torch.zeros((1, 1, 24, 24))
    source_face_mask[:, :, 4:12, 4:12] = 1
    target_face_mask = torch.zeros((1, 1, 24, 24))
    target_face_mask[:, :, 12:20, 12:20] = 1

    roi_terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
        input_face_mask=source_face_mask,
        reference_face_mask=target_face_mask,
        max_fraction=1.0,
    )
    color_only_terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
        max_fraction=1.0,
    )

    assert roi_terms['valid_fraction'].item() == 1
    assert roi_terms['input_face_fraction'].item() == pytest.approx(64 / 576)
    assert roi_terms['reference_face_fraction'].item() == pytest.approx(
        64 / 576
    )
    assert roi_terms['tone_loss'].item() < 0.01
    assert color_only_terms['tone_loss'].item() > (
        roi_terms['tone_loss'].item() + 0.10
    )


def test_face_roi_adapts_bhw_mask_to_image_spatial_shape() -> None:
    images = torch.zeros((2, 3, 16, 24))
    face_mask = torch.zeros((2, 8, 12))
    face_mask[0, 2:6, 3:9] = 1
    face_mask[1, 1:7, 2:10] = 1

    prepared = UnsupervisedPipeline._prepare_face_roi(
        face_mask,
        images,
        "face_mask",
    )

    assert prepared.shape == (2, 1, 16, 24)
    assert torch.equal(
        prepared[0, 0, 4:12, 6:18],
        torch.ones((8, 12)),
    )
    assert prepared.device == images.device
    assert prepared.dtype == images.dtype


def test_face_roi_accepts_rgb_bchw_and_channel_last_masks() -> None:
    images = torch.zeros((1, 3, 16, 24))
    channel_first = torch.zeros((1, 3, 8, 12))
    channel_first[:, 0, 2:6, 3:9] = 1
    channel_last = channel_first.permute(0, 2, 3, 1)

    prepared_first = UnsupervisedPipeline._prepare_face_roi(
        channel_first,
        images,
        "channel_first",
    )
    prepared_last = UnsupervisedPipeline._prepare_face_roi(
        channel_last,
        images,
        "channel_last",
    )

    assert prepared_first.shape == (1, 1, 16, 24)
    assert torch.equal(prepared_first, prepared_last)


def test_face_roi_rejects_unsafe_batch_broadcast() -> None:
    images = torch.zeros((2, 3, 16, 24))
    one_mask = torch.zeros((1, 1, 8, 12))

    with pytest.raises(ValueError, match="automatic batch broadcasting"):
        UnsupervisedPipeline._prepare_face_roi(
            one_mask,
            images,
            "face_mask",
        )


def test_face_roi_rejects_mismatched_spatial_aspect_ratio() -> None:
    images = torch.zeros((1, 3, 16, 24))
    square_mask = torch.zeros((1, 1, 8, 8))

    with pytest.raises(ValueError, match="aspect ratios match"):
        UnsupervisedPipeline._prepare_face_roi(
            square_mask,
            images,
            "face_mask",
        )


def test_black_face_roi_skips_only_the_skin_objective() -> None:
    source = torch.tensor([0.62, 0.41, 0.28]).view(1, 3, 1, 1)
    source = source.expand(1, 3, 16, 16)
    target = torch.tensor([0.72, 0.48, 0.35]).view(1, 3, 1, 1)
    target = target.expand(1, 3, 16, 16)
    prediction = source.clone().requires_grad_(True)
    black_roi = torch.zeros((1, 1, 16, 16))

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        prediction,
        source,
        target,
        input_face_mask=black_roi,
        reference_face_mask=black_roi,
    )
    terms['loss'].backward()

    assert terms['input_face_fraction'].item() == 0
    assert terms['reference_face_fraction'].item() == 0
    assert terms['valid_fraction'].item() == 0
    assert terms['loss'].item() == 0
    assert prediction.grad is not None
    assert prediction.grad.count_nonzero().item() == 0


def test_face_quality_gates_reject_uniform_skin_colored_false_box() -> None:
    source = torch.full((1, 3, 24, 24), 0.5)
    source[:, 0] = 0.90
    source[:, 1] = 0.72
    source[:, 2] = 0.62
    target = source.clone()
    face_roi = torch.zeros((1, 1, 24, 24))
    face_roi[:, :, 6:14, 6:14] = 1

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        source,
        source,
        target,
        input_face_mask=face_roi,
        reference_face_mask=face_roi,
        face_min_fraction=0.01,
        face_max_fraction=0.35,
        skin_face_density_min=0.10,
        skin_face_density_max=0.90,
        face_pair_area_ratio_min=0.5,
        face_pair_area_ratio_max=2.0,
    )

    assert terms['input_skin_face_density'].item() > 0.90
    assert terms['reference_skin_face_density'].item() > 0.90
    assert terms['valid_fraction'].item() == 0
    assert terms['loss'].item() == 0
    assert terms['input_mask'].count_nonzero().item() == 0


def test_skin_statistics_exclude_soft_weights_below_preview_threshold() -> None:
    inputs = torch.rand((1, 3, 4, 4))
    reference = torch.rand((1, 3, 4, 4))
    predictions = inputs.clone().requires_grad_(True)
    input_soft_mask = torch.full((1, 1, 4, 4), 0.24)
    reference_soft_mask = torch.full((1, 1, 4, 4), 0.24)
    input_soft_mask[:, :, 0, :2] = 0.26
    reference_soft_mask[:, :, 0, :2] = 0.26

    with patch.object(
        UnsupervisedPipeline,
        "_soft_skin_mask",
        side_effect=(input_soft_mask, reference_soft_mask),
    ):
        terms = UnsupervisedPipeline._reference_skin_tone_terms(
            predictions,
            inputs,
            reference,
            max_fraction=1.0,
        )

    assert terms['valid_fraction'].item() == 1
    assert terms['input_fraction'].item() == pytest.approx(2 / 16)
    assert terms['input_mask'][0, 0, 0, :2].tolist() == pytest.approx(
        [0.26, 0.26]
    )
    assert terms['input_mask'][0, 0, 1:].count_nonzero().item() == 0
    assert terms['input_mask'][0, 0, 0, 2:].count_nonzero().item() == 0


def test_face_quality_gates_reject_mismatched_pair_area() -> None:
    source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(4, 12), slice(4, 12)),
    )
    target = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.72, 0.48, 0.35),
        (slice(4, 8), slice(4, 8)),
    )
    source_roi = torch.zeros((1, 1, 24, 24))
    source_roi[:, :, 4:12, 4:12] = 1
    target_roi = torch.zeros((1, 1, 24, 24))
    target_roi[:, :, 4:8, 4:8] = 1

    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        source,
        source,
        target,
        input_face_mask=source_roi,
        reference_face_mask=target_roi,
        skin_face_density_max=1.0,
        face_pair_area_ratio_min=0.5,
        face_pair_area_ratio_max=2.0,
    )

    assert terms['face_pair_area_ratio'].item() == pytest.approx(4.0)
    assert terms['valid_fraction'].item() == 0


def test_face_quality_gates_reject_distant_pair_centers() -> None:
    source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(2, 10), slice(2, 10)),
    )
    target = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.72, 0.48, 0.35),
        (slice(14, 22), slice(14, 22)),
    )
    source_roi = torch.zeros((1, 1, 24, 24))
    source_roi[:, :, 2:10, 2:10] = 1
    target_roi = torch.zeros((1, 1, 24, 24))
    target_roi[:, :, 14:22, 14:22] = 1

    baseline_terms = UnsupervisedPipeline._reference_skin_tone_terms(
        source,
        source,
        target,
        input_face_mask=source_roi,
        reference_face_mask=target_roi,
        skin_face_density_max=1.0,
        face_pair_center_distance_max=1.0,
    )
    terms = UnsupervisedPipeline._reference_skin_tone_terms(
        source,
        source,
        target,
        input_face_mask=source_roi,
        reference_face_mask=target_roi,
        skin_face_density_max=1.0,
        face_pair_center_distance_max=0.30,
    )

    assert baseline_terms['valid_fraction'].item() == 1
    assert terms['face_pair_center_distance'].item() > 0.30
    assert terms['valid_fraction'].item() == 0
    assert terms['input_mask'].count_nonzero().item() == 0


def test_required_face_masks_fail_instead_of_using_color_only_fallback() -> None:
    pipeline = UnsupervisedPipeline.__new__(UnsupervisedPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.reference_skin_require_face_mask = True
    pipeline.reference_skin_tone_weight = 2.0
    image = torch.rand((1, 3, 16, 16))

    with pytest.raises(ValueError, match="requires face masks"):
        pipeline._face_masks_from_batch({
            "source": image,
            "target": image,
        })


def test_invalid_skin_samples_do_not_dilute_valid_batch() -> None:
    valid_source = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.62, 0.41, 0.28),
        (slice(4, 20), slice(4, 20)),
    )
    valid_target = _skin_patch_image(
        (0.20, 0.30, 0.70),
        (0.70, 0.46, 0.32),
        (slice(4, 20), slice(4, 20)),
    )
    valid_prediction = valid_source.clone()
    empty = torch.full_like(valid_source, 0.5)
    batch_terms = UnsupervisedPipeline._reference_skin_tone_terms(
        torch.cat([valid_prediction, empty]),
        torch.cat([valid_source, empty]),
        torch.cat([valid_target, empty]),
    )
    valid_terms = UnsupervisedPipeline._reference_skin_tone_terms(
        valid_prediction,
        valid_source,
        valid_target,
    )

    assert batch_terms['valid_fraction'].item() == 0.5
    assert torch.allclose(batch_terms['loss'], valid_terms['loss'])


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


def test_local_chroma_terms_accept_zero_spatial_weights() -> None:
    source = torch.full((1, 3, 16, 16), 0.5)
    prediction = source.clone()
    prediction[:, 0, 4:12, 4:12] = 0.9

    terms = UnsupervisedPipeline._local_chroma_terms(
        prediction,
        source,
        spatial_weights=torch.zeros((1, 1, 16, 16)),
    )

    assert all(torch.isfinite(value) for value in terms.values())
    assert all(value.item() == 0 for value in terms.values())


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


def _environment_pair(
    source_background: tuple[float, float, float],
    target_background: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.tensor(source_background).view(1, 3, 1, 1)
    source = source.expand(1, 3, 32, 32).clone()
    target = torch.tensor(target_background).view(1, 3, 1, 1)
    target = target.expand(1, 3, 32, 32).clone()
    source_face = torch.zeros((1, 1, 32, 32))
    target_face = torch.zeros((1, 1, 32, 32))
    source_face[:, :, 8:20, 6:18] = 1
    target_face[:, :, 10:22, 12:24] = 1
    source[:, :, 8:20, 6:18] = torch.tensor(
        [0.62, 0.41, 0.28]
    ).view(1, 3, 1, 1)
    target[:, :, 10:22, 12:24] = torch.tensor(
        [0.72, 0.48, 0.35]
    ).view(1, 3, 1, 1)
    return source, target, source_face, target_face


def test_environment_loss_matches_background_despite_face_misalignment() -> None:
    source, target, source_face, target_face = _environment_pair(
        (0.30, 0.34, 0.42),
        (0.48, 0.40, 0.30),
    )
    prediction = target.clone()
    prediction[:, :, 10:22, 12:24] = torch.tensor(
        [0.48, 0.40, 0.30]
    ).view(1, 3, 1, 1)
    prediction[:, :, 8:20, 6:18] = torch.tensor(
        [0.72, 0.48, 0.35]
    ).view(1, 3, 1, 1)

    terms = UnsupervisedPipeline._reference_environment_color_terms(
        prediction,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
    )

    assert terms["valid_fraction"].item() == pytest.approx(1.0)
    assert terms["loss"].item() < 0.01
    assert terms["warm_abs"].item() < 0.01
    assert terms["tint_abs"].item() < 0.01


def test_environment_loss_detects_source_colored_background() -> None:
    source, target, source_face, target_face = _environment_pair(
        (0.30, 0.34, 0.42),
        (0.48, 0.40, 0.30),
    )

    source_terms = UnsupervisedPipeline._reference_environment_color_terms(
        source,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
    )
    matched_prediction = target.clone()
    matched_prediction[:, :, 10:22, 12:24] = torch.tensor(
        [0.48, 0.40, 0.30]
    ).view(1, 3, 1, 1)
    matched_prediction[:, :, 8:20, 6:18] = torch.tensor(
        [0.72, 0.48, 0.35]
    ).view(1, 3, 1, 1)
    target_terms = UnsupervisedPipeline._reference_environment_color_terms(
        matched_prediction,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
    )

    assert source_terms["loss"].item() > target_terms["loss"].item() + 0.10
    assert source_terms["warm_abs"].item() > 0.10


def test_environment_loss_has_prediction_only_finite_gradients() -> None:
    source, target, source_face, target_face = _environment_pair(
        (0.30, 0.34, 0.42),
        (0.48, 0.40, 0.30),
    )
    source.requires_grad_(True)
    target.requires_grad_(True)
    prediction = source.detach().clone().requires_grad_(True)

    terms = UnsupervisedPipeline._reference_environment_color_terms(
        prediction,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
    )
    terms["loss"].backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0
    assert source.grad is None
    assert target.grad is None


def test_environment_pyramid_detects_spatial_color_swap() -> None:
    reference = torch.empty((1, 3, 32, 32))
    reference[:, :, :16] = torch.tensor(
        [0.52, 0.39, 0.28]
    ).view(1, 3, 1, 1)
    reference[:, :, 16:] = torch.tensor(
        [0.26, 0.35, 0.50]
    ).view(1, 3, 1, 1)
    prediction = reference.flip(-2)

    terms = UnsupervisedPipeline._reference_environment_color_terms(
        prediction,
        reference,
        reference,
        face_dilation=0,
    )

    assert terms["scale_1_loss"].item() < 0.01
    assert terms["scale_2_loss"].item() > 0.10
    assert terms["scale_4_loss"].item() > 0.10


def test_environment_loss_returns_differentiable_zero_without_background() -> None:
    prediction = torch.zeros(
        (1, 3, 16, 16),
        requires_grad=True,
    )
    image = torch.zeros_like(prediction)
    full_face = torch.ones((1, 1, 16, 16))

    terms = UnsupervisedPipeline._reference_environment_color_terms(
        prediction,
        image,
        image,
        input_face_mask=full_face,
        reference_face_mask=full_face,
        face_dilation=3,
    )
    terms["loss"].backward()

    assert terms["loss"].item() == 0
    assert terms["valid_fraction"].item() == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.count_nonzero().item() == 0


def test_invalid_environment_sample_does_not_dilute_valid_batch() -> None:
    source, target, source_face, target_face = _environment_pair(
        (0.30, 0.34, 0.42),
        (0.48, 0.40, 0.30),
    )
    valid_terms = UnsupervisedPipeline._reference_environment_color_terms(
        source,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
    )
    empty = torch.zeros_like(source)
    full_face = torch.ones_like(source_face)
    batch_terms = UnsupervisedPipeline._reference_environment_color_terms(
        torch.cat([source, empty]),
        torch.cat([source, empty]),
        torch.cat([target, empty]),
        input_face_mask=torch.cat([source_face, full_face]),
        reference_face_mask=torch.cat([target_face, full_face]),
        face_dilation=3,
    )

    assert torch.allclose(batch_terms["loss"], valid_terms["loss"])
    assert batch_terms["valid_fraction"].item() == pytest.approx(0.5)


def test_environment_loss_ignores_both_face_masks_if_either_is_implausible() -> None:
    source, target, source_face, target_face = _environment_pair(
        (0.30, 0.34, 0.42),
        (0.48, 0.40, 0.30),
    )
    target_face.fill_(1)

    gated_terms = UnsupervisedPipeline._reference_environment_color_terms(
        source,
        source,
        target,
        input_face_mask=source_face,
        reference_face_mask=target_face,
        face_dilation=3,
        face_min_fraction=0.01,
        face_max_fraction=0.35,
    )
    unmasked_terms = UnsupervisedPipeline._reference_environment_color_terms(
        source,
        source,
        target,
        face_dilation=3,
    )

    assert torch.allclose(gated_terms["loss"], unmasked_terms["loss"])
    assert gated_terms["valid_fraction"].item() == pytest.approx(1.0)


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
