import pytest
import torch

from cm_kan.ml.models import ReferenceCycleCmKAN, ReferenceStyleEncoder


def _solid_color(rgb: tuple[float, float, float], size: int = 32) -> torch.Tensor:
    return torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1).expand(
        1, 3, size, size
    )


def _reference_model() -> ReferenceCycleCmKAN:
    return ReferenceCycleCmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )


def test_reference_style_encoder_returns_per_image_color_statistics() -> None:
    encoder = ReferenceStyleEncoder()
    red = _solid_color((1.0, 0.0, 0.0))
    blue = _solid_color((0.0, 0.0, 1.0))

    styles = encoder(torch.cat([red, blue], dim=0))

    assert styles.shape == (2, ReferenceStyleEncoder.output_dim)
    assert torch.allclose(styles[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(styles[1, :3], torch.tensor([0.0, 0.0, 1.0]))
    assert not torch.allclose(styles[0], styles[1])


def test_reference_style_condition_describes_requested_change() -> None:
    model = _reference_model()
    source = _solid_color((0.45, 0.40, 0.35))
    warm_reference = _solid_color((0.70, 0.48, 0.24))

    identity_condition = model.style_condition(source, source)
    transfer_condition = model.style_condition(source, warm_reference)

    assert identity_condition.shape == (1, ReferenceStyleEncoder.output_dim)
    assert torch.allclose(identity_condition, torch.zeros_like(identity_condition))
    assert torch.allclose(
        transfer_condition,
        model.encode_style(warm_reference) - model.encode_style(source),
    )
    assert torch.count_nonzero(transfer_condition) > 0


def test_reference_model_adds_zero_initialized_style_affine_layers() -> None:
    model = _reference_model()

    for generator in (model.gen_ab, model.gen_ba):
        style_affine = generator.layers[0].generator.style_affine
        assert style_affine is not None
        assert style_affine[0].in_features == ReferenceStyleEncoder.output_dim
        assert torch.count_nonzero(style_affine[-1].weight) == 0
        assert torch.count_nonzero(style_affine[-1].bias) == 0


def test_reference_generator_rejects_a_missing_or_malformed_condition() -> None:
    model = _reference_model()
    source = torch.rand(1, 3, 32, 32)

    with pytest.raises(ValueError, match="reference style condition is required"):
        model.gen_ab(source)

    with pytest.raises(ValueError, match="must have shape"):
        model.gen_ab(source, torch.zeros(1, ReferenceStyleEncoder.output_dim - 1))
