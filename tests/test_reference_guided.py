import pytest
import torch

from cm_kan.ml.layers import CmKANLayer
from cm_kan.ml.models import ReferenceCycleCmKAN, ReferenceStyleEncoder


def _solid_color(rgb: tuple[float, float, float], size: int = 32) -> torch.Tensor:
    return torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1).expand(
        1, 3, size, size
    )


def _reference_model(**kwargs) -> ReferenceCycleCmKAN:
    return ReferenceCycleCmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
        **kwargs,
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


def test_bounded_logit_residual_has_identity_at_zero_residual() -> None:
    layer = CmKANLayer(
        in_channels=3,
        out_channels=3,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
        output_mode="bounded_logit_residual",
        max_logit_shift=1.5,
    )
    inputs = torch.tensor([0.20, 0.50, 0.80]).view(1, 3, 1, 1)

    output = layer._apply_output_mode(inputs, torch.zeros_like(inputs))

    assert torch.allclose(output, inputs, atol=1e-6, rtol=0)


@pytest.mark.parametrize("raw_value", [-100.0, 100.0])
def test_bounded_logit_residual_stays_in_gamut_and_limits_shift(
    raw_value: float,
) -> None:
    layer = CmKANLayer(
        in_channels=3,
        out_channels=3,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
        output_mode="bounded_logit_residual",
        max_logit_shift=1.5,
    )
    inputs = torch.tensor([0.20, 0.50, 0.80]).view(1, 3, 1, 1)
    raw = torch.full_like(inputs, raw_value)

    output = layer._apply_output_mode(inputs, raw)
    logit_shift = torch.logit(output) - torch.logit(inputs)

    assert torch.all(output > 0)
    assert torch.all(output < 1)
    assert logit_shift.abs().max().item() <= 1.50001


def test_bounded_logit_residual_keeps_finite_nonzero_gradients() -> None:
    layer = CmKANLayer(
        in_channels=3,
        out_channels=3,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
        output_mode="bounded_logit_residual",
        max_logit_shift=1.5,
    )
    inputs = torch.tensor([0.20, 0.50, 0.80]).view(1, 3, 1, 1)
    raw = torch.zeros_like(inputs, requires_grad=True)

    layer._apply_output_mode(inputs, raw).sum().backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad != 0)


def test_bounded_output_head_can_be_reset_to_identity_initialization() -> None:
    layer = CmKANLayer(
        in_channels=3,
        out_channels=3,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
        output_mode="bounded_logit_residual",
    )
    torch.nn.init.normal_(layer.generator.conv_reproj.pointwise2.weight)
    torch.nn.init.normal_(layer.generator.conv_reproj.pointwise2.bias)
    coefficient_end = int(layer.kan_params_indices[1])
    coefficient_weights = (
        layer.generator.conv_reproj.pointwise2.weight[:coefficient_end]
        .detach()
        .clone()
    )
    coefficient_bias = (
        layer.generator.conv_reproj.pointwise2.bias[:coefficient_end]
        .detach()
        .clone()
    )

    layer.reset_bounded_output_head()

    output_head = layer.generator.conv_reproj.pointwise2
    assert torch.equal(
        output_head.weight[:coefficient_end],
        coefficient_weights,
    )
    assert torch.equal(
        output_head.bias[:coefficient_end],
        coefficient_bias,
    )
    assert torch.count_nonzero(output_head.weight[coefficient_end:]) == 0
    assert torch.count_nonzero(output_head.bias[coefficient_end:]) == 0

    inputs = torch.rand(1, 3, 16, 16) * 0.8 + 0.1
    outputs = layer(inputs)
    assert torch.allclose(outputs, inputs, atol=1e-5, rtol=0)

    outputs.mean().backward()
    univariate_end = int(layer.kan_params_indices[2])
    assert output_head.weight.grad is not None
    assert (
        output_head.weight.grad[
            coefficient_end:univariate_end
        ].abs().sum().item()
        > 0
    )


def test_legacy_output_mode_keeps_raw_kan_output_unchanged() -> None:
    layer = CmKANLayer(
        in_channels=3,
        out_channels=3,
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )
    inputs = torch.rand(1, 3, 2, 2)
    raw_output = torch.randn_like(inputs)

    assert layer._apply_output_mode(inputs, raw_output) is raw_output


def test_reference_model_propagates_bounded_output_configuration() -> None:
    model = _reference_model(
        output_mode="bounded_logit_residual",
        max_logit_shift=1.5,
    )

    for generator in (model.gen_ab, model.gen_ba):
        for layer in generator.layers:
            assert layer.output_mode == "bounded_logit_residual"
            assert layer.max_logit_shift == pytest.approx(1.5)
