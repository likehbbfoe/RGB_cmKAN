import torch
import torch.nn as nn
from cm_kan.core import Logger
from .cm_kan import CmKAN, LightCmKAN
from .reference_style import ReferenceStyleEncoder
from ..layers import PatchDiscriminator


class CycleCmKAN(torch.nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
        condition_dim=0,
    ):
        super(CycleCmKAN, self).__init__()

        Logger.info(f"CycleCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        self.gen_ab = CmKAN(
            in_dims=in_dims,
            out_dims=out_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
            condition_dim=condition_dim,
        )
        self.gen_ba = CmKAN(
            in_dims=out_dims,
            out_dims=in_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
            condition_dim=condition_dim,
        )
        self.dis_a = PatchDiscriminator(in_dim=in_dims[0])
        self.dis_b = PatchDiscriminator(in_dim=out_dims[0])


class ReferenceCycleCmKAN(CycleCmKAN):
    """Cycle cmKAN whose spatial KAN weights follow a reference image style."""

    reference_guided = True

    def __init__(
        self,
        in_dims,
        out_dims,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
    ):
        super().__init__(
            in_dims=in_dims,
            out_dims=out_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
            condition_dim=ReferenceStyleEncoder.output_dim,
        )
        self.style_encoder = ReferenceStyleEncoder()

    def encode_style(self, reference: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(reference)

    def style_condition(
        self,
        inputs: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Describe the per-image color change requested by the reference."""
        return self.encode_style(reference) - self.encode_style(inputs)
