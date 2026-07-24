import torch
from torch import nn


class ReferenceStyleEncoder(nn.Module):
    """Extract differentiable global color and white-balance statistics."""

    output_dim = 10

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "rgb_to_xyz",
            torch.tensor(
                [
                    [0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041],
                ],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _linearize_srgb(images: torch.Tensor) -> torch.Tensor:
        images = images.clamp(0, 1)
        return torch.where(
            images <= 0.04045,
            images / 12.92,
            ((images + 0.055) / 1.055).pow(2.4),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Reference images must have shape (batch, 3, height, width), "
                f"got {images.shape}"
            )

        linear_rgb = self._linearize_srgb(images)
        reduce_dims = (2, 3)
        rgb_mean = linear_rgb.mean(dim=reduce_dims)
        rgb_std = linear_rgb.std(dim=reduce_dims, unbiased=False)

        xyz = torch.einsum(
            "ij,bjhw->bihw",
            self.rgb_to_xyz.to(dtype=linear_rgb.dtype),
            linear_rgb,
        )
        xyz_mean = xyz.mean(dim=reduce_dims)
        xyz_total = xyz_mean.sum(dim=1, keepdim=True).clamp_min(1e-6)
        chromaticity_xy = xyz_mean[:, :2] / xyz_total
        luminance = xyz[:, 1:2]
        luminance_mean = luminance.mean(dim=reduce_dims)
        luminance_std = luminance.std(
            dim=reduce_dims,
            unbiased=False,
        )

        return torch.cat(
            [
                rgb_mean,
                rgb_std,
                chromaticity_xy,
                luminance_mean,
                luminance_std,
            ],
            dim=1,
        )
