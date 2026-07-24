import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KANLayer
from .generator import GeneratorLayer, LightGeneratorLayer


class CmKANLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range, condition_dim=0,
                 output_mode='legacy', max_logit_shift=1.5,
                 direct_conditioning=False):
        super(CmKANLayer, self).__init__()

        self.output_mode = getattr(output_mode, 'value', output_mode)
        if self.output_mode not in {'legacy', 'bounded_logit_residual'}:
            raise ValueError(f'Unsupported cmKAN output mode: {self.output_mode}')
        if self.output_mode == 'bounded_logit_residual':
            if in_channels != out_channels:
                raise ValueError(
                    "bounded_logit_residual requires matching input and output "
                    f"channels, got {in_channels} and {out_channels}"
                )
            if max_logit_shift <= 0:
                raise ValueError("max_logit_shift must be greater than zero")
        self.max_logit_shift = float(max_logit_shift)

        self.kan_layer = KANLayer(in_dim=in_channels,
                                  out_dim=out_channels,
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  residual_std=residual_std,
                                  grid_range=grid_range)

        # Arbitrary layers configuration fc
        self.kan_params_num = 0
        self.kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape)
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape)
        self.kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len])

        self.kan_params_num = np.sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)

        self.generator = GeneratorLayer(
            in_channels,
            self.kan_params_num,
            condition_dim=condition_dim,
            direct_conditioning=direct_conditioning,
        )

    def kan(self, x, w):

        i, j = self.kan_params_indices[0], self.kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self.kan_params_indices[1], self.kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape)
        i, j = self.kan_params_indices[2], self.kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape)
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)

        return x.squeeze(0)

    def _apply_output_mode(self, inputs, raw_output):
        """Map raw KAN output to the configured image representation."""
        if self.output_mode == 'legacy':
            return raw_output

        # Predict a bounded shift around the input in logit space. Zero raw
        # output is therefore the identity mapping, while extreme KAN weights
        # can never create values that save_image would clip into color blobs.
        dtype_epsilon = torch.finfo(inputs.dtype).eps
        epsilon = max(1e-4, dtype_epsilon)
        base = torch.logit(inputs.clamp(epsilon, 1 - epsilon))
        shift = self.max_logit_shift * torch.tanh(raw_output)
        return torch.sigmoid(base + shift)

    def reset_bounded_output_head(self):
        """Start bounded residual translation at the identity mapping."""
        if self.output_mode != 'bounded_logit_residual':
            return
        output_head = self.generator.conv_reproj.pointwise2
        coefficient_end = int(self.kan_params_indices[1])

        # Keep random spline coefficients, but zero their multiplicative
        # weights and the residual weights. The raw KAN output is still zero,
        # while the nonzero coefficients let the univariate-weight rows receive
        # gradients on the first update instead of deadlocking both factors at
        # zero forever.
        nn.init.zeros_(output_head.weight[coefficient_end:])
        if output_head.bias is not None:
            nn.init.zeros_(output_head.bias[coefficient_end:])

    def _apply_spatial_kan(self, x, weights):
        """Apply predicted per-pixel KAN parameters to an input image."""
        inputs = x
        B, C, H, W = x.shape

        # kan weights (b, h * w, kan_params_num)
        weights = weights.permute(0, 2, 3, 1)
        weights = weights.reshape(B * H * W, self.kan_params_num)

        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # img (b * h * w, 3), weights (b * h * w, kan_params_num)
        x = self.kan(x, weights)

        x = x.view(B, H, W, self.kan_layer.out_dim).permute(0, 3, 1, 2)

        return self._apply_output_mode(inputs, x)

    def encode(self, x, condition=None):
        """Expose the contextual features that produce spatial KAN weights."""
        return self.generator.encode(x, condition)

    def forward_with_features(self, x, condition=None):
        """Return the translated image and its input contextual features."""
        weights, features = self.generator.forward_with_features(x, condition)
        return self._apply_spatial_kan(x, weights), features

    def forward(self, x, condition=None):
        output, _ = self.forward_with_features(x, condition)
        return output
    

class LightCmKANLayer(CmKANLayer):
    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range, condition_dim=0,
                 output_mode='legacy', max_logit_shift=1.5,
                 direct_conditioning=False):
        super(LightCmKANLayer, self).__init__(in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range, condition_dim=condition_dim,
                 output_mode=output_mode, max_logit_shift=max_logit_shift,
                 direct_conditioning=direct_conditioning)
        self.generator = LightGeneratorLayer(
            in_channels,
            self.kan_params_num,
            condition_dim=condition_dim,
            direct_conditioning=direct_conditioning,
        )
