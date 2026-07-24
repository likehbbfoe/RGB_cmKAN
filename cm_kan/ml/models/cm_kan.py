import torch
import torch.nn as nn
from cm_kan.ml.layers import CmKANLayer, LightCmKANLayer
from cm_kan.core import Logger


class CmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_dims,
        out_dims,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
        condition_dim=0,
        output_mode='legacy',
        max_logit_shift=1.5,
        direct_conditioning=False,
    ):
        super(CmKAN, self).__init__()

        Logger.info(f"CmKAN: in_dims={in_dims}, out_dims={out_dims}")

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                CmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range,
                         condition_dim=condition_dim,
                         output_mode=output_mode,
                         max_logit_shift=max_logit_shift,
                         direct_conditioning=direct_conditioning))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, condition=None):
        for layer in self.layers:
            x = layer(x, condition)
        return x

    def forward_with_features(self, x, condition=None):
        """Translate an image and return contextual features from every layer."""
        features = []
        for layer in self.layers:
            x, layer_features = layer.forward_with_features(x, condition)
            features.append(layer_features)
        return x, features

    def encode_features(self, x, condition=None):
        """Encode content without computing the final unused layer output."""
        features = []
        for index, layer in enumerate(self.layers):
            features.append(layer.encode(x, condition))
            if index + 1 < len(self.layers):
                x = layer(x, condition)
        return features


class LightCmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_dims,
        out_dims,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
        condition_dim=0,
        output_mode='legacy',
        max_logit_shift=1.5,
        direct_conditioning=False,
    ):
        super(LightCmKAN, self).__init__()

        Logger.info(f"LightCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                LightCmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range,
                         condition_dim=condition_dim,
                         output_mode=output_mode,
                         max_logit_shift=max_logit_shift,
                         direct_conditioning=direct_conditioning))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, condition=None):
        for layer in self.layers:
            x = layer(x, condition)
        return x

    def forward_with_features(self, x, condition=None):
        """Translate an image and return contextual features from every layer."""
        features = []
        for layer in self.layers:
            x, layer_features = layer.forward_with_features(x, condition)
            features.append(layer_features)
        return x, features

    def encode_features(self, x, condition=None):
        """Encode content without computing the final unused layer output."""
        features = []
        for index, layer in enumerate(self.layers):
            features.append(layer.encode(x, condition))
            if index + 1 < len(self.layers):
                x = layer(x, condition)
        return features
