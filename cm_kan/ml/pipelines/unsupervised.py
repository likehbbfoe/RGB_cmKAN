import itertools
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from torch import optim
from ..models import CycleCmKAN
from ..utils.image_pool import ImagePool
from cm_kan.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE,
)


class UnsupervisedPipeline(L.LightningModule):
    def __init__(self,
        model: CycleCmKAN,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
        pretrained: bool = False,
        pretrained_model: str = None,
        training_mode: str = 'pretrain',
        reverse_prediction: bool = False,
        adversarial_weight: float = 1.0,
        adversarial_ramp_epochs: int = 0,
        cycle_weight: float = 10.0,
        identity_weight: float = 5.0,
        domain_statistics_weight: float = 0.0,
        exposure_weight: float = 0.0,
        chroma_weight: float = 0.0,
        reflectance_weight: float = 0.0,
        patch_nce_weight: float = 0.0,
        patch_nce_num_patches: int = 256,
        patch_nce_temperature: float = 0.07,
        reference_style_weight: float = 0.0,
        reference_white_balance_weight: float = 0.0,
        reference_white_balance_ramp_epochs: int = 0,
        reference_local_chroma_weight: float = 0.0,
        reference_local_chroma_tail_weight: float = 0.0,
        reference_local_chroma_tail_fraction: float = 0.05,
        reference_local_chroma_threshold: float = 0.25,
        reference_local_red_tail_weight: float = 0.0,
        reference_local_red_tail_fraction: float = 0.02,
        reference_local_red_threshold: float = 0.1823215568,
        reference_red_overshoot_weight: float = 0.0,
        reference_red_overshoot_margin: float = 0.02,
        range_weight: float = 0.0,
        range_tail_weight: float = 0.0,
        range_tail_fraction: float = 0.01,
        warmup_epochs: int = 0,
        gradient_clip_val: float = 0.0,
        discriminator_lr_scale: float = 1.0,
    ) -> None:
        super(UnsupervisedPipeline, self).__init__()

        self.model = model
        self.fake_pool_a = ImagePool()
        self.fake_pool_b = ImagePool()
        self.adversarial_weight = adversarial_weight
        self.adversarial_ramp_epochs = adversarial_ramp_epochs
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.domain_statistics_weight = domain_statistics_weight
        self.exposure_weight = exposure_weight
        self.chroma_weight = chroma_weight
        self.reflectance_weight = reflectance_weight
        self.patch_nce_weight = patch_nce_weight
        self.patch_nce_num_patches = patch_nce_num_patches
        self.patch_nce_temperature = patch_nce_temperature
        self.reference_style_weight = reference_style_weight
        self.reference_white_balance_weight = reference_white_balance_weight
        self.reference_white_balance_ramp_epochs = (
            reference_white_balance_ramp_epochs
        )
        self.reference_local_chroma_weight = reference_local_chroma_weight
        self.reference_local_chroma_tail_weight = (
            reference_local_chroma_tail_weight
        )
        self.reference_local_chroma_tail_fraction = (
            reference_local_chroma_tail_fraction
        )
        self.reference_local_chroma_threshold = (
            reference_local_chroma_threshold
        )
        self.reference_local_red_tail_weight = reference_local_red_tail_weight
        self.reference_local_red_tail_fraction = (
            reference_local_red_tail_fraction
        )
        self.reference_local_red_threshold = reference_local_red_threshold
        self.reference_red_overshoot_weight = reference_red_overshoot_weight
        self.reference_red_overshoot_margin = reference_red_overshoot_margin
        self.range_weight = range_weight
        self.range_tail_weight = range_tail_weight
        self.range_tail_fraction = range_tail_fraction
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        self.discriminator_lr_scale = discriminator_lr_scale
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model
        normalized_training_mode = getattr(training_mode, "value", training_mode)
        self.adversarial = normalized_training_mode == 'adversarial' or pretrained
        if self.adversarial:
            self.automatic_optimization = False
        if self.pretrained and not self.pretrained_model:
            raise ValueError("pretrained_model is required when pretrained is true")
        self.reverse_prediction = reverse_prediction
        self.reference_guided = getattr(model, "reference_guided", False)
        if self.reference_guided and not self.adversarial:
            raise ValueError(
                "reference_cycle_cm_kan requires training_mode='adversarial'"
            )

        self.save_hyperparameters(ignore=['model', 'reverse_prediction'])

    def _identity_loss(self, predictions, targets):
        mae_loss = self.mae_loss(predictions, targets)
        return mae_loss

    def _style_condition(self, inputs, reference):
        if not self.reference_guided:
            return None
        if reference is None:
            raise ValueError(
                "reference_cycle_cm_kan requires a target reference image"
            )
        return self.model.style_condition(inputs, reference)

    def _reference_style_loss(self, predictions, reference):
        if not self.reference_guided:
            return predictions.new_zeros(())
        predicted_style = self.model.encode_style(predictions)
        reference_style = self.model.encode_style(reference).detach()
        return F.l1_loss(predicted_style, reference_style)

    def _reference_style_distances(self, inputs, predictions, reference):
        """Return per-image input/reference and prediction/reference distances."""
        if not self.reference_guided:
            zero = predictions.new_zeros(())
            return zero, zero, zero
        reference_style = self.model.encode_style(reference).detach()
        input_distance = (
            self.model.encode_style(inputs).detach() - reference_style
        ).abs().mean(dim=1)
        prediction_distance = (
            self.model.encode_style(predictions) - reference_style
        ).abs().mean(dim=1)
        ratio = prediction_distance / input_distance.clamp_min(1e-6)
        return (
            input_distance.mean(),
            prediction_distance.mean(),
            ratio.mean(),
        )

    @staticmethod
    def _linearize_srgb(images):
        images = images.clamp(0, 1)
        return torch.where(
            images <= 0.04045,
            images / 12.92,
            ((images + 0.055) / 1.055).pow(2.4),
        )

    @classmethod
    def _white_balance_statistics(
        cls,
        images,
        eps=1e-3,
        sigma=0.35,
        neutral_sigma=0.30,
    ):
        """Estimate color cast from midtone pixels that are likely neutral."""
        srgb = images.float().clamp(0, 1)
        linear_rgb = cls._linearize_srgb(srgb)
        red, green, blue = linear_rgb.unbind(dim=1)
        log_red_green = torch.log(red + eps) - torch.log(green + eps)
        log_blue_green = torch.log(blue + eps) - torch.log(green + eps)
        log_chroma = torch.stack(
            [log_red_green, log_blue_green],
            dim=1,
        )

        luminance = cls._luminance(linear_rgb)
        valid_weight = (
            torch.sigmoid((luminance - 0.01) / 0.01)
            * torch.sigmoid((0.98 - luminance) / 0.02)
        ).unsqueeze(1)
        channel_max = srgb.amax(dim=1, keepdim=True)
        channel_min = srgb.amin(dim=1, keepdim=True)
        saturation = (
            (channel_max - channel_min) / channel_max.clamp_min(eps)
        )
        neutral_weight = 0.02 + 0.98 * torch.exp(
            -0.5 * (saturation / neutral_sigma).square()
        )
        selection_weight = (valid_weight * neutral_weight).detach()
        reduce_dims = (2, 3)
        initial_center = (
            (selection_weight * log_chroma).sum(dim=reduce_dims, keepdim=True)
            / selection_weight.sum(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        )
        squared_distance = (
            (log_chroma - initial_center.detach()).square().sum(
                dim=1,
                keepdim=True,
            )
        )
        robust_weight = 0.05 + 0.95 * torch.exp(
            -squared_distance / (2 * sigma * sigma)
        )

        # Pixel selection is not a prediction target. Detaching prevents the
        # generator from lowering the loss by manipulating its own mask.
        weights = (selection_weight * robust_weight).detach()
        return (
            (weights * log_chroma).sum(dim=reduce_dims)
            / weights.sum(dim=reduce_dims).clamp_min(1e-6)
        )

    @staticmethod
    def _charbonnier(values, delta=0.01):
        return torch.sqrt(values.square() + delta * delta) - delta

    def _reference_white_balance_loss(self, predictions, reference):
        if not self.reference_guided:
            return predictions.new_zeros(())
        deltas = self._white_balance_statistics(predictions) - (
            self._white_balance_statistics(reference).detach()
        )
        warm_delta = 0.5 * (deltas[:, 0] - deltas[:, 1])
        tint_delta = 0.5 * (deltas[:, 0] + deltas[:, 1])
        return (
            self._charbonnier(warm_delta)
            + 0.5 * self._charbonnier(tint_delta)
        ).mean()

    def _reference_white_balance_deltas(self, predictions, reference):
        if not self.reference_guided:
            zero = predictions.new_zeros(())
            return zero, zero, zero, zero, zero, zero, zero
        deltas = self._white_balance_statistics(predictions) - (
            self._white_balance_statistics(reference).detach()
        )
        warm_deltas = 0.5 * (deltas[:, 0] - deltas[:, 1])
        tint_deltas = 0.5 * (deltas[:, 0] + deltas[:, 1])
        return (
            deltas[:, 0].mean(),
            deltas[:, 1].mean(),
            warm_deltas.mean(),
            warm_deltas.abs().mean(),
            (warm_deltas > 0).float().mean(),
            tint_deltas.mean(),
            tint_deltas.abs().mean(),
        )

    @staticmethod
    def _ramped_weight(weight, ramp_epochs, current_epoch):
        if ramp_epochs <= 0:
            return weight
        ramp_epoch = int(current_epoch) + 1
        progress = min(
            1.0,
            ramp_epoch / ramp_epochs,
        )
        return weight * progress

    def _effective_reference_white_balance_weight(self):
        return self._ramped_weight(
            self.reference_white_balance_weight,
            self.reference_white_balance_ramp_epochs,
            self.current_epoch,
        )

    @staticmethod
    def _ramped_adversarial_weight(
        weight,
        warmup_epochs,
        ramp_epochs,
        current_epoch,
    ):
        """Delay GAN pressure, then introduce it gradually after warmup."""
        if current_epoch < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return weight
        adversarial_epoch = current_epoch - warmup_epochs + 1
        progress = min(1.0, adversarial_epoch / ramp_epochs)
        return weight * progress

    def _effective_adversarial_weight(self):
        return self._ramped_adversarial_weight(
            self.adversarial_weight,
            self.warmup_epochs,
            self.adversarial_ramp_epochs,
            self.current_epoch,
        )

    def _cycle_loss(self, predictions, targets):
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15
        return loss

    def _disc_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)

    @staticmethod
    def _domain_statistics_loss(predictions, targets):
        """Match differentiable per-channel brightness and contrast moments."""
        reduce_dims = (0, 2, 3)
        prediction_mean = predictions.mean(dim=reduce_dims)
        target_mean = targets.mean(dim=reduce_dims)
        prediction_std = predictions.std(dim=reduce_dims, unbiased=False)
        target_std = targets.std(dim=reduce_dims, unbiased=False)
        return F.l1_loss(prediction_mean, target_mean) + F.l1_loss(
            prediction_std, target_std
        )

    @staticmethod
    def _top_fraction_mean(values, fraction):
        """Return a per-image CVaR over the largest spatial values."""
        if not 0 < fraction <= 1:
            raise ValueError("tail fraction must be in the interval (0, 1]")
        flattened = values.reshape(values.shape[0], -1)
        count = max(1, math.ceil(flattened.shape[1] * fraction))
        return flattened.topk(count, dim=1).values.mean(dim=1).mean()

    @classmethod
    def _range_terms(cls, predictions, tail_fraction=0.01):
        """Measure both average and sparse values clipped during image saving."""
        channel_overshoot = torch.maximum(
            F.relu(-predictions),
            F.relu(predictions - 1),
        )
        pixel_overshoot = channel_overshoot.amax(dim=1, keepdim=True)
        return {
            'mean': channel_overshoot.mean(),
            'tail': cls._top_fraction_mean(pixel_overshoot, tail_fraction),
            'out_of_range_fraction': (
                ((predictions < 0) | (predictions > 1))
                .any(dim=1, keepdim=True)
                .float()
                .mean()
            ),
        }

    @classmethod
    def _range_loss(cls, predictions):
        """Backward-compatible mean range loss."""
        return cls._range_terms(predictions)['mean']

    @classmethod
    def _exposure_loss(cls, predictions, inputs):
        """Preserve per-image luminance mean and contrast across translation."""
        prediction_luma = cls._luminance(predictions)
        input_luma = cls._luminance(inputs)
        reduce_dims = (1, 2)
        prediction_mean = prediction_luma.mean(dim=reduce_dims)
        input_mean = input_luma.mean(dim=reduce_dims)
        prediction_std = prediction_luma.std(dim=reduce_dims, unbiased=False)
        input_std = input_luma.std(dim=reduce_dims, unbiased=False)
        return F.l1_loss(prediction_mean, input_mean) + F.l1_loss(
            prediction_std, input_std
        )

    @staticmethod
    def _chromaticity(images, eps=1e-4):
        """Return intensity-invariant RGB ratios for skin/color preservation."""
        non_negative_images = images.clamp_min(0)
        intensity = non_negative_images.sum(dim=1, keepdim=True).clamp_min(eps)
        return non_negative_images / intensity

    @classmethod
    def _chroma_loss(cls, predictions, inputs):
        """Preserve hue while allowing a multiplicative illumination change."""
        return F.l1_loss(
            cls._chromaticity(predictions),
            cls._chromaticity(inputs),
        )

    @classmethod
    def _local_chroma_terms(
        cls,
        predictions,
        inputs,
        chroma_tail_fraction=0.05,
        red_tail_fraction=0.02,
        chroma_threshold=0.25,
        red_threshold=0.1823215568,
        eps=1e-3,
    ):
        """Measure local color drift after removing a robust global color gain."""
        prediction_rgb = cls._linearize_srgb(predictions.float())
        input_rgb = cls._linearize_srgb(inputs.float())

        def log_chroma(images):
            red, green, blue = images.unbind(dim=1)
            return torch.stack(
                [
                    torch.log(red + eps) - torch.log(green + eps),
                    torch.log(blue + eps) - torch.log(green + eps),
                ],
                dim=1,
            )

        chroma_delta = log_chroma(prediction_rgb) - log_chroma(input_rgb).detach()
        input_luminance = cls._luminance(input_rgb)
        weights = (
            torch.sigmoid((input_luminance - 0.01) / 0.01)
            * torch.sigmoid((0.98 - input_luminance) / 0.02)
        ).unsqueeze(1).detach()
        reduce_dims = (2, 3)
        initial_global_delta = (
            (weights * chroma_delta).sum(dim=reduce_dims, keepdim=True)
            / weights.sum(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        )
        # Keep the original whole-image mean term exactly compatible with v2.
        # A second robust center is used only by the new worst-region guards so
        # a small red patch cannot drag the center toward itself.
        mean_local_residual = chroma_delta - initial_global_delta
        per_pixel_loss = cls._charbonnier(mean_local_residual).mean(
            dim=1,
            keepdim=True,
        )
        per_image_loss = (
            (weights * per_pixel_loss).sum(dim=reduce_dims)
            / weights.sum(dim=reduce_dims).clamp_min(1e-6)
        )
        squared_distance = (
            (chroma_delta - initial_global_delta.detach())
            .square()
            .sum(dim=1, keepdim=True)
        )
        robust_weights = (
            1 / (1 + squared_distance / (0.25 * 0.25))
        ).detach()
        center_weights = weights * robust_weights
        global_delta = (
            (center_weights * chroma_delta).sum(
                dim=reduce_dims,
                keepdim=True,
            )
            / center_weights.sum(
                dim=reduce_dims,
                keepdim=True,
            ).clamp_min(1e-6)
        )
        local_residual = chroma_delta - global_delta
        residual_magnitude = torch.sqrt(
            local_residual.square().sum(dim=1, keepdim=True) + 1e-8
        )
        softness = 0.03
        chroma_excess = softness * F.softplus(
            (residual_magnitude - chroma_threshold) / softness
        )
        red_excess = softness * F.softplus(
            (local_residual[:, :1] - red_threshold) / softness
        )
        weight_sum = weights.sum(dim=reduce_dims).clamp_min(1e-6)
        return {
            'mean': per_image_loss.mean(),
            'chroma_tail': cls._top_fraction_mean(
                weights * chroma_excess,
                chroma_tail_fraction,
            ),
            'red_tail': cls._top_fraction_mean(
                weights * red_excess,
                red_tail_fraction,
            ),
            'chroma_bad_fraction': (
                (
                    weights
                    * (residual_magnitude > chroma_threshold).float()
                ).sum(dim=reduce_dims)
                / weight_sum
            ).mean(),
            'red_bad_fraction': (
                (
                    weights
                    * (local_residual[:, :1] > red_threshold).float()
                ).sum(dim=reduce_dims)
                / weight_sum
            ).mean(),
        }

    @classmethod
    def _local_chroma_loss(cls, predictions, inputs, eps=1e-3):
        """Backward-compatible mean local chroma loss."""
        return cls._local_chroma_terms(
            predictions,
            inputs,
            eps=eps,
        )['mean']

    def _reference_red_overshoot_loss(self, predictions, reference):
        """Penalize only a global R/G cast that is redder than its reference."""
        if not self.reference_guided:
            return predictions.new_zeros(())
        red_green_delta = (
            self._white_balance_statistics(predictions)
            - self._white_balance_statistics(reference).detach()
        )[:, 0]
        excess = F.relu(
            red_green_delta - self.reference_red_overshoot_margin
        )
        return self._charbonnier(excess).mean()

    @classmethod
    def _reflectance(cls, images, kernel_size=31, eps=1e-4):
        """Estimate log-domain detail after removing smooth illumination."""
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if min(images.shape[-2:]) <= kernel_size // 2:
            raise ValueError(
                "image height and width must be larger than half the reflectance "
                f"kernel size; got {images.shape[-2:]} and {kernel_size=}"
            )

        log_luminance = cls._luminance(images).clamp_min(eps).log().unsqueeze(1)
        padding = kernel_size // 2
        smooth_log_luminance = F.avg_pool2d(
            F.pad(
                log_luminance,
                (padding, padding, padding, padding),
                mode='reflect',
            ),
            kernel_size=kernel_size,
            stride=1,
        )
        return log_luminance - smooth_log_luminance

    @classmethod
    def _reflectance_loss(cls, predictions, inputs):
        """Keep local intrinsic contrast while permitting smooth relighting."""
        return F.l1_loss(cls._reflectance(predictions), cls._reflectance(inputs))

    @staticmethod
    def _patch_nce_loss(
        query_features,
        key_features,
        num_patches,
        temperature,
        random_sampling=True,
    ):
        """Contrast matching spatial patches against negatives in each image."""
        if len(query_features) != len(key_features):
            raise ValueError("query_features and key_features must have equal length")
        if not query_features:
            raise ValueError("PatchNCE requires at least one feature map")

        layer_losses = []
        for query, key in zip(query_features, key_features):
            if query.shape != key.shape:
                raise ValueError(
                    "PatchNCE feature shapes must match; "
                    f"got {query.shape} and {key.shape}"
                )

            batch_size, channels, height, width = query.shape
            available_patches = height * width
            sampled_patches = min(num_patches, available_patches)
            if random_sampling:
                patch_ids = torch.randperm(
                    available_patches, device=query.device
                )[:sampled_patches]
            else:
                patch_ids = torch.linspace(
                    0,
                    available_patches - 1,
                    steps=sampled_patches,
                    device=query.device,
                ).long()

            query_patches = query.flatten(2).transpose(1, 2)[:, patch_ids]
            key_patches = key.detach().flatten(2).transpose(1, 2)[:, patch_ids]
            query_patches = F.normalize(query_patches, dim=-1)
            key_patches = F.normalize(key_patches, dim=-1)

            logits = torch.bmm(
                query_patches, key_patches.transpose(1, 2)
            ) / temperature
            labels = torch.arange(
                sampled_patches, device=query.device
            ).expand(batch_size, sampled_patches)
            layer_losses.append(
                F.cross_entropy(
                    logits.reshape(-1, sampled_patches),
                    labels.reshape(-1),
                )
            )

        return torch.stack(layer_losses).mean()

    @staticmethod
    def _luminance(images):
        weights = images.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        return (images * weights).sum(dim=1)

    @classmethod
    def _luminance_mean(cls, images):
        return cls._luminance(images).mean()

    def _log_loss(self, name, value, batch_size, prog_bar=False):
        self.log(
            name,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch_size,
        )

    def _clip_optimizer_gradients(self, optimizer):
        if self.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
    
    @staticmethod
    def _set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            if self.reference_guided:
                for module in self.model.modules():
                    style_affine = getattr(module, "style_affine", None)
                    if style_affine is not None:
                        nn.init.zeros_(style_affine[-1].weight)
                        nn.init.zeros_(style_affine[-1].bias)

            # The bounded residual mode is intentionally initialized as an
            # identity transform after the generic Kaiming pass above.
            for module in self.model.modules():
                reset_output_head = getattr(
                    module,
                    "reset_bounded_output_head",
                    None,
                )
                if reset_output_head is not None:
                    reset_output_head()

            if self.pretrained:
                pipeline = UnsupervisedPipeline.load_from_checkpoint(
                    self.pretrained_model,
                    model=self.model,
                    optimiser=self.optimizer_type,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    pretrained=False,
                    training_mode='pretrain',
                )
                self.model.gen_ab = pipeline.model.gen_ab
                self.model.gen_ba = pipeline.model.gen_ba
                del pipeline
                Logger.info(f'Initialized model weights {self.pretrained_model}.')
        
        Logger.info('Initialized model weights with [bold green]Unsupervised[/bold green] pipeline.')
        if self.adversarial:
            Logger.info('Model is in [bold green]CycleGAN training[/bold green] mode.')
        else:
            Logger.info('Model is in [bold green]Generator pre-training[/bold green] mode.')

        if self.reverse_prediction:
            Logger.info('Model is in [bold green]Reversed prediction (b -> a)[/bold green] mode.')
        elif not self.reverse_prediction:
            Logger.info('Model is in [bold green]Normal prediction (a -> b)[/bold green] mode.')

    def configure_optimizers(self):
        if not self.adversarial:
            if self.optimizer_type == 'adam':
                optimizer = optim.Adam(itertools.chain(self.model.gen_ab.parameters(), self.model.gen_ba.parameters()),
                            lr=self.lr, weight_decay=self.weight_decay)
            else:
                raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=500, T_mult=1, eta_min=1e-5
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            if self.optimizer_type == 'adam':
                optG = optim.Adam(
                    itertools.chain(self.model.gen_ab.parameters(), self.model.gen_ba.parameters()),
                    lr=self.lr,
                    betas=(0.5, 0.999),
                    weight_decay=self.weight_decay,
                )

                optD = optim.Adam(
                    itertools.chain(self.model.dis_a.parameters(), self.model.dis_b.parameters()),
                    lr=self.lr * self.discriminator_lr_scale,
                    betas=(0.5, 0.999),
                    weight_decay=self.weight_decay,
                )
            else:
                raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
            gamma = lambda epoch: max(0.0, 1 - max(0, epoch + 1 - 100) / 101)
            schG = optim.lr_scheduler.LambdaLR(optG, lr_lambda=gamma)
            schD = optim.lr_scheduler.LambdaLR(optD, lr_lambda=gamma)
            return [optG, optD], [schG, schD]

    def forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._style_condition(x, reference)
        pred = self.model.gen_ab(x, condition)
        return pred
    
    def reversed_forward(
        self,
        x: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self._style_condition(x, reference)
        pred = self.model.gen_ba(x, condition)
        return pred
    
    def generator_training_step(
        self,
        imgA,
        imgB,
        adversarial_weight=None,
    ):
        """cycle images - using only generator nets"""
        effectiveAdversarialWeight = (
            self._effective_adversarial_weight()
            if adversarial_weight is None
            else adversarial_weight
        )
        conditionB = self._style_condition(imgA, imgB)
        conditionA = self._style_condition(imgB, imgA)
        if self.patch_nce_weight > 0:
            fakeB, source_features = self.model.gen_ab.forward_with_features(
                imgA, conditionB
            )
            fakeA, target_features = self.model.gen_ba.forward_with_features(
                imgB, conditionA
            )
            fakeB_features = self.model.gen_ab.encode_features(fakeB, conditionB)
            fakeA_features = self.model.gen_ba.encode_features(fakeA, conditionA)
            patchNceB = self._patch_nce_loss(
                fakeB_features,
                source_features,
                self.patch_nce_num_patches,
                self.patch_nce_temperature,
            )
            patchNceA = self._patch_nce_loss(
                fakeA_features,
                target_features,
                self.patch_nce_num_patches,
                self.patch_nce_temperature,
            )
        else:
            fakeB = self.model.gen_ab(imgA, conditionB)
            fakeA = self.model.gen_ba(imgB, conditionA)
            patchNceA = fakeA.new_zeros(())
            patchNceB = fakeB.new_zeros(())
        patchNceLoss = patchNceA + patchNceB

        cycledA = self.model.gen_ba(
            fakeB,
            self._style_condition(fakeB, imgA),
        )
        cycledB = self.model.gen_ab(
            fakeA,
            self._style_condition(fakeA, imgB),
        )
        
        sameB = self.model.gen_ab(imgB, self._style_condition(imgB, imgB))
        sameA = self.model.gen_ba(imgA, self._style_condition(imgA, imgA))
        
        if effectiveAdversarialWeight > 0:
            # gen_ab/gen_ba must fool their destination discriminators.
            predFakeB = self.model.dis_b(fakeB)
            adversarialB = self._disc_loss(predFakeB, 'real')
            predFakeA = self.model.dis_a(fakeA)
            adversarialA = self._disc_loss(predFakeA, 'real')
        else:
            adversarialA = fakeA.new_zeros(())
            adversarialB = fakeB.new_zeros(())
        
        # compute extra losses
        identityA = self._identity_loss(sameA, imgA)
        identityB = self._identity_loss(sameB, imgB)
        identityLoss = identityA + identityB
        
        # compute cycleLosses
        cycleA = self._cycle_loss(cycledA, imgA)
        cycleB = self._cycle_loss(cycledB, imgB)
        cycleLoss = cycleA + cycleB

        statisticsA = self._domain_statistics_loss(fakeA, imgA)
        statisticsB = self._domain_statistics_loss(fakeB, imgB)
        statisticsLoss = statisticsA + statisticsB

        exposureA = self._exposure_loss(fakeA, imgB)
        exposureB = self._exposure_loss(fakeB, imgA)
        exposureLoss = exposureA + exposureB

        if self.chroma_weight > 0:
            chromaA = self._chroma_loss(fakeA, imgB)
            chromaB = self._chroma_loss(fakeB, imgA)
        else:
            chromaA = fakeA.new_zeros(())
            chromaB = fakeB.new_zeros(())
        chromaLoss = chromaA + chromaB

        if self.reflectance_weight > 0:
            reflectanceA = self._reflectance_loss(fakeA, imgB)
            reflectanceB = self._reflectance_loss(fakeB, imgA)
        else:
            reflectanceA = fakeA.new_zeros(())
            reflectanceB = fakeB.new_zeros(())
        reflectanceLoss = reflectanceA + reflectanceB

        referenceStyleA = self._reference_style_loss(fakeA, imgA)
        referenceStyleB = self._reference_style_loss(fakeB, imgB)
        referenceStyleLoss = referenceStyleA + referenceStyleB

        referenceWhiteBalanceA = self._reference_white_balance_loss(fakeA, imgA)
        referenceWhiteBalanceB = self._reference_white_balance_loss(fakeB, imgB)
        referenceWhiteBalanceLoss = (
            referenceWhiteBalanceA + referenceWhiteBalanceB
        )
        effectiveReferenceWhiteBalanceWeight = (
            self._effective_reference_white_balance_weight()
        )
        useLocalChromaGuard = any(
            weight > 0
            for weight in (
                self.reference_local_chroma_weight,
                self.reference_local_chroma_tail_weight,
                self.reference_local_red_tail_weight,
            )
        )
        if useLocalChromaGuard:
            localTermsA = self._local_chroma_terms(
                fakeA,
                imgB,
                chroma_tail_fraction=(
                    self.reference_local_chroma_tail_fraction
                ),
                red_tail_fraction=self.reference_local_red_tail_fraction,
                chroma_threshold=self.reference_local_chroma_threshold,
                red_threshold=self.reference_local_red_threshold,
            )
            localTermsB = self._local_chroma_terms(
                fakeB,
                imgA,
                chroma_tail_fraction=(
                    self.reference_local_chroma_tail_fraction
                ),
                red_tail_fraction=self.reference_local_red_tail_fraction,
                chroma_threshold=self.reference_local_chroma_threshold,
                red_threshold=self.reference_local_red_threshold,
            )
        else:
            zero = fakeA.new_zeros(())
            localTermsA = localTermsB = {
                'mean': zero,
                'chroma_tail': zero,
                'red_tail': zero,
                'chroma_bad_fraction': zero,
                'red_bad_fraction': zero,
            }
        referenceLocalChromaA = localTermsA['mean']
        referenceLocalChromaB = localTermsB['mean']
        referenceLocalChromaLoss = (
            referenceLocalChromaA + referenceLocalChromaB
        )
        referenceLocalChromaTailLoss = (
            localTermsA['chroma_tail'] + localTermsB['chroma_tail']
        )
        referenceLocalRedTailLoss = (
            localTermsA['red_tail'] + localTermsB['red_tail']
        )
        referenceRedOvershootA = self._reference_red_overshoot_loss(
            fakeA,
            imgA,
        )
        referenceRedOvershootB = self._reference_red_overshoot_loss(
            fakeB,
            imgB,
        )
        referenceRedOvershootLoss = (
            referenceRedOvershootA + referenceRedOvershootB
        )

        rangeTermsA = self._range_terms(
            fakeA,
            tail_fraction=self.range_tail_fraction,
        )
        rangeTermsB = self._range_terms(
            fakeB,
            tail_fraction=self.range_tail_fraction,
        )
        rangeA = rangeTermsA['mean']
        rangeB = rangeTermsB['mean']
        rangeLoss = rangeA + rangeB
        rangeTailLoss = rangeTermsA['tail'] + rangeTermsB['tail']
        
        # gather all losses
        adversarialLoss = adversarialA + adversarialB
        gen_loss = (
            effectiveAdversarialWeight * adversarialLoss
            + self.cycle_weight * cycleLoss
            + self.identity_weight * identityLoss
            + self.domain_statistics_weight * statisticsLoss
            + self.exposure_weight * exposureLoss
            + self.chroma_weight * chromaLoss
            + self.reflectance_weight * reflectanceLoss
            + self.patch_nce_weight * patchNceLoss
            + self.reference_style_weight * referenceStyleLoss
            + effectiveReferenceWhiteBalanceWeight * referenceWhiteBalanceLoss
            + self.reference_local_chroma_weight * referenceLocalChromaLoss
            + (
                self.reference_local_chroma_tail_weight
                * referenceLocalChromaTailLoss
            )
            + (
                self.reference_local_red_tail_weight
                * referenceLocalRedTailLoss
            )
            + (
                self.reference_red_overshoot_weight
                * referenceRedOvershootLoss
            )
            + self.range_weight * rangeLoss
            + self.range_tail_weight * rangeTailLoss
        )

        batch_size = imgA.shape[0]
        self._log_loss('gen_loss', gen_loss, batch_size, prog_bar=True)
        self._log_loss(
            'effective_adversarial_weight',
            fakeB.new_tensor(effectiveAdversarialWeight),
            batch_size,
        )
        self._log_loss('gen_adversarial_a_loss', adversarialA, batch_size)
        self._log_loss('gen_adversarial_b_loss', adversarialB, batch_size)
        self._log_loss('gen_cycle_a_loss', cycleA, batch_size)
        self._log_loss('gen_cycle_b_loss', cycleB, batch_size)
        self._log_loss('gen_identity_a_loss', identityA, batch_size)
        self._log_loss('gen_identity_b_loss', identityB, batch_size)
        self._log_loss('gen_statistics_a_loss', statisticsA, batch_size)
        self._log_loss('gen_statistics_b_loss', statisticsB, batch_size)
        self._log_loss('gen_exposure_a_loss', exposureA, batch_size)
        self._log_loss('gen_exposure_b_loss', exposureB, batch_size)
        self._log_loss('gen_chroma_a_loss', chromaA, batch_size)
        self._log_loss('gen_chroma_b_loss', chromaB, batch_size)
        self._log_loss('gen_reflectance_a_loss', reflectanceA, batch_size)
        self._log_loss('gen_reflectance_b_loss', reflectanceB, batch_size)
        self._log_loss('gen_patch_nce_a_loss', patchNceA, batch_size)
        self._log_loss('gen_patch_nce_b_loss', patchNceB, batch_size)
        self._log_loss(
            'gen_reference_style_a_loss', referenceStyleA, batch_size
        )
        self._log_loss(
            'gen_reference_style_b_loss', referenceStyleB, batch_size
        )
        self._log_loss(
            'gen_reference_white_balance_a_loss',
            referenceWhiteBalanceA,
            batch_size,
        )
        self._log_loss(
            'gen_reference_white_balance_b_loss',
            referenceWhiteBalanceB,
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_chroma_a_loss',
            referenceLocalChromaA,
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_chroma_b_loss',
            referenceLocalChromaB,
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_chroma_tail_a_loss',
            localTermsA['chroma_tail'],
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_chroma_tail_b_loss',
            localTermsB['chroma_tail'],
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_red_tail_a_loss',
            localTermsA['red_tail'],
            batch_size,
        )
        self._log_loss(
            'gen_reference_local_red_tail_b_loss',
            localTermsB['red_tail'],
            batch_size,
        )
        self._log_loss(
            'gen_reference_red_overshoot_a_loss',
            referenceRedOvershootA,
            batch_size,
        )
        self._log_loss(
            'gen_reference_red_overshoot_b_loss',
            referenceRedOvershootB,
            batch_size,
        )
        (
            redGreenDeltaB,
            blueGreenDeltaB,
            warmBiasB,
            warmAbsoluteB,
            warmPositiveFractionB,
            tintBiasB,
            tintAbsoluteB,
        ) = (
            self._reference_white_balance_deltas(fakeB.detach(), imgB)
        )
        self._log_loss(
            'fake_b_reference_red_green_delta', redGreenDeltaB, batch_size
        )
        self._log_loss(
            'fake_b_reference_blue_green_delta', blueGreenDeltaB, batch_size
        )
        self._log_loss(
            'fake_b_reference_warm_bias', warmBiasB, batch_size
        )
        self._log_loss(
            'fake_b_reference_warm_abs', warmAbsoluteB, batch_size
        )
        self._log_loss(
            'fake_b_reference_warm_positive_fraction',
            warmPositiveFractionB,
            batch_size,
        )
        self._log_loss(
            'fake_b_reference_tint_bias', tintBiasB, batch_size
        )
        self._log_loss(
            'fake_b_reference_tint_abs', tintAbsoluteB, batch_size
        )
        self._log_loss(
            'effective_reference_white_balance_weight',
            fakeB.new_tensor(effectiveReferenceWhiteBalanceWeight),
            batch_size,
        )
        self._log_loss('gen_range_a_loss', rangeA, batch_size)
        self._log_loss('gen_range_b_loss', rangeB, batch_size)
        self._log_loss(
            'gen_range_tail_a_loss',
            rangeTermsA['tail'],
            batch_size,
        )
        self._log_loss(
            'gen_range_tail_b_loss',
            rangeTermsB['tail'],
            batch_size,
        )
        self._log_loss(
            'fake_b_out_of_range_fraction',
            rangeTermsB['out_of_range_fraction'],
            batch_size,
        )
        self._log_loss(
            'fake_a_luminance', self._luminance_mean(fakeA), batch_size
        )
        self._log_loss(
            'fake_b_luminance', self._luminance_mean(fakeB), batch_size
        )
        self._log_loss(
            'real_a_luminance', self._luminance_mean(imgA), batch_size
        )
        self._log_loss(
            'real_b_luminance', self._luminance_mean(imgB), batch_size
        )
        
        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        
        return gen_loss

    def generator_warmup_step(self, imgA, imgB):
        """Learn the non-adversarial translation objective before GAN updates."""
        warmup_loss = self.generator_training_step(
            imgA,
            imgB,
            adversarial_weight=0.0,
        )
        self._log_loss(
            'warmup_loss', warmup_loss, imgA.shape[0], prog_bar=True
        )
        return warmup_loss
    
    def discriminator_training_step(self, imgA, imgB):
        """Update Discriminator"""        
        fakeA = self.fake_pool_a.query(self.fakeA)
        fakeB = self.fake_pool_b.query(self.fakeB)
        
        # dis_a checks for domain A photos
        predRealA = self.model.dis_a(imgA)
        mseRealA = self._disc_loss(predRealA, 'real')
        
        predFakeA = self.model.dis_a(fakeA)
        mseFakeA = self._disc_loss(predFakeA, 'fake')
        
        # dis_b checks for domain B photos
        predRealB = self.model.dis_b(imgB)
        mseRealB = self._disc_loss(predRealB, 'real')
        
        predFakeB = self.model.dis_b(fakeB)
        mseFakeB = self._disc_loss(predFakeB, 'fake')
        
        # gather all losses
        dis_a_loss = 0.5 * (mseFakeA + mseRealA)
        dis_b_loss = 0.5 * (mseFakeB + mseRealB)
        dis_loss = dis_a_loss + dis_b_loss
        batch_size = imgA.shape[0]
        self._log_loss('dis_loss', dis_loss, batch_size, prog_bar=True)
        self._log_loss('dis_a_loss', dis_a_loss, batch_size)
        self._log_loss('dis_b_loss', dis_b_loss, batch_size)
        self._log_loss('dis_a_real_score', predRealA.mean(), batch_size)
        self._log_loss('dis_a_fake_score', predFakeA.mean(), batch_size)
        self._log_loss('dis_b_real_score', predRealB.mean(), batch_size)
        self._log_loss('dis_b_fake_score', predFakeB.mean(), batch_size)
        return dis_loss
    
    def generator_pretaining_step(self, imgAB_recolor, imgA, imgBA_recolor, imgB):
        reco_b = self.model.gen_ab(imgBA_recolor)
        reco_a = self.model.gen_ba(imgAB_recolor)
        loss = self._cycle_loss(reco_b, imgB) + self._cycle_loss(reco_a, imgA)
        self.log('pretrain_loss', loss.item(), prog_bar=True, logger=True)
        return loss

    @staticmethod
    def _unpack_adversarial_batch(batch):
        if isinstance(batch, Mapping):
            return batch['source'], batch['target']
        if len(batch) == 4:
            _, source, _, target = batch
            return source, target
        if len(batch) == 2:
            return batch
        raise ValueError(
            "Adversarial training expects a source/target mapping, a two-item "
            "batch, or the legacy four-item recolor batch"
        )

    def _unpaired_evaluation_step(self, batch, stage: str):
        source, target = self._unpack_adversarial_batch(batch)
        target_condition = self._style_condition(source, target)
        source_condition = self._style_condition(target, source)
        if self.patch_nce_weight > 0:
            fake_target, source_features = self.model.gen_ab.forward_with_features(
                source, target_condition
            )
            fake_source, target_features = self.model.gen_ba.forward_with_features(
                target, source_condition
            )
            patch_nce_loss = (
                self._patch_nce_loss(
                    self.model.gen_ab.encode_features(
                        fake_target, target_condition
                    ),
                    source_features,
                    self.patch_nce_num_patches,
                    self.patch_nce_temperature,
                    random_sampling=False,
                )
                + self._patch_nce_loss(
                    self.model.gen_ba.encode_features(
                        fake_source, source_condition
                    ),
                    target_features,
                    self.patch_nce_num_patches,
                    self.patch_nce_temperature,
                    random_sampling=False,
                )
            )
        else:
            fake_target = self.model.gen_ab(source, target_condition)
            fake_source = self.model.gen_ba(target, source_condition)
            patch_nce_loss = fake_target.new_zeros(())
        cycled_source = self.model.gen_ba(
            fake_target,
            self._style_condition(fake_target, source),
        )
        cycled_target = self.model.gen_ab(
            fake_source,
            self._style_condition(fake_source, target),
        )
        same_source = self.model.gen_ba(
            source,
            self._style_condition(source, source),
        )
        same_target = self.model.gen_ab(
            target,
            self._style_condition(target, target),
        )

        cycle_loss = (
            self._cycle_loss(cycled_source, source)
            + self._cycle_loss(cycled_target, target)
        )
        identity_loss = (
            self._identity_loss(same_source, source)
            + self._identity_loss(same_target, target)
        )
        adversarial_loss = (
            self._disc_loss(self.model.dis_a(fake_source), 'real')
            + self._disc_loss(self.model.dis_b(fake_target), 'real')
        )
        statistics_loss = (
            self._domain_statistics_loss(fake_source, source)
            + self._domain_statistics_loss(fake_target, target)
        )
        exposure_loss = (
            self._exposure_loss(fake_source, target)
            + self._exposure_loss(fake_target, source)
        )
        if self.chroma_weight > 0:
            chroma_loss = (
                self._chroma_loss(fake_source, target)
                + self._chroma_loss(fake_target, source)
            )
        else:
            chroma_loss = fake_source.new_zeros(())
        if self.reflectance_weight > 0:
            reflectance_loss = (
                self._reflectance_loss(fake_source, target)
                + self._reflectance_loss(fake_target, source)
            )
        else:
            reflectance_loss = fake_source.new_zeros(())
        reference_style_loss = (
            self._reference_style_loss(fake_source, source)
            + self._reference_style_loss(fake_target, target)
        )
        reference_white_balance_loss = (
            self._reference_white_balance_loss(fake_source, source)
            + self._reference_white_balance_loss(fake_target, target)
        )
        effective_reference_white_balance_weight = (
            self._effective_reference_white_balance_weight()
        )
        use_local_chroma_guard = any(
            weight > 0
            for weight in (
                self.reference_local_chroma_weight,
                self.reference_local_chroma_tail_weight,
                self.reference_local_red_tail_weight,
            )
        )
        if use_local_chroma_guard:
            fake_source_local_terms = self._local_chroma_terms(
                fake_source,
                target,
                chroma_tail_fraction=(
                    self.reference_local_chroma_tail_fraction
                ),
                red_tail_fraction=self.reference_local_red_tail_fraction,
                chroma_threshold=self.reference_local_chroma_threshold,
                red_threshold=self.reference_local_red_threshold,
            )
            fake_target_local_terms = self._local_chroma_terms(
                fake_target,
                source,
                chroma_tail_fraction=(
                    self.reference_local_chroma_tail_fraction
                ),
                red_tail_fraction=self.reference_local_red_tail_fraction,
                chroma_threshold=self.reference_local_chroma_threshold,
                red_threshold=self.reference_local_red_threshold,
            )
        else:
            zero = fake_source.new_zeros(())
            fake_source_local_terms = fake_target_local_terms = {
                'mean': zero,
                'chroma_tail': zero,
                'red_tail': zero,
                'chroma_bad_fraction': zero,
                'red_bad_fraction': zero,
            }
        reference_local_chroma_loss = (
            fake_source_local_terms['mean']
            + fake_target_local_terms['mean']
        )
        reference_local_chroma_tail_loss = (
            fake_source_local_terms['chroma_tail']
            + fake_target_local_terms['chroma_tail']
        )
        reference_local_red_tail_loss = (
            fake_source_local_terms['red_tail']
            + fake_target_local_terms['red_tail']
        )
        fake_target_red_overshoot_loss = (
            self._reference_red_overshoot_loss(fake_target, target)
        )
        reference_red_overshoot_loss = (
            self._reference_red_overshoot_loss(fake_source, source)
            + fake_target_red_overshoot_loss
        )
        fake_source_range_terms = self._range_terms(
            fake_source,
            tail_fraction=self.range_tail_fraction,
        )
        fake_target_range_terms = self._range_terms(
            fake_target,
            tail_fraction=self.range_tail_fraction,
        )
        range_loss = (
            fake_source_range_terms['mean']
            + fake_target_range_terms['mean']
        )
        range_tail_loss = (
            fake_source_range_terms['tail']
            + fake_target_range_terms['tail']
        )
        loss = (
            self.adversarial_weight * adversarial_loss
            + self.cycle_weight * cycle_loss
            + self.identity_weight * identity_loss
            + self.domain_statistics_weight * statistics_loss
            + self.exposure_weight * exposure_loss
            + self.chroma_weight * chroma_loss
            + self.reflectance_weight * reflectance_loss
            + self.patch_nce_weight * patch_nce_loss
            + self.reference_style_weight * reference_style_loss
            + (
                effective_reference_white_balance_weight
                * reference_white_balance_loss
            )
            + self.reference_local_chroma_weight * reference_local_chroma_loss
            + (
                self.reference_local_chroma_tail_weight
                * reference_local_chroma_tail_loss
            )
            + (
                self.reference_local_red_tail_weight
                * reference_local_red_tail_loss
            )
            + (
                self.reference_red_overshoot_weight
                * reference_red_overshoot_loss
            )
            + self.range_weight * range_loss
            + self.range_tail_weight * range_tail_loss
        )
        if self.reference_guided:
            input_distance, fake_distance, distance_ratio = (
                self._reference_style_distances(source, fake_target, target)
            )
            fake_target_luminance = self._luminance_mean(fake_target)
            real_target_luminance = self._luminance_mean(target)
            fake_target_luminance_error = (
                fake_target_luminance - real_target_luminance
            ).abs()
            fake_target_luminance_ratio = (
                fake_target_luminance
                / real_target_luminance.clamp_min(1e-6)
            )
            # Fixed-weight checkpoint score. It deliberately excludes the
            # discriminator because GAN scores change as D learns and are not
            # comparable across the warmup/ramp/full-training phases.
            reference_selection_loss = (
                10.0 * fake_distance
                + cycle_loss
                + identity_loss
                + reference_white_balance_loss
                + 0.5 * reflectance_loss
                + 0.25 * patch_nce_loss
                + fake_target_local_terms['mean']
                + fake_target_local_terms['chroma_tail']
                + 2.0 * fake_target_local_terms['red_tail']
                + fake_target_red_overshoot_loss
                + 2.0 * fake_target_luminance_error
                + (
                    10.0
                    * fake_target_range_terms['out_of_range_fraction']
                )
            )
        self.log(f'{stage}_cycle_loss', cycle_loss, prog_bar=False, logger=True)
        self.log(f'{stage}_identity_loss', identity_loss, prog_bar=False, logger=True)
        self.log(f'{stage}_adversarial_loss', adversarial_loss, prog_bar=False, logger=True)
        self.log(f'{stage}_statistics_loss', statistics_loss, prog_bar=False, logger=True)
        self.log(f'{stage}_exposure_loss', exposure_loss, prog_bar=False, logger=True)
        self.log(f'{stage}_chroma_loss', chroma_loss, prog_bar=False, logger=True)
        self.log(
            f'{stage}_reflectance_loss',
            reflectance_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_patch_nce_loss',
            patch_nce_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_style_loss',
            reference_style_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_white_balance_loss',
            reference_white_balance_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_effective_reference_white_balance_weight',
            reference_white_balance_loss.new_tensor(
                effective_reference_white_balance_weight
            ),
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_local_chroma_loss',
            reference_local_chroma_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_local_chroma_tail_loss',
            reference_local_chroma_tail_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_local_red_tail_loss',
            reference_local_red_tail_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_reference_red_overshoot_loss',
            reference_red_overshoot_loss,
            prog_bar=False,
            logger=True,
        )
        if self.reference_guided:
            self.log(
                f'{stage}_reference_selection_loss',
                reference_selection_loss,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_luminance_ratio',
                fake_target_luminance_ratio,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_source_reference_style_distance',
                input_distance,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_reference_style_distance',
                fake_distance,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_reference_style_ratio',
                distance_ratio,
                prog_bar=False,
                logger=True,
            )
            (
                red_green_delta,
                blue_green_delta,
                warm_bias,
                warm_absolute,
                warm_positive_fraction,
                tint_bias,
                tint_absolute,
            ) = (
                self._reference_white_balance_deltas(
                    fake_target,
                    target,
                )
            )
            (
                _,
                _,
                source_warm_bias,
                source_warm_absolute,
                _,
                source_tint_bias,
                source_tint_absolute,
            ) = self._reference_white_balance_deltas(source, target)
            self.log(
                f'{stage}_fake_target_red_green_delta',
                red_green_delta,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_blue_green_delta',
                blue_green_delta,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_warm_bias',
                warm_bias,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_warm_abs',
                warm_absolute,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_warm_positive_fraction',
                warm_positive_fraction,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_tint_bias',
                tint_bias,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_tint_abs',
                tint_absolute,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_source_target_warm_bias',
                source_warm_bias,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_source_target_warm_abs',
                source_warm_absolute,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_source_target_tint_bias',
                source_tint_bias,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_source_target_tint_abs',
                source_tint_absolute,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_local_chroma_mean',
                fake_target_local_terms['mean'],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_local_chroma_tail',
                fake_target_local_terms['chroma_tail'],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_local_red_tail',
                fake_target_local_terms['red_tail'],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_local_chroma_bad_fraction',
                fake_target_local_terms['chroma_bad_fraction'],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_local_red_bad_fraction',
                fake_target_local_terms['red_bad_fraction'],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f'{stage}_fake_target_red_overshoot_loss',
                fake_target_red_overshoot_loss,
                prog_bar=False,
                logger=True,
            )
        self.log(f'{stage}_range_loss', range_loss, prog_bar=False, logger=True)
        self.log(
            f'{stage}_range_tail_loss',
            range_tail_loss,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_fake_target_out_of_range_fraction',
            fake_target_range_terms['out_of_range_fraction'],
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_fake_source_luminance',
            self._luminance_mean(fake_source),
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_fake_target_luminance',
            self._luminance_mean(fake_target),
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_real_source_luminance',
            self._luminance_mean(source),
            prog_bar=False,
            logger=True,
        )
        self.log(
            f'{stage}_real_target_luminance',
            self._luminance_mean(target),
            prog_bar=False,
            logger=True,
        )
        self.log(f'{stage}_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}


    def training_step(self, batch, batch_idx):
        if not self.adversarial:
            img_ab_recolorized, img_a, img_ba_recolorized, img_b = batch
            loss = self.generator_pretaining_step(img_ab_recolorized, img_a, img_ba_recolorized, img_b)
            return {'loss': loss}
        else:
            img_a, img_b = self._unpack_adversarial_batch(batch)
            opt_gen, opt_disc = self.optimizers()
            sch_gen, sch_disc = self.lr_schedulers()

            if self.current_epoch < self.warmup_epochs:
                self.toggle_optimizer(opt_gen)
                self._set_requires_grad(
                    [self.model.dis_a, self.model.dis_b], requires_grad=False
                )
                opt_gen.zero_grad()
                warmup_loss = self.generator_warmup_step(img_a, img_b)
                self.manual_backward(warmup_loss)
                self._clip_optimizer_gradients(opt_gen)
                opt_gen.step()
                self.untoggle_optimizer(opt_gen)
                if self.trainer.is_last_batch:
                    sch_gen.step()
                return
            
            # train generator
            self.toggle_optimizer(opt_gen)
            self._set_requires_grad([self.model.dis_a, self.model.dis_b], requires_grad=False)
            opt_gen.zero_grad()
            gen_loss = self.generator_training_step(img_a, img_b)
            self.manual_backward(gen_loss)
            self._clip_optimizer_gradients(opt_gen)
            opt_gen.step()
            self.untoggle_optimizer(opt_gen)
            
            # train discriminator
            self.toggle_optimizer(opt_disc)
            self._set_requires_grad([self.model.dis_a, self.model.dis_b], requires_grad=True)
            opt_disc.zero_grad()
            disc_loss = self.discriminator_training_step(img_a, img_b)
            self.manual_backward(disc_loss)
            self._clip_optimizer_gradients(opt_disc)
            opt_disc.step()
            self.untoggle_optimizer(opt_disc)

            if self.trainer.is_last_batch:
                sch_gen.step()
                sch_disc.step()
            return
    
    def validation_step(self, batch, batch_idx):
        if isinstance(batch, Mapping):
            return self._unpaired_evaluation_step(batch, stage='val')

        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def test_step(self, batch, batch_idx):
        if isinstance(batch, Mapping):
            return self._unpaired_evaluation_step(batch, stage='test')

        if self.reverse_prediction:
            targets, inputs = batch
            predictions = self.reversed_forward(inputs)
        else:
            inputs, targets = batch
            predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('test_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def predict_step(self, batch, batch_idx):
        if self.reference_guided:
            if len(batch) != 3:
                raise ValueError(
                    "Reference-guided prediction expects "
                    "(paths, inputs, references)"
                )
            pathes, inputs, references = batch
            if self.reverse_prediction:
                output = self.reversed_forward(inputs, references)
            else:
                output = self(inputs, references)
            return output

        if self.reverse_prediction:
            pathes, inputs = batch
            output = self.reversed_forward(inputs)
        else:
            pathes, inputs = batch
            output = self(inputs)
        return output
