import itertools
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
        cycle_weight: float = 10.0,
        identity_weight: float = 5.0,
        domain_statistics_weight: float = 0.0,
        exposure_weight: float = 0.0,
        chroma_weight: float = 0.0,
        reflectance_weight: float = 0.0,
        patch_nce_weight: float = 0.0,
        patch_nce_num_patches: int = 256,
        patch_nce_temperature: float = 0.07,
        range_weight: float = 0.0,
        warmup_epochs: int = 0,
        gradient_clip_val: float = 0.0,
        discriminator_lr_scale: float = 1.0,
    ) -> None:
        super(UnsupervisedPipeline, self).__init__()

        self.model = model
        self.fake_pool_a = ImagePool()
        self.fake_pool_b = ImagePool()
        self.adversarial_weight = adversarial_weight
        self.cycle_weight = cycle_weight
        self.identity_weight = identity_weight
        self.domain_statistics_weight = domain_statistics_weight
        self.exposure_weight = exposure_weight
        self.chroma_weight = chroma_weight
        self.reflectance_weight = reflectance_weight
        self.patch_nce_weight = patch_nce_weight
        self.patch_nce_num_patches = patch_nce_num_patches
        self.patch_nce_temperature = patch_nce_temperature
        self.range_weight = range_weight
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

        self.save_hyperparameters(ignore=['model', 'reverse_prediction'])

    def _identity_loss(self, predictions, targets):
        mae_loss = self.mae_loss(predictions, targets)
        return mae_loss

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
    def _range_loss(predictions):
        """Penalize values that would be clipped when an image is saved."""
        return F.relu(-predictions).mean() + F.relu(predictions - 1).mean()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model.gen_ab(x)
        return pred
    
    def reversed_forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model.gen_ba(x)
        return pred
    
    def generator_training_step(self, imgA, imgB):        
        """cycle images - using only generator nets"""
        if self.patch_nce_weight > 0:
            fakeB, source_features = self.model.gen_ab.forward_with_features(imgA)
            fakeA, target_features = self.model.gen_ba.forward_with_features(imgB)
            fakeB_features = self.model.gen_ab.encode_features(fakeB)
            fakeA_features = self.model.gen_ba.encode_features(fakeA)
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
            fakeB = self.model.gen_ab(imgA)
            fakeA = self.model.gen_ba(imgB)
            patchNceA = fakeA.new_zeros(())
            patchNceB = fakeB.new_zeros(())
        patchNceLoss = patchNceA + patchNceB

        cycledA = self.model.gen_ba(fakeB)
        cycledB = self.model.gen_ab(fakeA)
        
        sameB = self.model.gen_ab(imgB)
        sameA = self.model.gen_ba(imgA)
        
        # generator gen_ab must fool discrim dis_b so label is real = 1
        predFakeB = self.model.dis_b(fakeB)
        adversarialB = self._disc_loss(predFakeB, 'real')
        
        # generator gen_ba must fool discrim dis_a so label is real
        predFakeA = self.model.dis_a(fakeA)
        adversarialA = self._disc_loss(predFakeA, 'real')
        
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

        rangeA = self._range_loss(fakeA)
        rangeB = self._range_loss(fakeB)
        rangeLoss = rangeA + rangeB
        
        # gather all losses
        adversarialLoss = adversarialA + adversarialB
        gen_loss = (
            self.adversarial_weight * adversarialLoss
            + self.cycle_weight * cycleLoss
            + self.identity_weight * identityLoss
            + self.domain_statistics_weight * statisticsLoss
            + self.exposure_weight * exposureLoss
            + self.chroma_weight * chromaLoss
            + self.reflectance_weight * reflectanceLoss
            + self.patch_nce_weight * patchNceLoss
            + self.range_weight * rangeLoss
        )

        batch_size = imgA.shape[0]
        self._log_loss('gen_loss', gen_loss, batch_size, prog_bar=True)
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
        self._log_loss('gen_range_a_loss', rangeA, batch_size)
        self._log_loss('gen_range_b_loss', rangeB, batch_size)
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
        """Initialize both generators near identity before adversarial updates."""
        fakeB = self.model.gen_ab(imgA)
        fakeA = self.model.gen_ba(imgB)
        sameB = self.model.gen_ab(imgB)
        sameA = self.model.gen_ba(imgA)
        warmup_loss = (
            self._cycle_loss(fakeB, imgA)
            + self._cycle_loss(fakeA, imgB)
            + self._cycle_loss(sameB, imgB)
            + self._cycle_loss(sameA, imgA)
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
        if self.patch_nce_weight > 0:
            fake_target, source_features = self.model.gen_ab.forward_with_features(
                source
            )
            fake_source, target_features = self.model.gen_ba.forward_with_features(
                target
            )
            patch_nce_loss = (
                self._patch_nce_loss(
                    self.model.gen_ab.encode_features(fake_target),
                    source_features,
                    self.patch_nce_num_patches,
                    self.patch_nce_temperature,
                    random_sampling=False,
                )
                + self._patch_nce_loss(
                    self.model.gen_ba.encode_features(fake_source),
                    target_features,
                    self.patch_nce_num_patches,
                    self.patch_nce_temperature,
                    random_sampling=False,
                )
            )
        else:
            fake_target = self.model.gen_ab(source)
            fake_source = self.model.gen_ba(target)
            patch_nce_loss = fake_target.new_zeros(())
        cycled_source = self.model.gen_ba(fake_target)
        cycled_target = self.model.gen_ab(fake_source)
        same_source = self.model.gen_ba(source)
        same_target = self.model.gen_ab(target)

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
        range_loss = (
            self._range_loss(fake_source)
            + self._range_loss(fake_target)
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
            + self.range_weight * range_loss
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
        self.log(f'{stage}_range_loss', range_loss, prog_bar=False, logger=True)
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
                    sch_disc.step()
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
        if self.reverse_prediction:
            pathes, inputs = batch
            output = self.reversed_forward(inputs)
        else:
            pathes, inputs = batch
            output = self(inputs)
        return output
