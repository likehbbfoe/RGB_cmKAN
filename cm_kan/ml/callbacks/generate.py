from collections.abc import Mapping

from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import torchvision
import os


class GenerateCallback(Callback):
    def __init__(
            self,
            every_n_epochs=1
        ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None
        self.is_unpaired = False

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, Mapping):
            return batch["source"], batch["target"], True
        inputs, targets = batch
        return inputs, targets, False

    def _capture_batch(self, batch, pl_module: LightningModule) -> None:
        self.input_imgs, self.target_imgs, self.is_unpaired = self._unpack_batch(
            batch
        )
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)

    def _make_preview(self, pl_module: LightningModule) -> tuple[torch.Tensor, str]:
        if not self.is_unpaired:
            predictions = pl_module(self.input_imgs)
            images = torch.stack(
                [self.input_imgs, self.target_imgs, predictions], dim=1
            ).flatten(0, 1)
            return images, "reconst"

        fake_targets = pl_module(self.input_imgs)
        cycled_sources = pl_module.reversed_forward(fake_targets)
        fake_sources = pl_module.reversed_forward(self.target_imgs)
        cycled_targets = pl_module(fake_sources)

        # Each row has three related images. Forward rows are followed by
        # reverse rows; the random unpaired source/target images are never
        # presented as if they were corresponding ground truth.
        forward_images = torch.stack(
            [self.input_imgs, fake_targets, cycled_sources], dim=1
        ).flatten(0, 1)
        reverse_images = torch.stack(
            [self.target_imgs, fake_sources, cycled_targets], dim=1
        ).flatten(0, 1)
        return torch.cat([forward_images, reverse_images], dim=0), "translation_cycle"

    def _save_preview(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prefix: str,
    ) -> None:
        was_training = pl_module.training
        with torch.no_grad():
            pl_module.eval()
            images, preview_name = self._make_preview(pl_module)
        if was_training:
            pl_module.train()

        grid = torchvision.utils.make_grid(images, nrow=3)
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.save_dir,
            f"{prefix}{preview_name}_{trainer.current_epoch}.png",
        )
        torchvision.utils.save_image(grid, save_path)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self._capture_batch(next(iter(dataloader)), pl_module)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self._capture_batch(next(iter(dataloader)), pl_module)
        self.save_dir = os.path.join(trainer.log_dir, 'figures')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="test_")
