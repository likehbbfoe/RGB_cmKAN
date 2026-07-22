from collections.abc import Mapping
import os

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class GenerateCallback(Callback):
    def __init__(self, every_n_epochs=1) -> None:
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

    def _make_preview(
        self,
        pl_module: LightningModule,
    ) -> tuple[torch.Tensor, str]:
        predictions = pl_module(self.input_imgs)
        images = self._four_column_preview(
            self.input_imgs,
            predictions,
            self.target_imgs,
        )
        preview_name = "source_to_target" if self.is_unpaired else "reconst"
        return images, preview_name

    @classmethod
    def _four_column_preview(
        cls,
        sources: torch.Tensor,
        translated: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Build source | translated | target | distribution rows."""
        height, width = sources.shape[-2:]
        distribution_tiles = torch.stack(
            [
                cls._distribution_tile(
                    source,
                    prediction,
                    target,
                    height,
                    width,
                )
                for source, prediction, target in zip(
                    sources,
                    translated,
                    targets,
                )
            ]
        ).to(device=sources.device, dtype=sources.dtype)
        return torch.stack(
            [sources, translated, targets, distribution_tiles],
            dim=1,
        ).flatten(0, 1)

    @staticmethod
    def _display_values(image: torch.Tensor) -> torch.Tensor:
        """Return all displayed RGB pixel values as one clipped distribution."""
        return image.detach().float().clamp(0, 1).cpu().flatten()

    @staticmethod
    def _histogram(values: torch.Tensor, bins: int = 64) -> tuple[list, list]:
        """Create a normalized histogram for values displayed in [0, 1]."""
        histogram = torch.histc(values, bins=bins, min=0.0, max=1.0)
        histogram /= histogram.sum().clamp_min(1)
        centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
        return centers.tolist(), histogram.tolist()

    @classmethod
    def _distribution_tile(
        cls,
        source: torch.Tensor,
        translated: torch.Tensor,
        target: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Render a compact all-channel pixel distribution as an image tile."""
        figure = Figure(figsize=(3.0, 3.0), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()

        series = (
            ("Source", source, "#7B8794", "--"),
            ("Translated", translated, "#E76F51", "-"),
            ("Target", target, "#264653", ":"),
        )
        for label, image, color, line_style in series:
            values = cls._display_values(image)
            x_values, probabilities = cls._histogram(values)
            axis.plot(
                x_values,
                probabilities,
                label=f"{label} ({values.mean().item():.3f})",
                color=color,
                linestyle=line_style,
                linewidth=1.8,
            )

        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Pixel value", fontsize=8)
        axis.set_ylabel("Proportion", fontsize=8)
        axis.set_title("Pixel distribution", fontsize=9, fontweight="bold")
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=6, loc="upper right")
        figure.tight_layout(pad=0.5)

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba()).copy()
        tile = torch.from_numpy(rgba[:, :, :3]).permute(2, 0, 1).float() / 255.0
        return F.interpolate(
            tile.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

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

        grid = torchvision.utils.make_grid(images, nrow=4)
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.save_dir,
            f"{prefix}{preview_name}_{trainer.current_epoch}.png",
        )
        torchvision.utils.save_image(grid, save_path)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self._capture_batch(next(iter(dataloader)), pl_module)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self._capture_batch(next(iter(dataloader)), pl_module)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="test_")
