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
    def __init__(self, every_n_epochs=1, max_preview_candidates=64) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_preview_candidates = max_preview_candidates
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None
        self.is_unpaired = False
        self.preview_pair_distance = None

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

    @classmethod
    def _chromaticity_centroid(cls, image: torch.Tensor) -> torch.Tensor | None:
        """Return the mean visible-pixel CIE xy coordinate for sample selection."""
        xy = cls._rgb_to_xy(image)
        if xy.shape[0] == 0:
            return None
        return xy.mean(dim=0)

    def _capture_most_distinct_batch(
        self,
        dataloader,
        pl_module: LightningModule,
    ) -> None:
        """Keep the unpaired validation pair with the largest xy centroid gap."""
        best_source = None
        best_target = None
        best_distance = -1.0
        candidate_count = 0

        for batch in dataloader:
            sources, targets, is_unpaired = self._unpack_batch(batch)
            if not is_unpaired:
                self._capture_batch(batch, pl_module)
                return

            for source, target in zip(sources, targets):
                source_centroid = self._chromaticity_centroid(source)
                target_centroid = self._chromaticity_centroid(target)
                if source_centroid is not None and target_centroid is not None:
                    distance = torch.linalg.vector_norm(
                        source_centroid - target_centroid
                    ).item()
                    if distance > best_distance:
                        best_distance = distance
                        best_source = source.detach().cpu().clone()
                        best_target = target.detach().cpu().clone()

                candidate_count += 1
                if candidate_count >= self.max_preview_candidates:
                    break
            if candidate_count >= self.max_preview_candidates:
                break

        if best_source is None or best_target is None:
            raise ValueError("No valid source/target pair found for preview")

        self.input_imgs = best_source.unsqueeze(0).to(pl_module.device)
        self.target_imgs = best_target.unsqueeze(0).to(pl_module.device)
        self.is_unpaired = True
        self.preview_pair_distance = best_distance

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
        preview_scale: int = 2,
    ) -> torch.Tensor:
        """Build source | translated | target | scatter rows."""
        sources = sources.detach().float().cpu()
        translated = translated.detach().float().cpu()
        targets = targets.detach().float().cpu()
        source_height, source_width = sources.shape[-2:]
        height = source_height * preview_scale
        width = source_width * preview_scale
        resized_sources = F.interpolate(
            sources,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        resized_translated = F.interpolate(
            translated,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        resized_targets = F.interpolate(
            targets,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        scatter_tiles = torch.stack(
            [
                cls._scatter_tile(
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
        )
        return torch.stack(
            [resized_sources, resized_translated, resized_targets, scatter_tiles],
            dim=1,
        ).flatten(0, 1)

    @staticmethod
    def _rgb_to_xy(image: torch.Tensor) -> torch.Tensor:
        """Convert a CHW sRGB image to valid CIE 1931 xy chromaticities."""
        srgb = (
            image.detach()
            .float()
            .clamp(0, 1)
            .cpu()
            .permute(1, 2, 0)
            .reshape(-1, 3)
        )
        linear_rgb = torch.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055).pow(2.4),
        )
        rgb_to_xyz = linear_rgb.new_tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        xyz = linear_rgb @ rgb_to_xyz.T
        xyz_sum = xyz.sum(dim=1)
        valid = (xyz_sum > 1e-6) & (xyz[:, 1] > 1e-4)
        return xyz[valid, :2] / xyz_sum[valid, None]

    @staticmethod
    def _adaptive_xy_limits(
        xy: torch.Tensor,
        minimum_span: float = 0.03,
        margin_ratio: float = 0.18,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Zoom to the central point cloud without changing CIE xy geometry."""
        lower = torch.quantile(xy, 0.02, dim=0)
        upper = torch.quantile(xy, 0.98, dim=0)
        center = (lower + upper) / 2
        span = max((upper - lower).max().item(), minimum_span)
        half_span = span * (0.5 + margin_ratio)

        def bounded_limits(value: float, lower_bound: float, upper_bound: float):
            low = value - half_span
            high = value + half_span
            if low < lower_bound:
                high += lower_bound - low
                low = lower_bound
            if high > upper_bound:
                low -= high - upper_bound
                high = upper_bound
            return max(lower_bound, low), min(upper_bound, high)

        return (
            bounded_limits(center[0].item(), 0.0, 0.8),
            bounded_limits(center[1].item(), 0.0, 0.9),
        )

    @classmethod
    def _scatter_tile(
        cls,
        source: torch.Tensor,
        translated: torch.Tensor,
        target: torch.Tensor,
        height: int,
        width: int,
        max_points: int = 600,
    ) -> torch.Tensor:
        """Render source, translated, and target CIE xy chromaticities."""
        figure = Figure(figsize=(4.0, 4.0), dpi=200)
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()

        series = (
            ("Source", source, "#0072B2", "o", 1, 0.32),
            ("Target", target, "#CC79A7", "^", 2, 0.32),
            ("Translated", translated, "#E69F00", "D", 3, 0.55),
        )
        prepared_series = []
        for label, image, color, marker, zorder, alpha in series:
            xy = cls._rgb_to_xy(image)
            if xy.shape[0] == 0:
                continue
            point_count = min(max_points, xy.shape[0])
            point_indices = torch.linspace(
                0,
                xy.shape[0] - 1,
                steps=point_count,
            ).long()
            sampled_xy = xy[point_indices]
            prepared_series.append(
                (label, color, marker, zorder, alpha, xy, sampled_xy)
            )

        if not prepared_series:
            raise ValueError("No valid chromaticity points found for preview")

        centroids = {}
        for (
            label,
            color,
            marker,
            zorder,
            alpha,
            xy,
            sampled_xy,
        ) in prepared_series:
            axis.scatter(
                sampled_xy[:, 0].numpy(),
                sampled_xy[:, 1].numpy(),
                s=8,
                alpha=alpha,
                color=color,
                marker=marker,
                edgecolors="none",
                rasterized=True,
                label=label,
                zorder=zorder,
            )
            centroid = xy.mean(dim=0)
            centroids[label] = centroid
            axis.scatter(
                centroid[0].item(),
                centroid[1].item(),
                marker="X",
                s=60,
                linewidths=0.7,
                color=color,
                edgecolors="white",
                zorder=zorder + 1,
            )

        combined_xy = torch.cat(
            [item[-1] for item in prepared_series],
            dim=0,
        )
        x_limits, y_limits = cls._adaptive_xy_limits(combined_xy)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("CIE x", fontsize=10)
        axis.set_ylabel("CIE y", fontsize=10)
        title = "CIE 1931 xy chromaticity"
        if {"Source", "Target", "Translated"}.issubset(centroids):
            source_target_distance = torch.linalg.vector_norm(
                centroids["Source"] - centroids["Target"]
            ).item()
            translated_target_distance = torch.linalg.vector_norm(
                centroids["Translated"] - centroids["Target"]
            ).item()
            title += (
                f"\ncentroid distance: S-T {source_target_distance:.4f}, "
                f"F-T {translated_target_distance:.4f}"
            )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="upper right", markerscale=2)
        figure.tight_layout(pad=0.7)

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
        self._capture_most_distinct_batch(dataloader, pl_module)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self._capture_most_distinct_batch(dataloader, pl_module)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_preview(trainer, pl_module, prefix="test_")
