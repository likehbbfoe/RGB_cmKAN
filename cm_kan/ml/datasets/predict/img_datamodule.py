import os
from typing import Optional, Tuple

import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ConvertImageDtype,
    ToImageTensor,
)
from torch.utils.data import DataLoader

from .img_dataset import ImagePredictDataset, ImagePairedPredictDataset
from cm_kan.core.config.pipeline import PipelineType


class ImgPredictDataModule(L.LightningDataModule):
    def __init__(
            self,
            input_path: str,
            reference_path: Optional[str] = None,
            pipeline_type: PipelineType = PipelineType.supervised,
            reference_guided: bool = False,
            batch_size: int = 4,
            img_exts: Tuple[str, ...] = (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            ),
            num_workers: int = min(12, os.cpu_count() - 1),
    ) -> None:
        super().__init__()
        self.predict_dataset = None
        self.input_paths = self._collect_image_paths(
            input_path,
            img_exts,
            allow_file=True,
            label="input",
        )
        if not self.input_paths:
            raise ValueError(f"No supported input images were found in '{input_path}'.")

        self.reference_guided = reference_guided
        self.uses_reference = (
            pipeline_type == PipelineType.pair_based or reference_guided
        )
        self.reference_paths = None
        if self.uses_reference:
            if reference_path is None:
                raise ValueError("A reference image or directory is required for prediction.")

            reference_paths = self._collect_image_paths(
                reference_path,
                img_exts,
                allow_file=reference_guided,
                label="reference",
            )
            if not reference_paths:
                raise ValueError(
                    f"No supported reference images were found in '{reference_path}'."
                )

            if reference_guided:
                reference_count = len(reference_paths)
                input_count = len(self.input_paths)
                if reference_count not in (1, input_count):
                    raise ValueError(
                        "Reference-guided prediction accepts either one reference "
                        "image (broadcast to all inputs) or one sorted reference per "
                        f"sorted input. Got {reference_count} reference images for "
                        f"{input_count} input images."
                    )
            elif len(reference_paths) != len(self.input_paths):
                raise ValueError(
                    "Pair-based prediction requires the same number of input and "
                    f"reference images; got {len(self.input_paths)} inputs and "
                    f"{len(reference_paths)} references."
                )

            self.reference_paths = reference_paths

        self.image_transform = Compose([
            ToImageTensor(),
            ConvertImageDtype(dtype=torch.float32),
        ])
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pipeline_type = pipeline_type

    @staticmethod
    def _collect_image_paths(
        path: str,
        img_exts: Tuple[str, ...],
        allow_file: bool,
        label: str,
    ) -> list[str]:
        normalized_exts = tuple(ext.lower() for ext in img_exts)

        if os.path.isfile(path):
            if not allow_file:
                raise ValueError(f"The {label} path '{path}' must be a directory.")
            if not path.lower().endswith(normalized_exts):
                raise ValueError(
                    f"Unsupported {label} image extension for '{path}'. "
                    f"Supported extensions: {', '.join(img_exts)}."
                )
            return [path]

        if not os.path.isdir(path):
            expected = "an image file or directory" if allow_file else "a directory"
            raise ValueError(f"The {label} path '{path}' must be {expected}.")

        paths = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if os.path.isfile(os.path.join(path, fname))
            and fname.lower().endswith(normalized_exts)
        ]
        return sorted(paths)

    def setup(self, stage: str) -> None:
        if stage == 'predict' or stage is None:
            if self.uses_reference:
                self.dataset = ImagePairedPredictDataset(
                    self.input_paths, self.reference_paths, self.image_transform,
                )
            else:
                self.dataset = ImagePredictDataset(
                    self.input_paths, self.image_transform,
                )
            
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
