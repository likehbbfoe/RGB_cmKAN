from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cm_kan.ml.utils.io import read_rgb_image


ImageTransform = Callable[[np.ndarray], torch.Tensor]


def _ensure_rgb(image: np.ndarray, path: Path) -> np.ndarray:
    """Return an HWC image with exactly three color channels."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], repeats=3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image at '{path}', got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels at '{path}', got {image.shape[-1]}")

    # torch 2.0 cannot create tensors from NumPy uint16/uint32 arrays.
    # Convert these losslessly to normalized float values before torchvision.
    if np.issubdtype(image.dtype, np.unsignedinteger) and image.dtype != np.uint8:
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    return np.ascontiguousarray(image)


class UnpairedImageDataset(Dataset):
    """Load independent images from source and target color domains."""

    def __init__(
        self,
        source_paths: Sequence[Path],
        target_paths: Sequence[Path],
        transform: ImageTransform,
        random_pairing: bool,
    ) -> None:
        if not source_paths:
            raise ValueError("source_paths must contain at least one image")
        if not target_paths:
            raise ValueError("target_paths must contain at least one image")

        self.source_paths = tuple(source_paths)
        self.target_paths = tuple(target_paths)
        self.transform = transform
        self.random_pairing = random_pairing

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_path = self.source_paths[index % len(self.source_paths)]
        if self.random_pairing:
            target_index = torch.randint(len(self.target_paths), size=()).item()
        else:
            target_index = index % len(self.target_paths)
        target_path = self.target_paths[target_index]

        source = _ensure_rgb(read_rgb_image(str(source_path)), source_path)
        target = _ensure_rgb(read_rgb_image(str(target_path)), target_path)

        # The two domains are unpaired, so each image receives independent
        # random geometry from the same transform pipeline.
        return {
            "source": self.transform(source),
            "target": self.transform(target),
        }

    def __len__(self) -> int:
        return max(len(self.source_paths), len(self.target_paths))
