import torch
import os
from pathlib import Path
from torch.utils.data import Dataset
from cm_kan.ml.utils.io import read_rgb_image
from cm_kan.ml.datasets.custom_unpaired.img_dataset import _ensure_rgb
from typing import List
from torchvision.transforms.v2 import Compose


class ImagePredictDataset(Dataset):
    def __init__(self, paths: List[str], transform: Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        path = self.paths[idx]
        x = _ensure_rgb(read_rgb_image(path), Path(path))
        path = os.path.basename(path)
        if self.transform is not None:
            x = self.transform(x)
        return path, x

    def __len__(self) -> int:
        return len(self.paths)
    

class ImagePairedPredictDataset(Dataset):
    def __init__(self, paths: List[str], ref_paths: List[str], transform: Compose) -> None:
        if len(ref_paths) == 1:
            ref_paths = ref_paths * len(paths)
        elif len(paths) != len(ref_paths):
            raise ValueError(
                "Reference image count must be 1 (broadcast to every input) "
                f"or match the input image count; got {len(ref_paths)} reference "
                f"images for {len(paths)} input images."
            )

        self.paths = paths
        self.transform = transform
        self.ref_paths = ref_paths

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        x = _ensure_rgb(read_rgb_image(path), Path(path))

        ref_path = self.ref_paths[idx]
        y = _ensure_rgb(read_rgb_image(ref_path), Path(ref_path))
        
        path = os.path.basename(path)
        
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return path, x, y

    def __len__(self) -> int:
        return len(self.paths)
