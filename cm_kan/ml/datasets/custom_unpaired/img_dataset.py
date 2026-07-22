from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cm_kan.ml.utils.io import read_rgb_image


ImageTransform = Callable[[np.ndarray], torch.Tensor]


def _relative_parent(path: Path, root: Path) -> str:
    """Return a stable scene-group key relative to a domain root."""
    try:
        return path.relative_to(root).parent.as_posix()
    except ValueError as exc:
        raise ValueError(f"Image path '{path}' is not inside domain root '{root}'") from exc


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
        pair_by_subdirectory: bool = False,
        source_root: Path | None = None,
        target_root: Path | None = None,
    ) -> None:
        if not source_paths:
            raise ValueError("source_paths must contain at least one image")
        if not target_paths:
            raise ValueError("target_paths must contain at least one image")

        self.source_paths = tuple(source_paths)
        self.target_paths = tuple(target_paths)
        self.transform = transform
        self.random_pairing = random_pairing
        self.pair_by_subdirectory = pair_by_subdirectory

        self.source_groups = None
        self.target_indices_by_group = None
        if pair_by_subdirectory:
            if source_root is None or target_root is None:
                raise ValueError(
                    "source_root and target_root are required for grouped pairing"
                )
            source_root = Path(source_root).expanduser()
            target_root = Path(target_root).expanduser()
            self.source_groups = tuple(
                _relative_parent(path, source_root) for path in self.source_paths
            )
            target_groups = tuple(
                _relative_parent(path, target_root) for path in self.target_paths
            )

            source_group_names = set(self.source_groups)
            target_group_names = set(target_groups)
            if source_group_names != target_group_names:
                source_only = sorted(source_group_names - target_group_names)
                target_only = sorted(target_group_names - source_group_names)
                raise ValueError(
                    "Grouped pairing requires matching relative subdirectories; "
                    f"source-only groups={source_only}, target-only groups={target_only}"
                )

            target_indices_by_group = {}
            for target_index, group in enumerate(target_groups):
                target_indices_by_group.setdefault(group, []).append(target_index)
            self.target_indices_by_group = {
                group: tuple(indices)
                for group, indices in target_indices_by_group.items()
            }

    def _target_index(self, index: int, source_index: int) -> int:
        if self.pair_by_subdirectory:
            group = self.source_groups[source_index]
            candidates = self.target_indices_by_group[group]
            if self.random_pairing:
                candidate_index = torch.randint(len(candidates), size=()).item()
                return candidates[candidate_index]
            return candidates[index % len(candidates)]

        if self.random_pairing:
            return torch.randint(len(self.target_paths), size=()).item()
        return index % len(self.target_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = index % len(self.source_paths)
        source_path = self.source_paths[source_index]
        target_index = self._target_index(index, source_index)
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
