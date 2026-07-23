from collections.abc import Callable, Sequence
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from cm_kan.ml.utils.io import read_rgb_image


ImageTransform = Callable[[np.ndarray], torch.Tensor]
PairedImageTransform = Callable[
    [np.ndarray, np.ndarray],
    tuple[torch.Tensor, torch.Tensor],
]
ALIGNED_PAIRING_MODES = ("weak_aligned", "one_to_one")


def _natural_path_key(path: Path) -> tuple[tuple[bool, object], ...]:
    """Sort numbered filenames as 2 before 10 instead of lexicographically."""
    return tuple(
        (part.isdigit(), int(part) if part.isdigit() else part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
        if part
    )


def _relative_parent(path: Path, root: Path) -> str:
    """Return a stable scene-group key relative to a domain root."""
    try:
        return path.relative_to(root).parent.as_posix()
    except ValueError as exc:
        raise ValueError(f"Image path '{path}' is not inside domain root '{root}'") from exc


def _relative_stem(path: Path, root: Path) -> str:
    """Return a relative path key without the image extension."""
    try:
        return path.relative_to(root).with_suffix("").as_posix()
    except ValueError as exc:
        raise ValueError(f"Image path '{path}' is not inside domain root '{root}'") from exc


def _build_weak_aligned_target_indices(
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    source_root: Path,
    target_root: Path,
) -> tuple[int, ...]:
    """Build a deterministic source-to-target map for approximately paired data.

    A complete relative-stem match is preferred. Otherwise, images are grouped
    by relative subdirectory and monotonically mapped by normalized position.
    The latter supports groups with unequal source and target counts without
    silently crossing scene boundaries.
    """
    source_keys = tuple(_relative_stem(path, source_root) for path in source_paths)
    target_keys = tuple(_relative_stem(path, target_root) for path in target_paths)
    target_index_by_key = {
        key: index for index, key in enumerate(target_keys)
    }
    has_unique_keys = (
        len(set(source_keys)) == len(source_keys)
        and len(target_index_by_key) == len(target_keys)
    )
    if (
        has_unique_keys
        and len(source_keys) == len(target_keys)
        and set(source_keys) == set(target_keys)
    ):
        return tuple(target_index_by_key[key] for key in source_keys)

    source_by_group: dict[str, list[tuple[int, Path]]] = {}
    target_by_group: dict[str, list[tuple[int, Path]]] = {}
    for index, path in enumerate(source_paths):
        group = _relative_parent(path, source_root)
        source_by_group.setdefault(group, []).append((index, path))
    for index, path in enumerate(target_paths):
        group = _relative_parent(path, target_root)
        target_by_group.setdefault(group, []).append((index, path))

    source_groups = set(source_by_group)
    target_groups = set(target_by_group)
    if source_groups != target_groups:
        source_only = sorted(source_groups - target_groups)
        target_only = sorted(target_groups - source_groups)
        raise ValueError(
            "weak_aligned pairing requires matching relative subdirectories; "
            f"source-only groups={source_only}, target-only groups={target_only}"
        )

    target_indices = [-1] * len(source_paths)
    for group in sorted(source_groups):
        sources = sorted(
            source_by_group[group],
            key=lambda item: _natural_path_key(item[1]),
        )
        targets = sorted(
            target_by_group[group],
            key=lambda item: _natural_path_key(item[1]),
        )
        source_count = len(sources)
        target_count = len(targets)

        for local_index, (source_index, _) in enumerate(sources):
            if source_count == 1 or target_count == 1:
                target_local_index = 0
            else:
                target_local_index = round(
                    local_index * (target_count - 1) / (source_count - 1)
                )
            target_indices[source_index] = targets[target_local_index][0]

    if any(index < 0 for index in target_indices):
        raise RuntimeError("Failed to construct a complete weak-aligned pairing map")
    return tuple(target_indices)


def _build_one_to_one_target_indices(
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    source_root: Path,
    target_root: Path,
) -> tuple[int, ...]:
    """Build a deterministic bijection between approximately paired domains.

    Matching relative stems are preferred. If the names differ, each matching
    relative subdirectory is naturally sorted and zipped. Unlike
    ``weak_aligned``, this mode never repeats or drops a target image.
    """
    if len(source_paths) != len(target_paths):
        raise ValueError(
            "one_to_one pairing requires equal source and target counts; "
            f"got source={len(source_paths)}, target={len(target_paths)}"
        )

    source_keys = tuple(_relative_stem(path, source_root) for path in source_paths)
    target_keys = tuple(_relative_stem(path, target_root) for path in target_paths)
    target_index_by_key = {
        key: index for index, key in enumerate(target_keys)
    }
    has_unique_keys = (
        len(set(source_keys)) == len(source_keys)
        and len(target_index_by_key) == len(target_keys)
    )
    if has_unique_keys and set(source_keys) == set(target_keys):
        return tuple(target_index_by_key[key] for key in source_keys)

    source_by_group: dict[str, list[tuple[int, Path]]] = {}
    target_by_group: dict[str, list[tuple[int, Path]]] = {}
    for index, path in enumerate(source_paths):
        group = _relative_parent(path, source_root)
        source_by_group.setdefault(group, []).append((index, path))
    for index, path in enumerate(target_paths):
        group = _relative_parent(path, target_root)
        target_by_group.setdefault(group, []).append((index, path))

    source_groups = set(source_by_group)
    target_groups = set(target_by_group)
    if source_groups != target_groups:
        source_only = sorted(source_groups - target_groups)
        target_only = sorted(target_groups - source_groups)
        raise ValueError(
            "one_to_one pairing requires matching relative subdirectories; "
            f"source-only groups={source_only}, target-only groups={target_only}"
        )

    target_indices = [-1] * len(source_paths)
    for group in sorted(source_groups):
        sources = sorted(
            source_by_group[group],
            key=lambda item: _natural_path_key(item[1]),
        )
        targets = sorted(
            target_by_group[group],
            key=lambda item: _natural_path_key(item[1]),
        )
        if len(sources) != len(targets):
            raise ValueError(
                "one_to_one pairing requires equal counts inside every relative "
                f"subdirectory; group='{group}', source={len(sources)}, "
                f"target={len(targets)}"
            )
        for (source_index, _), (target_index, _) in zip(sources, targets):
            target_indices[source_index] = target_index

    expected_target_indices = list(range(len(target_paths)))
    if sorted(target_indices) != expected_target_indices:
        raise RuntimeError(
            "Failed to construct a complete one-to-one source/target pairing"
        )
    return tuple(target_indices)


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
        pairing_mode: str = "random",
        paired_transform: PairedImageTransform | None = None,
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
        self.pairing_mode = getattr(pairing_mode, "value", pairing_mode)
        self.paired_transform = paired_transform
        valid_pairing_modes = ("random", *ALIGNED_PAIRING_MODES)
        if self.pairing_mode not in valid_pairing_modes:
            raise ValueError(
                "pairing_mode must be 'random', 'weak_aligned', or 'one_to_one', "
                f"got '{self.pairing_mode}'"
            )

        expanded_source_root = (
            Path(source_root).expanduser() if source_root is not None else None
        )
        expanded_target_root = (
            Path(target_root).expanduser() if target_root is not None else None
        )
        self.aligned_target_indices = None
        if self.pairing_mode in ALIGNED_PAIRING_MODES:
            if expanded_source_root is None or expanded_target_root is None:
                raise ValueError(
                    "source_root and target_root are required for aligned pairing"
                )
            index_builder = (
                _build_one_to_one_target_indices
                if self.pairing_mode == "one_to_one"
                else _build_weak_aligned_target_indices
            )
            self.aligned_target_indices = index_builder(
                self.source_paths,
                self.target_paths,
                expanded_source_root,
                expanded_target_root,
            )
        # Keep the old internal attribute for downstream code that inspected it.
        self.weak_aligned_target_indices = (
            self.aligned_target_indices
            if self.pairing_mode == "weak_aligned"
            else None
        )

        self.source_groups = None
        self.target_indices_by_group = None
        if pair_by_subdirectory and self.pairing_mode == "random":
            if expanded_source_root is None or expanded_target_root is None:
                raise ValueError(
                    "source_root and target_root are required for grouped pairing"
                )
            self.source_groups = tuple(
                _relative_parent(path, expanded_source_root)
                for path in self.source_paths
            )
            target_groups = tuple(
                _relative_parent(path, expanded_target_root)
                for path in self.target_paths
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
        if self.pairing_mode in ALIGNED_PAIRING_MODES:
            return self.aligned_target_indices[source_index]

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

        if (
            self.pairing_mode in ALIGNED_PAIRING_MODES
            and self.paired_transform is not None
        ):
            transformed_source, transformed_target = self.paired_transform(
                source,
                target,
            )
        else:
            transformed_source = self.transform(source)
            transformed_target = self.transform(target)

        return {
            "source": transformed_source,
            "target": transformed_target,
        }

    def __len__(self) -> int:
        if self.pairing_mode in ALIGNED_PAIRING_MODES:
            return len(self.source_paths)
        return max(len(self.source_paths), len(self.target_paths))
