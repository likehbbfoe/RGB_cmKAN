import random
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToImageTensor,
)

from .img_dataset import ALIGNED_PAIRING_MODES, UnpairedImageDataset


DEFAULT_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


class WeakAlignedTrainTransform:
    """Apply shared normalized crop positions and flips to a rough image pair."""

    def __init__(
        self,
        resize_size: int,
        crop_size: int,
        horizontal_flip_probability: float,
        vertical_flip_probability: float,
    ) -> None:
        self.prepare = Compose([
            ToImageTensor(),
            Resize(resize_size, antialias=True),
            ConvertImageDtype(dtype=torch.float32),
        ])
        self.crop_size = crop_size
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability

    @staticmethod
    def _crop_offset(length: int, crop_size: int, position: float) -> int:
        available = length - crop_size
        if available < 0:
            raise ValueError(
                f"Cannot crop size {crop_size} from resized dimension {length}"
            )
        return round(position * available)

    def _crop(self, image, vertical_position, horizontal_position):
        top = self._crop_offset(
            image.shape[-2],
            self.crop_size,
            vertical_position,
        )
        left = self._crop_offset(
            image.shape[-1],
            self.crop_size,
            horizontal_position,
        )
        return image[
            ...,
            top:top + self.crop_size,
            left:left + self.crop_size,
        ]

    def __call__(self, source, target):
        source = self.prepare(source)
        target = self.prepare(target)
        random_values = torch.rand(4).tolist()
        vertical_position, horizontal_position = random_values[:2]
        source = self._crop(
            source,
            vertical_position,
            horizontal_position,
        )
        target = self._crop(
            target,
            vertical_position,
            horizontal_position,
        )

        if random_values[2] < self.horizontal_flip_probability:
            source = source.flip(-1)
            target = target.flip(-1)
        if random_values[3] < self.vertical_flip_probability:
            source = source.flip(-2)
            target = target.flip(-2)
        return source, target


def _find_images(
    directory: str,
    extensions: Tuple[str, ...],
    recursive: bool,
) -> list[Path]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: '{root}'")

    normalized_extensions = {extension.lower() for extension in extensions}
    candidates = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )
    if not paths:
        raise ValueError(f"No supported images found in '{root}'")
    return paths


def _split_paths(
    paths: list[Path],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = paths.copy()
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * val_fraction))
    test_count = max(1, round(len(shuffled) * test_fraction))
    train_count = len(shuffled) - val_count - test_count
    if train_count < 1:
        raise ValueError(
            "Each domain needs enough images to create non-empty train, val, and "
            f"test splits; got {len(shuffled)} images"
        )

    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]
    return train, val, test


class CustomUnpairedDataModule(L.LightningDataModule):
    """Data module for a root containing independent source and target domains."""

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        val_source_dir: str | None = None,
        val_target_dir: str | None = None,
        test_source_dir: str | None = None,
        test_target_dir: str | None = None,
        batch_size: int = 2,
        val_batch_size: int = 2,
        test_batch_size: int = 2,
        crop_size: int = 256,
        resize_size: int = 286,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.0,
        num_workers: int = 4,
        recursive: bool = True,
        pair_by_subdirectory: bool = False,
        seed: int = 42,
        image_extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
        pairing_mode: str = "random",
    ) -> None:
        super().__init__()
        if resize_size < crop_size:
            raise ValueError("resize_size must be greater than or equal to crop_size")
        if val_fraction <= 0 or test_fraction <= 0:
            raise ValueError("val_fraction and test_fraction must both be greater than 0")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be less than 1")

        source_root = Path(source_dir).expanduser()
        target_root = Path(target_dir).expanduser()
        source_paths = _find_images(source_dir, image_extensions, recursive)
        target_paths = _find_images(target_dir, image_extensions, recursive)

        has_val_source = val_source_dir is not None
        has_val_target = val_target_dir is not None
        if has_val_source != has_val_target:
            raise ValueError(
                "val_source_dir and val_target_dir must be provided together"
            )

        has_test_source = test_source_dir is not None
        has_test_target = test_target_dir is not None
        if has_test_source != has_test_target:
            raise ValueError(
                "test_source_dir and test_target_dir must be provided together"
            )

        self.pairing_mode = getattr(pairing_mode, "value", pairing_mode)
        valid_pairing_modes = ("random", *ALIGNED_PAIRING_MODES)
        if self.pairing_mode not in valid_pairing_modes:
            raise ValueError(
                "pairing_mode must be 'random', 'weak_aligned', or 'one_to_one', "
                f"got '{self.pairing_mode}'"
            )
        if self.pairing_mode in ALIGNED_PAIRING_MODES and not has_val_source:
            raise ValueError(
                "Aligned pairing requires explicit val_source_dir and "
                "val_target_dir so pairs are not split independently"
            )

        if has_val_source:
            val_source_root = Path(val_source_dir).expanduser()
            val_target_root = Path(val_target_dir).expanduser()
            self.train_source_paths = source_paths
            self.train_target_paths = target_paths
            self.val_source_paths = _find_images(
                val_source_dir, image_extensions, recursive
            )
            self.val_target_paths = _find_images(
                val_target_dir, image_extensions, recursive
            )

            if has_test_source:
                test_source_root = Path(test_source_dir).expanduser()
                test_target_root = Path(test_target_dir).expanduser()
                self.test_source_paths = _find_images(
                    test_source_dir, image_extensions, recursive
                )
                self.test_target_paths = _find_images(
                    test_target_dir, image_extensions, recursive
                )
            else:
                # A separate test split is optional. Reusing val keeps the
                # test command usable without leaking validation into train.
                self.test_source_paths = self.val_source_paths.copy()
                self.test_target_paths = self.val_target_paths.copy()
                test_source_root = val_source_root
                test_target_root = val_target_root
        elif has_test_source:
            raise ValueError("Explicit test directories require explicit val directories")
        else:
            val_source_root = source_root
            val_target_root = target_root
            test_source_root = source_root
            test_target_root = target_root
            (
                self.train_source_paths,
                self.val_source_paths,
                self.test_source_paths,
            ) = _split_paths(source_paths, val_fraction, test_fraction, seed)
            (
                self.train_target_paths,
                self.val_target_paths,
                self.test_target_paths,
            ) = _split_paths(target_paths, val_fraction, test_fraction, seed + 1)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pair_by_subdirectory = pair_by_subdirectory
        self.train_source_root = source_root
        self.train_target_root = target_root
        self.val_source_root = val_source_root
        self.val_target_root = val_target_root
        self.test_source_root = test_source_root
        self.test_target_root = test_target_root

        # Color jitter is intentionally omitted: the task is to learn the
        # source/target color distributions, so color-changing augmentation
        # would alter the supervision signal.
        self.train_transform = Compose([
            ToImageTensor(),
            Resize(resize_size, antialias=True),
            RandomCrop((crop_size, crop_size)),
            RandomHorizontalFlip(p=horizontal_flip_probability),
            RandomVerticalFlip(p=vertical_flip_probability),
            ConvertImageDtype(dtype=torch.float32),
        ])
        self.aligned_train_transform = WeakAlignedTrainTransform(
            resize_size=resize_size,
            crop_size=crop_size,
            horizontal_flip_probability=horizontal_flip_probability,
            vertical_flip_probability=vertical_flip_probability,
        )
        self.weak_aligned_train_transform = self.aligned_train_transform
        self.eval_transform = Compose([
            ToImageTensor(),
            Resize(resize_size, antialias=True),
            CenterCrop((crop_size, crop_size)),
            ConvertImageDtype(dtype=torch.float32),
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = UnpairedImageDataset(
                self.train_source_paths,
                self.train_target_paths,
                transform=self.train_transform,
                random_pairing=True,
                pair_by_subdirectory=self.pair_by_subdirectory,
                source_root=self.train_source_root,
                target_root=self.train_target_root,
                pairing_mode=self.pairing_mode,
                paired_transform=(
                    self.aligned_train_transform
                    if self.pairing_mode in ALIGNED_PAIRING_MODES
                    else None
                ),
            )
            self.val_dataset = UnpairedImageDataset(
                self.val_source_paths,
                self.val_target_paths,
                transform=self.eval_transform,
                random_pairing=False,
                pair_by_subdirectory=self.pair_by_subdirectory,
                source_root=self.val_source_root,
                target_root=self.val_target_root,
                pairing_mode=self.pairing_mode,
            )
        if stage in ("test", None):
            self.test_dataset = UnpairedImageDataset(
                self.test_source_paths,
                self.test_target_paths,
                transform=self.eval_transform,
                random_pairing=False,
                pair_by_subdirectory=self.pair_by_subdirectory,
                source_root=self.test_source_root,
                target_root=self.test_target_root,
                pairing_mode=self.pairing_mode,
            )

    def _loader(
        self,
        dataset: UnpairedImageDataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, self.test_batch_size, shuffle=False)
