import inspect
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pytest
import torch

from cm_kan.cli.train import _domain_path
from cm_kan.cli.custom_unpaired import override_data_root
from cm_kan.core.config.data import CustomUnpairedDataParams, PairingMode
from cm_kan.ml.datasets.custom_unpaired import CustomUnpairedDataModule
from cm_kan.ml.datasets.custom_unpaired.img_datamodule import (
    WeakAlignedTrainTransform,
)
from cm_kan.ml.datasets.custom_unpaired.img_dataset import (
    UnpairedImageDataset,
    _ensure_rgb,
)


def _write_images(directory: Path, count: int, offset: int) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        value = (offset + index * 10) % 256
        image = np.full((24, 32, 3), value, dtype=np.uint8)
        imageio.imwrite(directory / f"image_{index}.png", image)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((24, 32, 3), value, dtype=np.uint8))


def test_custom_unpaired_data_module_loads_unequal_domains(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    _write_images(source_dir, count=7, offset=0)
    _write_images(target_dir, count=5, offset=100)

    data_module = CustomUnpairedDataModule(
        source_dir=str(source_dir),
        target_dir=str(target_dir),
        batch_size=2,
        val_batch_size=1,
        test_batch_size=1,
        crop_size=16,
        resize_size=20,
        val_fraction=0.2,
        test_fraction=0.2,
        horizontal_flip_probability=0.5,
        num_workers=0,
        seed=7,
    )
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    assert set(batch) == {"source", "target"}
    assert batch["source"].shape == (2, 3, 16, 16)
    assert batch["target"].shape == (2, 3, 16, 16)
    assert batch["source"].dtype == torch.float32
    assert batch["target"].dtype == torch.float32
    assert 0 <= batch["source"].min() <= batch["source"].max() <= 1
    assert 0 <= batch["target"].min() <= batch["target"].max() <= 1

    first_eval_sample = data_module.val_dataset[0]
    second_eval_sample = data_module.val_dataset[0]
    assert torch.equal(first_eval_sample["source"], second_eval_sample["source"])
    assert torch.equal(first_eval_sample["target"], second_eval_sample["target"])


def test_uint16_images_are_normalized_before_torchvision(tmp_path: Path) -> None:
    image = np.full((8, 10, 3), 65535, dtype=np.uint16)
    normalized = _ensure_rgb(image, tmp_path / "16_bit.tiff")

    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert normalized.min() == normalized.max() == 1.0


def test_explicit_validation_split_is_used_without_resplitting(tmp_path: Path) -> None:
    train_source = tmp_path / "train" / "source"
    train_target = tmp_path / "train" / "target"
    val_source = tmp_path / "val" / "source"
    val_target = tmp_path / "val" / "target"
    _write_images(train_source, count=7, offset=0)
    _write_images(train_target, count=5, offset=100)
    _write_images(val_source, count=2, offset=20)
    _write_images(val_target, count=3, offset=120)

    data_module = CustomUnpairedDataModule(
        source_dir=str(train_source),
        target_dir=str(train_target),
        val_source_dir=str(val_source),
        val_target_dir=str(val_target),
        crop_size=16,
        resize_size=20,
        num_workers=0,
    )

    assert len(data_module.train_source_paths) == 7
    assert len(data_module.train_target_paths) == 5
    assert len(data_module.val_source_paths) == 2
    assert len(data_module.val_target_paths) == 3
    assert data_module.test_source_paths == data_module.val_source_paths
    assert data_module.test_target_paths == data_module.val_target_paths


def test_data_root_layout_prefers_train_real_directory(tmp_path: Path) -> None:
    train_domain = tmp_path / "train" / "samsung"
    (train_domain / "real").mkdir(parents=True)
    (train_domain / "recolor").mkdir()
    val_domain = tmp_path / "val" / "samsung"
    val_domain.mkdir(parents=True)

    assert _domain_path(str(tmp_path), "train", "samsung") == str(
        train_domain / "real"
    )
    assert _domain_path(str(tmp_path), "val", "samsung") == str(val_domain)


def test_data_root_override_uses_explicit_val_and_optional_test(tmp_path: Path) -> None:
    for split in ("train", "val"):
        (tmp_path / split / "source").mkdir(parents=True)
        (tmp_path / split / "target").mkdir(parents=True)

    config = {
        "data": {
            "type": "custom_unpaired",
            "train": {"source": "old/source", "target": "old/target"},
            "test": {"source": "old/test/source", "target": "old/test/target"},
        }
    }

    override_data_root(config, str(tmp_path), "source", "target")

    assert config["data"]["train"] == {
        "source": str(tmp_path / "train" / "source"),
        "target": str(tmp_path / "train" / "target"),
    }
    assert config["data"]["val"] == {
        "source": str(tmp_path / "val" / "source"),
        "target": str(tmp_path / "val" / "target"),
    }
    assert "test" not in config["data"]


def test_grouped_sampling_never_crosses_scene_subdirectories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_images(source_root / "indoor", count=2, offset=10)
    _write_images(source_root / "outdoor", count=2, offset=30)
    _write_images(target_root / "indoor", count=3, offset=100)
    _write_images(target_root / "outdoor", count=3, offset=200)

    dataset = UnpairedImageDataset(
        source_paths=sorted(source_root.rglob("*.png")),
        target_paths=sorted(target_root.rglob("*.png")),
        transform=lambda image: torch.from_numpy(image.copy()),
        random_pairing=True,
        pair_by_subdirectory=True,
        source_root=source_root,
        target_root=target_root,
    )

    for index in range(len(dataset.source_paths)):
        source_group = dataset.source_groups[index]
        for _ in range(10):
            target_mean = dataset[index]["target"].float().mean().item()
            if source_group == "indoor":
                assert 100 <= target_mean <= 120
            else:
                assert 200 <= target_mean <= 220


def test_grouped_sampling_rejects_mismatched_scene_directories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_images(source_root / "indoor", count=1, offset=10)
    _write_images(target_root / "outdoor", count=1, offset=200)

    with pytest.raises(ValueError, match="matching relative subdirectories"):
        UnpairedImageDataset(
            source_paths=sorted(source_root.rglob("*.png")),
            target_paths=sorted(target_root.rglob("*.png")),
            transform=lambda image: torch.from_numpy(image.copy()),
            random_pairing=True,
            pair_by_subdirectory=True,
            source_root=source_root,
            target_root=target_root,
        )


def test_weak_aligned_prefers_complete_relative_stem_matches(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_image(source_root / "scene" / "first.png", 10)
    _write_image(source_root / "scene" / "second.png", 20)
    _write_image(target_root / "scene" / "first.bmp", 110)
    _write_image(target_root / "scene" / "second.bmp", 120)

    source_paths = [
        source_root / "scene" / "first.png",
        source_root / "scene" / "second.png",
    ]
    target_paths = [
        target_root / "scene" / "second.bmp",
        target_root / "scene" / "first.bmp",
    ]
    dataset = UnpairedImageDataset(
        source_paths=source_paths,
        target_paths=target_paths,
        transform=lambda image: torch.from_numpy(image.copy()),
        random_pairing=True,
        source_root=source_root,
        target_root=target_root,
        pairing_mode="weak_aligned",
    )

    assert dataset.weak_aligned_target_indices == (1, 0)
    assert dataset[0]["target"].float().mean().item() == 110
    assert dataset[1]["target"].float().mean().item() == 120


def test_weak_aligned_maps_unequal_groups_by_normalized_position(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_images(source_root / "indoor", count=4, offset=10)
    _write_images(source_root / "outdoor", count=2, offset=50)
    _write_images(target_root / "indoor", count=3, offset=100)
    _write_images(target_root / "outdoor", count=1, offset=200)

    dataset = UnpairedImageDataset(
        source_paths=sorted(source_root.rglob("*.png")),
        target_paths=sorted(target_root.rglob("*.png")),
        transform=lambda image: torch.from_numpy(image.copy()),
        random_pairing=True,
        source_root=source_root,
        target_root=target_root,
        pairing_mode="weak_aligned",
    )

    paired_target_names = [
        dataset.target_paths[dataset._target_index(index, index)].relative_to(
            target_root
        ).as_posix()
        for index in range(len(dataset))
    ]
    assert paired_target_names == [
        "indoor/image_0.png",
        "indoor/image_1.png",
        "indoor/image_1.png",
        "indoor/image_2.png",
        "outdoor/image_0.png",
        "outdoor/image_0.png",
    ]
    assert len(dataset) == 6


def test_weak_aligned_fallback_uses_natural_numeric_order(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    for name, value in (("source_1.png", 1), ("source_2.png", 2), ("source_10.png", 10)):
        _write_image(source_root / name, value)
    for name, value in (("target_1.png", 101), ("target_2.png", 102), ("target_10.png", 110)):
        _write_image(target_root / name, value)

    dataset = UnpairedImageDataset(
        source_paths=sorted(source_root.glob("*.png")),
        target_paths=sorted(target_root.glob("*.png")),
        transform=lambda image: torch.from_numpy(image.copy()),
        random_pairing=True,
        source_root=source_root,
        target_root=target_root,
        pairing_mode="weak_aligned",
    )

    target_by_source = {
        path.name: dataset.target_paths[
            dataset.weak_aligned_target_indices[index]
        ].name
        for index, path in enumerate(dataset.source_paths)
    }
    assert target_by_source == {
        "source_1.png": "target_1.png",
        "source_2.png": "target_2.png",
        "source_10.png": "target_10.png",
    }


def test_weak_aligned_rejects_mismatched_scene_groups(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write_images(source_root / "indoor", count=2, offset=10)
    _write_images(target_root / "outdoor", count=1, offset=100)

    with pytest.raises(ValueError, match="matching relative subdirectories"):
        UnpairedImageDataset(
            source_paths=sorted(source_root.rglob("*.png")),
            target_paths=sorted(target_root.rglob("*.png")),
            transform=lambda image: torch.from_numpy(image.copy()),
            random_pairing=True,
            source_root=source_root,
            target_root=target_root,
            pairing_mode="weak_aligned",
        )


def test_weak_aligned_train_geometry_is_synchronized(tmp_path: Path) -> None:
    train_source = tmp_path / "train" / "source"
    train_target = tmp_path / "train" / "target"
    val_source = tmp_path / "val" / "source"
    val_target = tmp_path / "val" / "target"

    gradient = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    for directory in (train_source, train_target, val_source, val_target):
        directory.mkdir(parents=True)
        imageio.imwrite(directory / "same.png", gradient)

    data_module = CustomUnpairedDataModule(
        source_dir=str(train_source),
        target_dir=str(train_target),
        val_source_dir=str(val_source),
        val_target_dir=str(val_target),
        crop_size=16,
        resize_size=20,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
        num_workers=0,
        pairing_mode="weak_aligned",
    )
    data_module.setup("fit")

    train_sample = data_module.train_dataset[0]
    assert torch.equal(train_sample["source"], train_sample["target"])
    first_val_sample = data_module.val_dataset[0]
    second_val_sample = data_module.val_dataset[0]
    assert torch.equal(first_val_sample["source"], first_val_sample["target"])
    assert torch.equal(first_val_sample["source"], second_val_sample["source"])


def test_weak_aligned_transform_supports_different_aspect_ratios() -> None:
    transform = WeakAlignedTrainTransform(
        resize_size=20,
        crop_size=16,
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
    )
    source = np.full((24, 48, 3), 80, dtype=np.uint8)
    target = np.full((48, 24, 3), 120, dtype=np.uint8)

    transformed_source, transformed_target = transform(source, target)

    assert transformed_source.shape == (3, 16, 16)
    assert transformed_target.shape == (3, 16, 16)
    assert transform._crop_offset(40, 16, 0.25) == 6
    assert transform._crop_offset(80, 16, 0.25) == 16


def test_pairing_mode_config_accepts_weak_aligned() -> None:
    params = CustomUnpairedDataParams(pairing_mode="weak_aligned")

    assert params.pairing_mode is PairingMode.weak_aligned


def test_pairing_mode_is_appended_to_data_module_signature() -> None:
    parameter_names = list(
        inspect.signature(CustomUnpairedDataModule).parameters
    )

    assert parameter_names[-1] == "pairing_mode"
