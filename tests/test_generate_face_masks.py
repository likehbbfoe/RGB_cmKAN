from pathlib import Path

import numpy as np
from PIL import Image

from scripts.generate_face_masks import (
    ellipse_roi_mask,
    final_skin_mask,
    generate_face_masks,
    mask_output_path,
    select_largest_face,
    summarize_records,
    to_uint8_rgb,
    write_preview,
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_select_largest_face_uses_box_area() -> None:
    boxes = [(1, 2, 12, 8), (4, 5, 10, 10), (0, 0, 5, 5)]

    assert select_largest_face(boxes) == (4, 5, 10, 10)
    assert select_largest_face([]) is None


def test_ellipse_roi_mask_is_single_channel_and_inside_box() -> None:
    mask = ellipse_roi_mask(
        height=40,
        width=50,
        box=(10, 8, 20, 16),
    )

    assert mask.shape == (40, 50)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[16, 20] == 255
    assert mask[0, 0] == 0
    assert mask[8, 10] == 0
    assert not mask[:, :10].any()
    assert not mask[:, 30:].any()


def test_ellipse_roi_mask_is_black_without_detection() -> None:
    mask = ellipse_roi_mask(height=9, width=11, box=None)

    assert mask.shape == (9, 11)
    assert not mask.any()


def test_mask_output_path_mirrors_relative_path_as_png(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "data" / "train" / "source"
    image_path = image_root / "scene_2" / "portrait.jpeg"
    output_root = tmp_path / "face_masks" / "train" / "source"

    assert mask_output_path(image_path, image_root, output_root) == (
        output_root / "scene_2" / "portrait.png"
    )


def test_to_uint8_rgb_supports_grayscale_float_images() -> None:
    grayscale = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    converted = to_uint8_rgb(grayscale)

    assert converted.shape == (1, 3, 3)
    assert converted.dtype == np.uint8
    assert converted[0, 0].tolist() == [0, 0, 0]
    assert converted[0, 2].tolist() == [255, 255, 255]


def test_final_skin_mask_intersects_color_candidate_with_face_roi() -> None:
    image = np.full((12, 16, 3), (32, 64, 180), dtype=np.uint8)
    image[3:9, 4:12] = (158, 105, 71)
    face_roi = np.zeros((12, 16), dtype=np.uint8)
    face_roi[2:10, 3:13] = 255

    mask = final_skin_mask(image, face_roi)

    assert mask[5, 6] == 255
    assert mask[0, 0] == 0
    assert mask[5, 2] == 0


def test_generate_face_masks_mirrors_tree_and_writes_black_misses(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "face_masks"
    image_values = {
        ("train", "source", "scene/a.jpg"): 40,
        ("train", "target", "scene/a.png"): 200,
        ("val", "source", "b.png"): 40,
        ("val", "target", "b.png"): 200,
    }
    for (split, domain, relative_path), value in image_values.items():
        _write_rgb(data_root / split / domain / relative_path, value)

    def fake_detector(image: np.ndarray):
        if int(image.mean()) < 100:
            return []
        return [(1, 1, 4, 4), (6, 4, 18, 16)]

    records = generate_face_masks(
        data_root=data_root,
        output_root=output_root,
        detector=fake_detector,
        splits=("train", "val"),
        domains=("source", "target"),
    )

    assert len(records) == 4
    assert sum(record.detected for record in records) == 2
    missed_mask = np.asarray(
        Image.open(output_root / "train/source/scene/a.png")
    )
    detected_path = output_root / "train/target/scene/a.png"
    detected_image = Image.open(detected_path)
    detected_mask = np.asarray(detected_image)
    assert not missed_mask.any()
    assert detected_image.mode == "L"
    assert detected_mask.any()
    assert detected_mask.shape == (24, 32)
    assert all(record.mask_path.is_file() for record in records)

    summary = summarize_records(records)
    assert "total=4 detected=2 missed=2" in summary
    assert "train/source: total=1 detected=0 missed=1" in summary
    assert "val/target: total=1 detected=1 missed=0" in summary

    preview_path = output_root / "face_mask_preview.png"
    manifest_path = write_preview(
        records,
        preview_path,
        sample_count=4,
        panel_size=64,
    )
    preview = Image.open(preview_path)
    assert preview.mode == "RGB"
    assert preview.width >= 4 * 64
    assert preview.height >= 4 * 64
    manifest = manifest_path.read_text(encoding="utf-8")
    assert manifest_path == preview_path.with_suffix(".tsv")
    assert "row\tsplit\tdomain\tstatus" in manifest
    assert "train\tsource\tMISSED / BLACK MASK" in manifest


def test_generate_face_masks_rejects_data_root_as_output_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    for split in ("train", "val"):
        for domain in ("source", "target"):
            _write_rgb(data_root / split / domain / "same.png", 100)

    try:
        generate_face_masks(
            data_root=data_root,
            output_root=data_root,
            detector=lambda image: [],
        )
    except ValueError as exc:
        assert "different from the data root" in str(exc)
    else:
        raise AssertionError(
            "Using the data root for masks could overwrite input PNG files"
        )


def test_train_real_directory_uses_domain_level_mask_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "face_masks"
    _write_rgb(
        data_root / "train/source/real/scene/portrait.jpg",
        120,
    )

    records = generate_face_masks(
        data_root=data_root,
        output_root=output_root,
        detector=lambda image: [(4, 3, 16, 16)],
        splits=("train",),
        domains=("source",),
    )

    expected = (
        output_root / "train/source/scene/portrait.png"
    ).resolve()
    assert len(records) == 1
    assert records[0].mask_path == expected
    assert expected.is_file()
    assert not (output_root / "train/source/real").exists()


def test_existing_manual_mask_is_preserved_without_overwrite(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "face_masks"
    image_path = data_root / "train/source/portrait.png"
    _write_rgb(image_path, 120)
    mask_path = output_root / "train/source/portrait.png"
    mask_path.parent.mkdir(parents=True)
    manual_mask = np.zeros((24, 32), dtype=np.uint8)
    manual_mask[5:10, 7:14] = 255
    Image.fromarray(manual_mask, mode="L").save(mask_path)

    records = generate_face_masks(
        data_root=data_root,
        output_root=output_root,
        detector=lambda image: [],
        splits=("train",),
        domains=("source",),
    )

    assert records[0].reused is True
    assert records[0].detected is True
    assert np.array_equal(np.asarray(Image.open(mask_path)), manual_mask)
