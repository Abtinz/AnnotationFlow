"""Tests for deterministic YOLO dataset splitting."""

from pathlib import Path

from app.dataset_splitter import DatasetItem, split_items, write_split_dataset


def test_split_items_uses_ratios_and_assigns_remainder_to_train(tmp_path: Path) -> None:
    """Split counts follow configured ratios, with leftovers going to train."""
    items = [
        DatasetItem(image_path=tmp_path / f"{index}.jpg", label_path=tmp_path / f"{index}.txt")
        for index in range(11)
    ]

    split = split_items(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(split.train) == 9
    assert len(split.valid) == 1
    assert len(split.test) == 1


def test_write_split_dataset_copies_images_and_labels(tmp_path: Path) -> None:
    """The writer creates YOLO train/valid/test image and label directories."""
    source = tmp_path / "source"
    source.mkdir()
    items = []
    for index in range(3):
        image = source / f"image_{index}.jpg"
        label = source / f"image_{index}.txt"
        image.write_bytes(f"image {index}".encode("utf-8"))
        label.write_text(f"0 0.{index} 0.5 0.1 0.1\n", encoding="utf-8")
        items.append(DatasetItem(image_path=image, label_path=label))

    output_dir = tmp_path / "dataset"
    write_split_dataset(items, output_dir, train_ratio=0.34, val_ratio=0.0, test_ratio=0.66, seed=1)

    assert len(list((output_dir / "train" / "images").iterdir())) == 2
    assert len(list((output_dir / "train" / "labels").iterdir())) == 2
    assert len(list((output_dir / "valid" / "images").iterdir())) == 0
    assert len(list((output_dir / "test" / "images").iterdir())) == 1
