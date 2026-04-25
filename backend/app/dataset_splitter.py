"""Dataset split utilities for train/valid/test exports."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DatasetItem:
    """One image and its matching YOLO label file."""

    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test item groups."""

    train: list[DatasetItem]
    valid: list[DatasetItem]
    test: list[DatasetItem]


def split_items(
    items: Sequence[DatasetItem],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    """Shuffle and split dataset items deterministically.

    Any remainder from integer rounding is assigned to the training split so
    the training set remains the largest split for small datasets.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train_count += total - (train_count + val_count + test_count)

    train = shuffled[:train_count]
    valid = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]
    return DatasetSplit(train=train, valid=valid, test=test)


def ensure_yolo_split_dirs(output_dir: Path) -> None:
    """Create YOLO train/valid/test image and label directories."""
    for split_name in ("train", "valid", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)


def _copy_items(items: Sequence[DatasetItem], output_dir: Path, split_name: str) -> None:
    """Copy one split's images and labels into the YOLO directory layout."""
    for item in items:
        image_dest = output_dir / split_name / "images" / item.image_path.name
        label_dest = output_dir / split_name / "labels" / item.label_path.name
        shutil.copy2(item.image_path, image_dest)
        shutil.copy2(item.label_path, label_dest)


def write_split_dataset(
    items: Sequence[DatasetItem],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    """Split items and copy them into a YOLO train/valid/test dataset."""
    dataset_split = split_items(
        items,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    ensure_yolo_split_dirs(output_dir)
    _copy_items(dataset_split.train, output_dir, "train")
    _copy_items(dataset_split.valid, output_dir, "valid")
    _copy_items(dataset_split.test, output_dir, "test")
    return dataset_split
