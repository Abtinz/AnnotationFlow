"""YOLO object-detection label and data.yaml writers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def _normalized(value: float, divisor: int) -> float:
    """Normalize a pixel value and clamp it into YOLO's expected 0..1 range."""
    if divisor <= 0:
        raise ValueError("image dimensions must be positive")
    return max(0.0, min(1.0, value / divisor))


def prediction_to_yolo_row(
    prediction: Mapping[str, object],
    image_width: int,
    image_height: int,
) -> str:
    """Convert one Roboflow-style detection into a YOLO label row.

    Roboflow object detections use pixel-space center coordinates and box
    dimensions. YOLO detection labels use the same center-based shape after
    normalizing each coordinate by the image width or height.
    """
    class_id = int(prediction["class_id"])
    x_center = _normalized(float(prediction["x"]), image_width)
    y_center = _normalized(float(prediction["y"]), image_height)
    width = _normalized(float(prediction["width"]), image_width)
    height = _normalized(float(prediction["height"]), image_height)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def write_label_file(
    label_path: Path,
    predictions: Sequence[Mapping[str, object]],
    image_width: int,
    image_height: int,
) -> Path:
    """Write a YOLO `.txt` label file for one image."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        prediction_to_yolo_row(prediction, image_width=image_width, image_height=image_height)
        for prediction in predictions
    ]
    content = "".join(f"{row}\n" for row in rows)
    label_path.write_text(content, encoding="utf-8")
    return label_path


def write_data_yaml(
    dataset_dir: Path,
    class_names: Sequence[str],
    train_ref: str = "../train/images",
    val_ref: str = "../valid/images",
    test_ref: str = "../test/images",
) -> Path:
    """Write a YOLOv8-compatible `data.yaml` file."""
    if not class_names:
        raise ValueError("class_names must contain at least one class")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    names_literal = "[" + ", ".join(repr(name) for name in class_names) + "]"
    content = (
        f"train: {train_ref}\n"
        f"val: {val_ref}\n"
        f"test: {test_ref}\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_literal}\n"
    )
    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml_path.write_text(content, encoding="utf-8")
    return data_yaml_path
