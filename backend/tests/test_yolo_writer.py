"""Tests for YOLO object-detection output generation."""

from pathlib import Path

from app.yolo_writer import prediction_to_yolo_row, write_data_yaml, write_label_file


def test_prediction_to_yolo_row_normalizes_roboflow_detection() -> None:
    """Roboflow pixel coordinates are converted into normalized YOLO values."""
    prediction = {
        "class_id": 5,
        "x": 1587.5,
        "y": 1421.5,
        "width": 2399,
        "height": 2441,
    }

    row = prediction_to_yolo_row(prediction, image_width=4394, image_height=3352)

    assert row == "5 0.361288 0.424075 0.545972 0.728222"


def test_write_label_file_writes_one_prediction_per_line(tmp_path: Path) -> None:
    """Each detection is written as one YOLO row in a matching txt file."""
    label_path = tmp_path / "labels" / "image.txt"
    predictions = [
        {"class_id": 0, "x": 50, "y": 25, "width": 20, "height": 10},
        {"class_id": 1, "x": 10, "y": 15, "width": 4, "height": 6},
    ]

    write_label_file(label_path, predictions, image_width=100, image_height=50)

    assert label_path.read_text(encoding="utf-8") == (
        "0 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.100000 0.300000 0.040000 0.120000\n"
    )


def test_write_data_yaml_uses_yolo_detection_layout(tmp_path: Path) -> None:
    """The data.yaml file references train, valid, and test image folders."""
    write_data_yaml(tmp_path, ["bus", "truck"])

    assert (tmp_path / "data.yaml").read_text(encoding="utf-8") == (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 2\n"
        "names: ['bus', 'truck']\n"
    )
