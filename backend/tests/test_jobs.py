"""Tests for the image-to-YOLO job pipeline."""

from pathlib import Path

from PIL import Image

from app.jobs import PipelineConfig, process_image_paths


class FakeRoboflowClient:
    """Fake Roboflow client returning one bus detection for every image."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Return a Roboflow-like detection response."""
        return [
            {
                "detections": {
                    "image": {"width": 100, "height": 50},
                    "predictions": [
                        {
                            "class_id": 5,
                            "class": "bus",
                            "x": 50,
                            "y": 25,
                            "width": 20,
                            "height": 10,
                        }
                    ],
                },
                "upload_error_status": False,
            }
        ]


def test_process_image_paths_builds_yolo_dataset_with_remapped_classes(tmp_path: Path) -> None:
    """The pipeline normalizes images, calls Roboflow, and writes YOLO output."""
    image_path = tmp_path / "bus.png"
    Image.new("RGB", (100, 50), "yellow").save(image_path)
    job_dir = tmp_path / "job"
    config = PipelineConfig(
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        split_seed=42,
        include_empty_labels=False,
    )

    summary = process_image_paths(
        image_paths=[image_path],
        job_dir=job_dir,
        roboflow_client=FakeRoboflowClient(),
        config=config,
    )

    assert summary["processed_images"] == 1
    assert summary["duplicate_images"] == 0
    assert summary["classes"] == ["bus"]
    assert (job_dir / "dataset" / "train" / "images" / "bus.jpg").exists()
    assert (job_dir / "dataset" / "train" / "labels" / "bus.txt").read_text(
        encoding="utf-8"
    ) == "0 0.500000 0.500000 0.200000 0.200000\n"
    assert (job_dir / "dataset" / "data.yaml").read_text(encoding="utf-8") == (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        "names: ['bus']\n"
    )
    assert (job_dir / "logs.jsonl").exists()
    assert (job_dir / "roboflow" / "raw_results.jsonl").exists()

