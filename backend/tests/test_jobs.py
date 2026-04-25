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


class FakeRoboflowPredictionsClient:
    """Fake client matching the actual find-bus workflow response shape."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Return detections under the `predictions` key."""
        return [
            {
                "predictions": {
                    "image": {"width": 100, "height": 50},
                    "predictions": [
                        {
                            "class_id": 0,
                            "class": "bus",
                            "x": 50,
                            "y": 25,
                            "width": 20,
                            "height": 10,
                        }
                    ],
                }
            }
        ]


class PartiallyFailingRoboflowClient:
    """Fake client that raises for one image and succeeds for another."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Raise a transient workflow error for the selected failure image."""
        if image_path.name == "fail.jpg":
            raise RuntimeError("Roboflow 502 Bad Gateway")
        return [
            {
                "predictions": {
                    "image": {"width": 100, "height": 50},
                    "predictions": [
                        {
                            "class_id": 0,
                            "class": "bus",
                            "x": 50,
                            "y": 25,
                            "width": 20,
                            "height": 10,
                        }
                    ],
                }
            }
        ]


class NullMetadataRoboflowClient:
    """Fake client returning detections with null image dimensions."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Return a response shape seen from external workflow edge cases."""
        return [
            {
                "predictions": {
                    "image": {"width": None, "height": None},
                    "predictions": [
                        {
                            "class_id": 0,
                            "class": "bus",
                            "x": 50,
                            "y": 25,
                            "width": 20,
                            "height": 10,
                        }
                    ],
                }
            }
        ]


class EmptyRoboflowClient:
    """Fake client returning no detections for every image."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Return a Roboflow-like empty detection response."""
        return [{"predictions": {"image": {"width": 100, "height": 50}, "predictions": []}}]


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


def test_process_image_paths_continues_after_workflow_error(tmp_path: Path) -> None:
    """One Roboflow failure should not fail the entire batch job."""
    ok_image = tmp_path / "ok.png"
    fail_image = tmp_path / "fail.png"
    Image.new("RGB", (100, 50), "yellow").save(ok_image)
    Image.new("RGB", (100, 50), "red").save(fail_image)
    job_dir = tmp_path / "job"
    config = PipelineConfig(
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        split_seed=42,
        include_empty_labels=False,
    )

    summary = process_image_paths(
        image_paths=[ok_image, fail_image],
        job_dir=job_dir,
        roboflow_client=PartiallyFailingRoboflowClient(),
        config=config,
    )

    assert summary["processed_images"] == 1
    assert summary["failed_images"] == 1
    assert summary["classes"] == ["bus"]
    assert (job_dir / "dataset" / "train" / "labels" / "ok.txt").read_text(
        encoding="utf-8"
    ) == "0 0.500000 0.500000 0.200000 0.200000\n"
    assert not (job_dir / "dataset" / "train" / "labels" / "fail.txt").exists()
    assert "Roboflow workflow failed" in (job_dir / "logs.jsonl").read_text(encoding="utf-8")
    assert "Roboflow 502 Bad Gateway" in (job_dir / "roboflow" / "raw_results.jsonl").read_text(
        encoding="utf-8"
    )


def test_process_image_paths_uses_image_size_when_workflow_metadata_is_null(tmp_path: Path) -> None:
    """Null Roboflow image dimensions should fall back to the normalized image size."""
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
        roboflow_client=NullMetadataRoboflowClient(),
        config=config,
    )

    assert summary["processed_images"] == 1
    assert summary["failed_images"] == 0
    assert (job_dir / "dataset" / "train" / "labels" / "bus.txt").read_text(
        encoding="utf-8"
    ) == "0 0.500000 0.500000 0.200000 0.200000\n"


def test_process_image_paths_counts_images_skipped_without_detections(tmp_path: Path) -> None:
    """Images with no detections are counted separately from workflow failures."""
    image_path = tmp_path / "empty.png"
    Image.new("RGB", (100, 50), "white").save(image_path)
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
        roboflow_client=EmptyRoboflowClient(),
        config=config,
    )

    assert summary["processed_images"] == 0
    assert summary["failed_images"] == 0
    assert summary["skipped_images"] == 1
    assert "Skipped image with no detections" in (job_dir / "logs.jsonl").read_text(encoding="utf-8")
    assert (job_dir / "dataset" / "data.yaml").read_text(encoding="utf-8") == (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        "names: ['object']\n"
    )
    assert (job_dir / "logs.jsonl").exists()
    assert (job_dir / "roboflow" / "raw_results.jsonl").exists()


def test_process_image_paths_accepts_predictions_key_from_workflow(tmp_path: Path) -> None:
    """The real find-bus workflow shape is parsed into YOLO labels."""
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
        roboflow_client=FakeRoboflowPredictionsClient(),
        config=config,
    )

    assert summary["processed_images"] == 1
    assert summary["classes"] == ["bus"]
    assert (job_dir / "dataset" / "train" / "labels" / "bus.txt").read_text(
        encoding="utf-8"
    ) == "0 0.500000 0.500000 0.200000 0.200000\n"
