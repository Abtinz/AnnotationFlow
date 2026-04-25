"""Job orchestration for the image-to-YOLO dataset pipeline."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from app.dataset_splitter import DatasetItem, write_split_dataset
from app.duplicate_detector import file_sha256
from app.image_normalizer import normalize_to_jpg
from app.yolo_writer import write_data_yaml, write_label_file


class ImageWorkflowClient(Protocol):
    """Protocol for a client that can run one image through a workflow."""

    def run_image(self, image_path: Path) -> Any:
        """Run workflow inference for one normalized image."""


@dataclass(frozen=True)
class PipelineConfig:
    """Settings that control dataset building for one job."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_seed: int
    include_empty_labels: bool


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def _log(job_dir: Path, message: str, level: str = "info", **extra: Any) -> None:
    """Write one structured log line for UI polling."""
    _append_jsonl(
        job_dir / "logs.jsonl",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **extra,
        },
    )


def _first_result(workflow_result: Any) -> dict[str, Any]:
    """Return the first Roboflow result object from list or dict responses."""
    if isinstance(workflow_result, list):
        if not workflow_result:
            return {}
        first = workflow_result[0]
        return first if isinstance(first, dict) else {}
    return workflow_result if isinstance(workflow_result, dict) else {}


def _detections_from_result(result: dict[str, Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Extract image metadata and prediction rows from a Roboflow result."""
    detections = result.get("detections", {})
    if not isinstance(detections, dict):
        return {"width": 0, "height": 0}, []

    image = detections.get("image", {})
    predictions = detections.get("predictions", [])
    if not isinstance(image, dict) or not isinstance(predictions, list):
        return {"width": 0, "height": 0}, []

    return (
        {"width": int(image.get("width", 0)), "height": int(image.get("height", 0))},
        [prediction for prediction in predictions if isinstance(prediction, dict)],
    )


def _remap_predictions(
    predictions: Iterable[dict[str, Any]],
    class_to_yolo_id: dict[str, int],
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Map Roboflow class names to contiguous YOLO class ids."""
    remapped: list[dict[str, Any]] = []
    for prediction in predictions:
        class_name = str(prediction.get("class", prediction.get("class_id", "object")))
        if class_name not in class_to_yolo_id:
            class_to_yolo_id[class_name] = len(class_names)
            class_names.append(class_name)
        remapped_prediction = dict(prediction)
        remapped_prediction["class_id"] = class_to_yolo_id[class_name]
        remapped.append(remapped_prediction)
    return remapped


def process_image_paths(
    image_paths: list[Path],
    job_dir: Path,
    roboflow_client: ImageWorkflowClient,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Process images through cleanup, Roboflow inference, and YOLO export."""
    originals_dir = job_dir / "originals"
    normalized_dir = job_dir / "normalized"
    labels_dir = job_dir / "labels"
    dataset_dir = job_dir / "dataset"
    raw_results_path = job_dir / "roboflow" / "raw_results.jsonl"

    for directory in (originals_dir, normalized_dir, labels_dir, dataset_dir, raw_results_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    seen_digests: set[str] = set()
    duplicate_count = 0
    dataset_items: list[DatasetItem] = []
    class_to_yolo_id: dict[str, int] = {}
    class_names: list[str] = []

    _log(job_dir, "Job started", total_images=len(image_paths))

    for image_path in image_paths:
        digest = file_sha256(image_path)
        if digest in seen_digests:
            duplicate_count += 1
            _log(job_dir, "Skipped exact duplicate image", image=str(image_path), level="warning")
            continue
        seen_digests.add(digest)

        original_path = originals_dir / image_path.name
        shutil.copy2(image_path, original_path)
        normalized_path = normalized_dir / f"{image_path.stem}.jpg"
        normalize_to_jpg(original_path, normalized_path)
        _log(job_dir, "Normalized image to JPG", image=normalized_path.name)

        workflow_result = roboflow_client.run_image(normalized_path)
        _append_jsonl(raw_results_path, {"image": normalized_path.name, "result": workflow_result})

        result = _first_result(workflow_result)
        if result.get("upload_error_status"):
            _log(
                job_dir,
                str(result.get("upload_message", "Roboflow upload warning")),
                image=normalized_path.name,
                level="warning",
            )

        image_meta, predictions = _detections_from_result(result)
        if not predictions and not config.include_empty_labels:
            _log(job_dir, "Skipped image with no detections", image=normalized_path.name, level="warning")
            continue

        remapped_predictions = _remap_predictions(predictions, class_to_yolo_id, class_names)
        label_path = labels_dir / f"{normalized_path.stem}.txt"
        write_label_file(
            label_path,
            remapped_predictions,
            image_width=image_meta["width"],
            image_height=image_meta["height"],
        )
        dataset_items.append(DatasetItem(image_path=normalized_path, label_path=label_path))
        _log(job_dir, "Wrote YOLO labels", image=normalized_path.name, detections=len(predictions))

    if not class_names:
        class_names.append("object")

    dataset_split = write_split_dataset(
        dataset_items,
        dataset_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    write_data_yaml(dataset_dir, class_names)

    summary = {
        "input_images": len(image_paths),
        "processed_images": len(dataset_items),
        "duplicate_images": duplicate_count,
        "classes": class_names,
        "train_count": len(dataset_split.train),
        "valid_count": len(dataset_split.valid),
        "test_count": len(dataset_split.test),
        "dataset_dir": str(dataset_dir),
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(job_dir, "Job completed", **summary)
    return summary
