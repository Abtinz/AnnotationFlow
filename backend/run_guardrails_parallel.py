from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.dataset_splitter import DatasetItem, write_split_dataset
from app.duplicate_detector import file_sha256
from app.image_normalizer import normalize_to_jpg
from app.jobs import PipelineConfig
from app.roboflow_client import RoboflowWorkflowClient, RoboflowWorkflowConfig
from app.yolo_writer import write_data_yaml, write_label_file


WORKFLOW_ID = "find-guardrail"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    normalized_path: Path
    label_path: Path | None
    predictions: list[dict[str, Any]]
    failed: bool = False
    skipped: bool = False


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def log(job_dir: Path, message: str, level: str = "info", **extra: Any) -> None:
    append_jsonl(
        job_dir / "logs.jsonl",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **extra,
        },
    )


def compact_workflow_result(workflow_result: Any) -> Any:
    if isinstance(workflow_result, list):
        return [compact_workflow_result(item) for item in workflow_result]
    if isinstance(workflow_result, dict):
        return {key: value for key, value in workflow_result.items() if key != "visualization"}
    return workflow_result


def first_result(workflow_result: Any) -> dict[str, Any]:
    if isinstance(workflow_result, list):
        if not workflow_result:
            return {}
        return workflow_result[0] if isinstance(workflow_result[0], dict) else {}
    return workflow_result if isinstance(workflow_result, dict) else {}


def detections_from_result(result: dict[str, Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    detections = result.get("detections") or result.get("predictions") or {}
    if not isinstance(detections, dict):
        return {"width": 0, "height": 0}, []
    image = detections.get("image", {})
    predictions = detections.get("predictions", [])
    if not isinstance(image, dict) or not isinstance(predictions, list):
        return {"width": 0, "height": 0}, []
    return (
        {"width": int(image.get("width") or 0), "height": int(image.get("height") or 0)},
        [prediction for prediction in predictions if isinstance(prediction, dict)],
    )


def image_size(meta: dict[str, int], image_path: Path) -> dict[str, int]:
    if meta["width"] > 0 and meta["height"] > 0:
        return meta
    with Image.open(image_path) as image:
        width, height = image.size
    return {"width": width, "height": height}


def make_client(settings: Settings) -> RoboflowWorkflowClient:
    return RoboflowWorkflowClient(
        RoboflowWorkflowConfig(
            api_url=settings.roboflow_api_url,
            api_key=settings.roboflow_api_key,
            workspace_name=settings.roboflow_workspace_name,
            workflow_id=WORKFLOW_ID,
            use_cache=settings.roboflow_use_cache,
            confidence=settings.roboflow_confidence,
        )
    )


def process_one(
    image_path: Path,
    job_dir: Path,
    settings: Settings,
    config: PipelineConfig,
    raw_results_lock: threading.Lock,
) -> ImageResult:
    normalized_dir = job_dir / "normalized"
    labels_dir = job_dir / "labels"
    raw_results_path = job_dir / "roboflow" / "raw_results.jsonl"

    normalized_path = normalized_dir / f"{image_path.stem}.jpg"
    label_path = labels_dir / f"{normalized_path.stem}.txt"
    try:
        normalize_to_jpg(image_path, normalized_path)
        log(job_dir, "Normalized image to JPG", image=normalized_path.name)

        workflow_result = make_client(settings).run_image(normalized_path)
        with raw_results_lock:
            append_jsonl(
                raw_results_path,
                {"image": normalized_path.name, "result": compact_workflow_result(workflow_result)},
            )
        result = first_result(workflow_result)
        meta, predictions = detections_from_result(result)
        if not predictions and not config.include_empty_labels:
            normalized_path.unlink(missing_ok=True)
            log(job_dir, "Skipped image with no detections", image=normalized_path.name, level="warning")
            return ImageResult(image_path, normalized_path, None, [], skipped=True)

        meta = image_size(meta, normalized_path)
        write_label_file(label_path, predictions, image_width=meta["width"], image_height=meta["height"])
        log(job_dir, "Wrote YOLO labels", image=normalized_path.name, detections=len(predictions))
        return ImageResult(image_path, normalized_path, label_path, predictions)
    except Exception as error:
        normalized_path.unlink(missing_ok=True)
        with raw_results_lock:
            append_jsonl(
                raw_results_path,
                {"image": normalized_path.name, "error": str(error), "error_type": type(error).__name__},
            )
        log(job_dir, "Roboflow workflow failed", image=normalized_path.name, level="error", error=str(error))
        return ImageResult(image_path, normalized_path, None, [], failed=True)


def remap_label_file(result: ImageResult, class_to_yolo_id: dict[str, int], class_names: list[str]) -> DatasetItem:
    assert result.label_path is not None
    with Image.open(result.normalized_path) as image:
        width, height = image.size
    remapped: list[dict[str, Any]] = []
    for prediction in result.predictions:
        class_name = str(prediction.get("class", prediction.get("class_id", "object")))
        if class_name not in class_to_yolo_id:
            class_to_yolo_id[class_name] = len(class_names)
            class_names.append(class_name)
        next_prediction = dict(prediction)
        next_prediction["class_id"] = class_to_yolo_id[class_name]
        remapped.append(next_prediction)
    write_label_file(result.label_path, remapped, image_width=width, image_height=height)
    return DatasetItem(image_path=result.normalized_path, label_path=result.label_path)


def main() -> None:
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/abtinzandi/Downloads/Gard Rails")
    job_id = sys.argv[2] if len(sys.argv) > 2 else f"guardrails-parallel-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    settings = Settings()
    if not settings.roboflow_api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set in .env")
    config = PipelineConfig(
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        split_seed=settings.split_seed,
        include_empty_labels=settings.include_empty_labels,
    )
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    job_dir = Path("output") / "jobs" / job_id
    for directory in ("normalized", "labels", "dataset", "roboflow"):
        (job_dir / directory).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    unique_paths: list[Path] = []
    duplicate_count = 0
    for path in image_paths:
        digest = file_sha256(path)
        if digest in seen:
            duplicate_count += 1
            log(job_dir, "Skipped exact duplicate image", image=str(path), level="warning")
            continue
        seen.add(digest)
        unique_paths.append(path)

    print(f"Processing {len(unique_paths)} unique images from {image_dir} with {workers} workers", flush=True)
    print(f"Workflow: {settings.roboflow_workspace_name}/{WORKFLOW_ID}", flush=True)
    print(f"Job dir: {job_dir}", flush=True)
    log(job_dir, "Job started", total_images=len(image_paths), unique_images=len(unique_paths), workers=workers)

    raw_results_lock = threading.Lock()
    results: list[ImageResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_one, path, job_dir, settings, config, raw_results_lock)
            for path in unique_paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 10 == 0 or index == len(futures):
                print(f"Completed {index}/{len(futures)}", flush=True)

    class_to_yolo_id: dict[str, int] = {}
    class_names: list[str] = []
    dataset_items = [
        remap_label_file(result, class_to_yolo_id, class_names)
        for result in results
        if result.label_path is not None and not result.failed and not result.skipped
    ]
    if not class_names:
        class_names.append("object")

    dataset_dir = job_dir / "dataset"
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
        "failed_images": sum(1 for result in results if result.failed),
        "skipped_images": sum(1 for result in results if result.skipped),
        "classes": class_names,
        "train_count": len(dataset_split.train),
        "valid_count": len(dataset_split.valid),
        "test_count": len(dataset_split.test),
        "dataset_dir": str(dataset_dir),
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log(job_dir, "Job completed", **summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
