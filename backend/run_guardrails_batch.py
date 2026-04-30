from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.jobs import PipelineConfig, process_image_paths
from app.roboflow_client import RoboflowWorkflowClient, RoboflowWorkflowConfig


WORKFLOW_ID = "find-guardrail"


def image_paths_for(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}
    )


def run(image_dir: Path, job_name: str | None = None) -> dict[str, object]:
    settings = Settings()
    image_paths = image_paths_for(image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}")
    if not settings.roboflow_api_key:
        raise SystemExit("ROBOFLOW_API_KEY is not set in .env")

    job_id = job_name or f"guardrails-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_dir = Path("output") / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    client = RoboflowWorkflowClient(
        RoboflowWorkflowConfig(
            api_url=settings.roboflow_api_url,
            api_key=settings.roboflow_api_key,
            workspace_name=settings.roboflow_workspace_name,
            workflow_id=WORKFLOW_ID,
            use_cache=settings.roboflow_use_cache,
            confidence=settings.roboflow_confidence,
        )
    )
    config = PipelineConfig(
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        split_seed=settings.split_seed,
        include_empty_labels=settings.include_empty_labels,
    )

    print(f"Processing {len(image_paths)} images from {image_dir}", flush=True)
    print(f"Workflow: {settings.roboflow_workspace_name}/{WORKFLOW_ID}", flush=True)
    print(f"Job dir: {job_dir}", flush=True)
    return process_image_paths(image_paths, job_dir, client, config)


def main() -> None:
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/abtinzandi/Downloads/Gard Rails")
    job_name = sys.argv[2] if len(sys.argv) > 2 else None
    summary = run(image_dir, job_name)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
