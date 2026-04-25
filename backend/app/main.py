"""FastAPI application entrypoint for AnnotationFlow."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.jobs import ImageWorkflowClient, PipelineConfig, process_image_paths
from app.roboflow_client import RoboflowWorkflowClient, RoboflowWorkflowConfig

app = FastAPI(
    title="AnnotationFlow API",
    description="Automated YOLO object-detection dataset builder.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal health response for local and Docker checks."""
    return {"status": "ok"}


def get_workflow_client(settings: Settings = Depends(get_settings)) -> ImageWorkflowClient:
    """Build the configured Roboflow Workflow client."""
    if not settings.roboflow_api_key:
        raise HTTPException(status_code=500, detail="ROBOFLOW_API_KEY is not configured")
    return RoboflowWorkflowClient(
        RoboflowWorkflowConfig(
            api_url=settings.roboflow_api_url,
            api_key=settings.roboflow_api_key,
            workspace_name=settings.roboflow_workspace_name,
            workflow_id=settings.roboflow_workflow_id,
            use_cache=settings.roboflow_use_cache,
            confidence=settings.roboflow_confidence,
        )
    )


def _pipeline_config(settings: Settings) -> PipelineConfig:
    """Build pipeline config from application settings."""
    return PipelineConfig(
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        split_seed=settings.split_seed,
        include_empty_labels=settings.include_empty_labels,
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty object if absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from disk for UI log polling."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.post("/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    workflow_client: ImageWorkflowClient = Depends(get_workflow_client),
) -> dict[str, Any]:
    """Create a processing job from uploaded images.

    The first implementation runs synchronously, which keeps job behavior easy
    to verify. The API shape still returns a job id so this can move to
    background execution later without changing the frontend contract.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    job_id = uuid.uuid4().hex
    upload_dir = settings.upload_dir / "jobs" / job_id
    job_dir = settings.output_dir / "jobs" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    uploaded_paths: list[Path] = []
    for upload in files:
        filename = Path(upload.filename or "upload").name
        upload_path = upload_dir / filename
        with upload_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
        uploaded_paths.append(upload_path)

    summary = process_image_paths(
        image_paths=uploaded_paths,
        job_dir=job_dir,
        roboflow_client=workflow_client,
        config=_pipeline_config(settings),
    )
    return {"job_id": job_id, "status": "completed", "summary": summary}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return current job status and summary."""
    job_dir = settings.output_dir / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    summary = _read_json(job_dir / "summary.json")
    status = "completed" if summary else "running"
    return {"job_id": job_id, "status": status, "summary": summary}


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return structured logs for a job."""
    job_dir = settings.output_dir / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "logs": _read_jsonl(job_dir / "logs.jsonl")}


@app.get("/jobs/{job_id}/dataset")
def download_job_dataset(job_id: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    """Return the generated YOLO dataset as a ZIP file."""
    job_dir = settings.output_dir / "jobs" / job_id
    dataset_dir = job_dir / "dataset"
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    zip_path = job_dir / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=Path("dataset") / path.relative_to(dataset_dir))

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}-dataset.zip",
    )
