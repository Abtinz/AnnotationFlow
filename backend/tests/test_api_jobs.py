"""Tests for FastAPI job endpoints."""

import io
import zipfile
from pathlib import Path

import app.main as main_module
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import app, get_settings


class FakeWorkflowClient:
    """Fake workflow client used by API tests."""

    def run_image(self, image_path: Path) -> list[dict[str, object]]:
        """Return one object detection for each uploaded image."""
        return [
            {
                "detections": {
                    "image": {"width": 10, "height": 10},
                    "predictions": [
                        {
                            "class_id": 5,
                            "class": "bus",
                            "x": 5,
                            "y": 5,
                            "width": 4,
                            "height": 4,
                        }
                    ],
                }
            }
        ]


class CapturingWorkflowClient(FakeWorkflowClient):
    """Fake workflow client that exposes the effective Roboflow config."""

    def __init__(self, config: object) -> None:
        self.config = config


def test_create_job_rejects_invalid_runtime_ratios(tmp_path: Path, monkeypatch) -> None:
    """Invalid split ratios fail before a background job starts."""
    settings = Settings(
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "uploads",
        roboflow_api_key="env-key",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(main_module, "RoboflowWorkflowClient", lambda config: FakeWorkflowClient())
    image_path = tmp_path / "bus.png"
    Image.new("RGB", (10, 10), "yellow").save(image_path)

    try:
        client = TestClient(app)
        with image_path.open("rb") as image_file:
            response = client.post(
                "/jobs",
                data={"config": '{"TRAIN_RATIO":0.9,"VAL_RATIO":0.2,"TEST_RATIO":0.1}'},
                files=[("files", ("bus.png", image_file, "image/png"))],
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
    finally:
        app.dependency_overrides.clear()


def test_create_job_accepts_optional_runtime_config(tmp_path: Path, monkeypatch) -> None:
    """Runtime form config overrides settings for one job."""
    settings = Settings(
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "uploads",
        roboflow_api_key="env-key",
        roboflow_workspace_name="env-workspace",
        roboflow_workflow_id="env-workflow",
        roboflow_use_cache=True,
        roboflow_confidence=0.4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    captured_configs = []

    def fake_roboflow_client(config: object) -> CapturingWorkflowClient:
        captured_configs.append(config)
        return CapturingWorkflowClient(config)

    monkeypatch.setattr(main_module, "RoboflowWorkflowClient", fake_roboflow_client)
    image_path = tmp_path / "bus.png"
    Image.new("RGB", (10, 10), "yellow").save(image_path)

    try:
        client = TestClient(app)
        with image_path.open("rb") as image_file:
            response = client.post(
                "/jobs",
                data={
                    "config": (
                        '{"ROBOFLOW_API_KEY":"runtime-key",'
                        '"ROBOFLOW_WORKSPACE_NAME":"runtime-workspace",'
                        '"ROBOFLOW_WORKFLOW_ID":"runtime-workflow",'
                        '"ROBOFLOW_USE_CACHE":false,'
                        '"ROBOFLOW_CONFIDENCE":0.73,'
                        '"TRAIN_RATIO":1,'
                        '"VAL_RATIO":0,'
                        '"TEST_RATIO":0}'
                    )
                },
                files=[("files", ("bus.png", image_file, "image/png"))],
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "processing"

        job_response = client.get(f"/jobs/{payload['job_id']}")
        summary = job_response.json()["summary"]
        assert summary["train_count"] == 1
        assert summary["valid_count"] == 0
        assert summary["test_count"] == 0
        assert summary["runtime_config"]["roboflow_workspace_name"] == "runtime-workspace"
        assert summary["runtime_config"]["roboflow_workflow_id"] == "runtime-workflow"
        assert summary["runtime_config"]["roboflow_use_cache"] is False
        assert summary["runtime_config"]["roboflow_confidence"] == 0.73
        assert "roboflow_api_key" not in summary["runtime_config"]
        assert captured_configs[0].api_key == "runtime-key"
    finally:
        app.dependency_overrides.clear()


def test_create_job_processes_upload_and_returns_logs(tmp_path: Path, monkeypatch) -> None:
    """Uploading an image creates a processed job with readable logs."""
    settings = Settings(
        output_dir=tmp_path / "output",
        upload_dir=tmp_path / "uploads",
        roboflow_api_key="env-key",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(main_module, "RoboflowWorkflowClient", lambda config: FakeWorkflowClient())
    image_path = tmp_path / "bus.png"
    Image.new("RGB", (10, 10), "yellow").save(image_path)

    try:
        client = TestClient(app)
        with image_path.open("rb") as image_file:
            response = client.post(
                "/jobs",
                files=[("files", ("bus.png", image_file, "image/png"))],
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "processing"

        job_response = client.get(f"/jobs/{payload['job_id']}")
        assert job_response.status_code == 200
        assert job_response.json()["status"] == "completed"
        assert job_response.json()["summary"]["classes"] == ["bus"]

        logs_response = client.get(f"/jobs/{payload['job_id']}/logs")
        assert logs_response.status_code == 200
        assert any(log["message"] == "Job completed" for log in logs_response.json()["logs"])

        dataset_response = client.get(f"/jobs/{payload['job_id']}/dataset")
        assert dataset_response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(dataset_response.content)) as archive:
            assert "dataset/data.yaml" in archive.namelist()
            assert "dataset/train/labels/bus.txt" in archive.namelist()
    finally:
        app.dependency_overrides.clear()
