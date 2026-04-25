"""Tests for FastAPI job endpoints."""

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_workflow_client


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


def test_create_job_processes_upload_and_returns_logs(tmp_path: Path, monkeypatch) -> None:
    """Uploading an image creates a processed job with readable logs."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    app.dependency_overrides[get_workflow_client] = lambda: FakeWorkflowClient()
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
        assert payload["status"] == "completed"
        assert payload["summary"]["classes"] == ["bus"]

        job_response = client.get(f"/jobs/{payload['job_id']}")
        assert job_response.status_code == 200
        assert job_response.json()["status"] == "completed"

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
