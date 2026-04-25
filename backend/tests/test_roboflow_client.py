"""Tests for the Roboflow Workflow client wrapper."""

from pathlib import Path

from app.roboflow_client import RoboflowWorkflowClient, RoboflowWorkflowConfig


class FakeInferenceClient:
    """Small fake that records the workflow request for assertions."""

    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def run_workflow(self, **kwargs: object) -> list[dict[str, object]]:
        """Return a minimal Roboflow-like response while saving arguments."""
        self.request = kwargs
        return [{"detections": {"image": {"width": 10, "height": 20}, "predictions": []}}]


def test_run_image_passes_configured_workflow_request(tmp_path: Path) -> None:
    """The wrapper calls the configured workspace and workflow for one image."""
    fake_client = FakeInferenceClient()
    config = RoboflowWorkflowConfig(
        api_url="https://serverless.roboflow.com",
        api_key="secret",
        workspace_name="abtinzandi",
        workflow_id="find-bus",
        use_cache=True,
        confidence=0.4,
    )
    image_path = tmp_path / "bus.jpg"
    image_path.write_bytes(b"image")

    client = RoboflowWorkflowClient(config=config, inference_client=fake_client)
    result = client.run_image(image_path)

    assert result == [{"detections": {"image": {"width": 10, "height": 20}, "predictions": []}}]
    assert fake_client.request == {
        "workspace_name": "abtinzandi",
        "workflow_id": "find-bus",
        "images": {"image": str(image_path)},
        "use_cache": True,
        "parameters": {"confidence": 0.4},
    }

