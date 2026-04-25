"""Roboflow Workflow client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class WorkflowRunner(Protocol):
    """Protocol for the subset of the inference SDK used by AnnotationFlow."""

    def run_workflow(self, **kwargs: object) -> Any:
        """Run a Roboflow workflow."""


@dataclass(frozen=True)
class RoboflowWorkflowConfig:
    """Connection settings for a user-defined Roboflow Workflow."""

    api_url: str
    api_key: str
    workspace_name: str
    workflow_id: str
    use_cache: bool
    confidence: float | None = None


class RoboflowWorkflowClient:
    """Small wrapper around `inference_sdk.InferenceHTTPClient`.

    The wrapper keeps Roboflow-specific request construction out of the job
    pipeline and makes network behavior easy to replace in tests.
    """

    def __init__(
        self,
        config: RoboflowWorkflowConfig,
        inference_client: WorkflowRunner | None = None,
    ) -> None:
        self.config = config
        self._client = inference_client or self._build_inference_client(config)

    @staticmethod
    def _build_inference_client(config: RoboflowWorkflowConfig) -> WorkflowRunner:
        """Create the official Roboflow inference HTTP client."""
        from inference_sdk import InferenceHTTPClient

        return InferenceHTTPClient(api_url=config.api_url, api_key=config.api_key)

    def run_image(self, image_path: Path) -> Any:
        """Run the configured Roboflow Workflow on one image path."""
        request: dict[str, object] = {
            "workspace_name": self.config.workspace_name,
            "workflow_id": self.config.workflow_id,
            "images": {"image": str(image_path)},
            "use_cache": self.config.use_cache,
        }
        if self.config.confidence is not None:
            # Roboflow workflows can expose named parameters; confidence is the
            # first one this project needs for the bus workflow test case.
            request["parameters"] = {"confidence": self.config.confidence}

        return self._client.run_workflow(**request)
