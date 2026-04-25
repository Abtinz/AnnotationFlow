"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration values used by the backend pipeline."""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_cors_origins: str = "http://localhost:8081,http://127.0.0.1:8081"
    roboflow_api_url: str = "https://serverless.roboflow.com"
    roboflow_api_key: str = ""
    roboflow_workspace_name: str = ""
    roboflow_workflow_id: str = ""
    roboflow_use_cache: bool = True
    roboflow_confidence: float = 0.4
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("output")
    dataset_name: str = "annotationflow_dataset"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    include_empty_labels: bool = False

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
