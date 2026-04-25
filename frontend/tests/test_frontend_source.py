"""Source checks for the first frontend screen."""

from pathlib import Path


def test_app_wires_upload_status_logs_and_dataset_download() -> None:
    """The frontend app references the expected backend job endpoints."""
    source = Path("src/main.tsx").read_text(encoding="utf-8")

    assert 'type="file"' in source
    assert 'fetch(`${API_BASE_URL}/jobs`' in source
    assert 'fetch(`${API_BASE_URL}/jobs/${jobId}`)' in source
    assert 'fetch(`${API_BASE_URL}/jobs/${jobId}/logs`)' in source
    assert "dataset" in source
