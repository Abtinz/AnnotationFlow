"""Source checks for the first mobile screen."""

from pathlib import Path


def test_app_wires_upload_status_logs_and_dataset_download() -> None:
    """The mobile app references the expected backend job endpoints."""
    source = Path("App.tsx").read_text(encoding="utf-8")

    assert "DocumentPicker.getDocumentAsync" in source
    assert 'fetch(`${API_BASE_URL}/jobs`' in source
    assert 'fetch(`${API_BASE_URL}/jobs/${jobId}`)' in source
    assert 'fetch(`${API_BASE_URL}/jobs/${jobId}/logs`)' in source
    assert "dataset" in source

