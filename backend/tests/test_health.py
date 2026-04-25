"""Tests for the FastAPI health endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok() -> None:
    """The health endpoint reports that the API process is alive."""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_frontend_origins() -> None:
    """Browser requests from supported frontend origins receive CORS headers."""
    client = TestClient(app)

    for origin in ("http://localhost:8081", "http://127.0.0.1:8081"):
        response = client.options(
            "/jobs",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
