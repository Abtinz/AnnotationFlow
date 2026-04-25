"""FastAPI application entrypoint for AnnotationFlow."""

from fastapi import FastAPI

app = FastAPI(
    title="AnnotationFlow API",
    description="Automated YOLO object-detection dataset builder.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal health response for local and Docker checks."""
    return {"status": "ok"}

