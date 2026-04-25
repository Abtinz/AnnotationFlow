"""Image validation and JPG normalization utilities."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Register HEIC/HEIF support once when the module is imported.
register_heif_opener()

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".heic",
    ".heif",
}


class UnsupportedImageFormatError(ValueError):
    """Raised when an uploaded file extension is not supported."""


def validate_supported_image(path: Path) -> None:
    """Validate that a file has an image extension supported by the pipeline."""
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise UnsupportedImageFormatError(f"Unsupported image format: {path.suffix}")


def normalize_to_jpg(source_path: Path, destination_path: Path, quality: int = 95) -> Path:
    """Convert a supported image file to an RGB JPG.

    Args:
        source_path: Input image file path.
        destination_path: Output `.jpg` file path.
        quality: JPEG quality from 1 to 100.

    Returns:
        The destination path that was written.

    Raises:
        UnsupportedImageFormatError: If the source extension is not supported.
        ValueError: If quality is outside Pillow's accepted 1..100 range.
    """
    validate_supported_image(source_path)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        # EXIF transpose keeps phone images upright before training/export.
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.save(
            destination_path,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )

    return destination_path
