"""Tests for image validation and JPG normalization."""

from pathlib import Path

import pytest
from PIL import Image

from app.image_normalizer import UnsupportedImageFormatError, normalize_to_jpg


def test_normalize_to_jpg_converts_png_to_rgb_jpg(tmp_path: Path) -> None:
    """A supported non-JPG image is written as an RGB JPG."""
    source = tmp_path / "sample.png"
    destination = tmp_path / "normalized" / "sample.jpg"
    Image.new("RGBA", (12, 8), (255, 0, 0, 128)).save(source)

    normalized = normalize_to_jpg(source, destination)

    assert normalized == destination
    with Image.open(destination) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (12, 8)


def test_normalize_to_jpg_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported file extensions fail before image conversion is attempted."""
    source = tmp_path / "notes.txt"
    source.write_text("not an image", encoding="utf-8")

    with pytest.raises(UnsupportedImageFormatError):
        normalize_to_jpg(source, tmp_path / "notes.jpg")

