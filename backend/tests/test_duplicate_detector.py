"""Tests for exact duplicate detection."""

from pathlib import Path

from app.duplicate_detector import find_exact_duplicates


def test_find_exact_duplicates_groups_files_with_same_content(tmp_path: Path) -> None:
    """Files with identical bytes are grouped under the same digest."""
    first = tmp_path / "first.jpg"
    duplicate = tmp_path / "duplicate.jpg"
    unique = tmp_path / "unique.jpg"
    first.write_bytes(b"same image bytes")
    duplicate.write_bytes(b"same image bytes")
    unique.write_bytes(b"different image bytes")

    duplicate_groups = find_exact_duplicates([first, duplicate, unique])

    assert duplicate_groups == [[first, duplicate]]

