"""Duplicate detection utilities for uploaded images."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest for a file.

    The file is read in chunks so large image uploads do not need to be loaded
    into memory all at once.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_exact_duplicates(paths: Iterable[Path]) -> list[list[Path]]:
    """Group files that have identical bytes.

    Args:
        paths: Candidate files to compare.

    Returns:
        A stable list of duplicate groups. Each group contains two or more
        paths with identical SHA256 hashes, preserving the input order within
        each group.
    """
    paths_by_digest: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        paths_by_digest[file_sha256(path)].append(path)

    return [group for group in paths_by_digest.values() if len(group) > 1]
