"""Atomic non-clobber publication of an immutable producer-owned stage file."""
from __future__ import annotations

import os
from pathlib import Path
import stat


def publish_file_no_clobber(staged, destination) -> Path:
    """Publish by hard link, retaining the closed stage and refusing collisions.

    Callers own fresh staging, destination naming and the source's immutability.
    Both paths must be on a filesystem supporting hard links. No copying,
    replacement, deletion, parent creation or historical ownership is inferred.
    """
    source = Path(staged)
    target = Path(destination)
    if not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("publication requires a regular immutable stage file")
    os.link(source, target)
    return target
