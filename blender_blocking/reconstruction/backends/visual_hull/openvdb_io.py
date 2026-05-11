from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_io import artifact_paths


def openvdb_artifact_path(root: Path) -> Path:
    return artifact_paths(root)["openvdb"]
