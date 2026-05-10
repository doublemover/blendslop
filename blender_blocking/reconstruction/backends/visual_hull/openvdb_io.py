from __future__ import annotations

from pathlib import Path
from typing import Any


def openvdb_artifact_path(root: Path) -> Path:
    return root / "volume" / "volume.vdb"
