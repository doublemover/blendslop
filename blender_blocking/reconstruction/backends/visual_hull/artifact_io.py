from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def artifact_paths(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {
        "volume_dir": root / "volume",
        "sdf_volume_dir": root / "sdf_volume",
        "mesh_obj": root / "mesh" / "visual_hull.obj",
        "openvdb": root / "volume" / "volume.vdb",
    }
