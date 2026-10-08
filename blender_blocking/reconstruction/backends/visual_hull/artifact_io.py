from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def artifact_paths(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {
        "volume_dir": root / "vol",
        "sdf_volume_dir": root / "sdf",
        "mesh_obj": root / "m" / "vh.obj",
        "openvdb": root / "vol" / "v.vdb",
    }
