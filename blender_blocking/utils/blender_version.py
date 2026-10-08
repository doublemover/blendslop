"""Runtime identity and API checks for the single supported Blender release."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

SUPPORTED_BLENDER_VERSION = (5, 2, 2)
SUPPORTED_BLENDER_LABEL = "5.2.2 LTS"
CURRENT_BOOLEAN_SOLVERS = ("EXACT", "FLOAT", "MANIFOLD")


def get_blender_version() -> Optional[Tuple[int, int, int]]:
    return bpy.app.version if BLENDER_AVAILABLE else None


def get_blender_version_string() -> Optional[str]:
    return bpy.app.version_string if BLENDER_AVAILABLE else None


def require_supported_blender() -> None:
    """Reject a different release or preview before native work starts."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError(f"Blender {SUPPORTED_BLENDER_LABEL} stable is required")
    if (
        bpy.app.version != SUPPORTED_BLENDER_VERSION
        or bpy.app.version_cycle != "release"
    ):
        raise RuntimeError(
            f"Supported runtime is Blender {SUPPORTED_BLENDER_LABEL} stable; "
            f"found {bpy.app.version_string} ({bpy.app.version_cycle})"
        )


def get_boolean_solver() -> str:
    """Use the current release's exact solver by default."""
    if BLENDER_AVAILABLE:
        require_supported_blender()
    return "EXACT"


def get_available_boolean_solvers() -> List[str]:
    if not BLENDER_AVAILABLE:
        return list(CURRENT_BOOLEAN_SOLVERS)
    require_supported_blender()
    solver_prop = bpy.types.BooleanModifier.bl_rna.properties["solver"]
    return [item.identifier for item in solver_prop.enum_items]


def is_boolean_solver_available(solver: str) -> bool:
    return solver in get_available_boolean_solvers()


def resolve_boolean_solver(override: Optional[str]) -> str:
    """Validate an explicit current solver instead of substituting another."""
    if override is None or override == "auto":
        return get_boolean_solver()
    if not is_boolean_solver_available(override):
        raise ValueError(
            f"Boolean solver {override!r} is unavailable in Blender {SUPPORTED_BLENDER_LABEL}"
        )
    return override


def get_version_info() -> Optional[Dict[str, Any]]:
    if not BLENDER_AVAILABLE:
        return None
    require_supported_blender()
    version = get_blender_version()
    return {
        "version": version,
        "version_string": get_blender_version_string(),
        "major": version[0], "minor": version[1], "patch": version[2],
        "supported_release": SUPPORTED_BLENDER_LABEL,
        "recommended_boolean_solver": get_boolean_solver(),
        "available_boolean_solvers": get_available_boolean_solvers(),
    }


def print_version_info() -> None:
    info = get_version_info()
    if info is None:
        print("Blender not available")
    else:
        print(f"Blender {info['version_string']}; supported stable: {info['supported_release']}")
        print(f"Boolean solvers: {', '.join(info['available_boolean_solvers'])}")


if __name__ == "__main__":
    print_version_info()
