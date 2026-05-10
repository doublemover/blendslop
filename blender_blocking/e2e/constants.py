from __future__ import annotations
from pathlib import Path
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
TEMP_OUTPUT_ROOT = REPO_ROOT / "temp"
ALL_RECONSTRUCTION_MODES = (
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "shape_program",
    "ensemble",
)
BACKEND_MODES = {
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "shape_program",
    "ensemble",
}
RENDER_IOU_MODES = {
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
}
DEFAULT_ENSEMBLE_CANDIDATES = (
    "visual_hull_voxel",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "shape_program",
)
DEFAULT_SYNTHETIC_MATRIX_MODES = (
    "legacy",
    "loft_profile",
    "silhouette_intersection",
    "visual_hull_voxel",
)
BACKEND_STATUS_OK = {"success", "degraded", "research_only"}
VALIDATION_MODES = ("auto", "render-iou", "backend-status", "novel-view")
