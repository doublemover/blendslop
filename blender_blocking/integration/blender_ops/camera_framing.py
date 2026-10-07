"""Camera framing utilities for orthographic renders."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def compute_bounds_world(
    objects: Iterable[object],
) -> Tuple[mathutils.Vector, mathutils.Vector]:
    """Compute world-space bounds for a list of Blender objects."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available")

    min_coords = mathutils.Vector((float("inf"),) * 3)
    max_coords = mathutils.Vector((float("-inf"),) * 3)

    from reconstruction.output_targets import output_mesh_targets
    found = False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in output_mesh_targets(objects):
        evaluated = obj.evaluated_get(depsgraph)
        for vertex in evaluated.bound_box:
            world_coord = evaluated.matrix_world @ mathutils.Vector(vertex)
            min_coords.x = min(min_coords.x, world_coord.x)
            min_coords.y = min(min_coords.y, world_coord.y)
            min_coords.z = min(min_coords.z, world_coord.z)
            max_coords.x = max(max_coords.x, world_coord.x)
            max_coords.y = max(max_coords.y, world_coord.y)
            max_coords.z = max(max_coords.z, world_coord.z)
            found = True

    if not found:
        min_coords = mathutils.Vector((0.0, 0.0, 0.0))
        max_coords = mathutils.Vector((0.0, 0.0, 0.0))

    return min_coords, max_coords


def configure_ortho_camera_for_view(
    camera: object,
    view: str,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    *,
    margin_frac: float = 0.08,
    resolution: Tuple[int, int] = (512, 512),
    distance_factor: float = 2.0,
) -> None:
    """Configure an orthographic camera for a given view and bounds."""
    if not BLENDER_AVAILABLE:
        raise RuntimeError("Blender API not available")

    center = mathutils.Vector(
        (
            (bounds_min[0] + bounds_max[0]) / 2.0,
            (bounds_min[1] + bounds_max[1]) / 2.0,
            (bounds_min[2] + bounds_max[2]) / 2.0,
        )
    )

    width_extent = max(bounds_max[0] - bounds_min[0], 1e-6)
    depth_extent = max(bounds_max[1] - bounds_min[1], 1e-6)
    height_extent = max(bounds_max[2] - bounds_min[2], 1e-6)

    aspect = resolution[0] / resolution[1] if resolution[1] else 1.0

    if view == "front":
        view_width = width_extent
        view_height = height_extent
        location = mathutils.Vector(
            (center.x, center.y - depth_extent * distance_factor, center.z)
        )
        rotation = (math.radians(90), 0.0, 0.0)
    elif view == "side":
        view_width = depth_extent
        view_height = height_extent
        location = mathutils.Vector(
            (center.x + width_extent * distance_factor, center.y, center.z)
        )
        rotation = (math.radians(90), 0.0, math.radians(90))
    elif view == "top":
        view_width = width_extent
        view_height = depth_extent
        location = mathutils.Vector(
            (center.x, center.y, center.z + height_extent * distance_factor)
        )
        # Canonical top: horizontal +X, vertical +Y (also used by hull and solvers).
        rotation = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown view: {view}")

    ortho_scale = max(
        view_width * (1.0 + 2.0 * margin_frac),
        view_height * aspect * (1.0 + 2.0 * margin_frac),
    )

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(ortho_scale, 1e-3)
    camera.location = location
    camera.rotation_euler = rotation
