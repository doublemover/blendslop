"""Frozen, bounded surface-quality workload and authored rounded-dot reference.

This workload is distinct from the historical synthetic suite. Its declaration
is preparation, never a record of successful reconstruction.
"""

from __future__ import annotations

import math
import numpy as np


def quality_workload() -> dict:
    """Fixed parameters, units, cameras and gates established before candidates."""
    cases = [
        ("sphere", {"primitive": "sphere", "radius": .8}),
        ("anisotropic_ellipsoid", {"primitive": "ellipsoid", "radii": [.9, .55, .7]}),
        ("cylinder", {"primitive": "cylinder", "radius": .65, "height": 1.6}),
        ("tapered_frustum", {"primitive": "frustum", "radius_bottom": .8, "radius_top": .35, "height": 1.8}),
        ("smooth_vase", {"builder": "analytic_vase", "height": 2.6, "radius_mean": .6, "radius_amplitude": .2}),
        ("torus", {"primitive": "torus", "major_radius": .7, "minor_radius": .22}),
        ("capsule", {"primitive": "capsule", "radius": .4, "segment_height": 1.2}),
        ("rounded_box", {"primitive": "rounded_box", "dimensions": {"width": 1.6, "depth": 1.1, "height": 1.3}, "radius": .15}),
        ("thin_plate", {"primitive": "box", "dimensions": {"width": 1.6, "depth": .08, "height": 1.2}}),
        ("concave_arch", {"builder": "compound", "parts": [
            {"type": "box", "center": [0, 0, 0], "size": [1.8, .4, 1.8]},
            {"type": "box", "center": [0, 0, -.3], "size": [1.0, .8, 1.4], "boolean": "subtract"}]}),
        ("asymmetric_multipart_solid", {"builder": "compound", "parts": [
            {"type": "box", "center": [0, 0, -.25], "size": [1.6, .65, .5]},
            {"type": "box", "center": [-.4, .1, .4], "size": [.45, .45, 1.1]},
            {"type": "box", "center": [.45, -.08, .12], "size": [.35, .45, .65]}]}),
        ("rounded_triangle_dot", {"builder": "rounded_triangle", "vertices_xy": [[0, .9], [-.7794228634059948, -.45], [.7794228634059948, -.45]], "corner_radius": .16, "thickness": .48, "dome": "z=+-thickness/2*cos(phi), outline_scale=sin(phi)", "corner_segments": 32, "dome_segments": 64}),
    ]
    return {
        "protocol": "quality-surface-fixed-v1", "status": "prepared_unmeasured",
        "units": "Blender world units; no independent centering, scaling or alignment",
        "seed": 61007, "sample_count_per_direction": 4096,
        "normal_orientation": "oriented geometric triangle normals; no absolute-dot flip allowance",
        "seams": "caps and authored corners retained; area samples omit measure-zero edges only",
        "resolution": [512, 512], "camera_projection": "orthographic, shared reference frame",
        "views": ["front", "side", "top", "oblique_35_28", "oblique_145_40"],
        "oblique_degrees": [[35, 28], [145, 40]],
        "limits": {"timeout_seconds": 1200, "max_rss_bytes": 8589934592, "required_cases": 12},
        "dependencies": ["Blender native Python", "NumPy", "SciPy", "Pillow", "OpenCV"],
        "held_out_before_tuning": ["torus", "thin_plate", "concave_arch", "asymmetric_multipart_solid", "rounded_triangle_dot"],
        "contracts": {
            "existing_silhouette_gates": "unchanged; require each existing area, boundary and signed-distance gate",
            "surface": {"symmetric_mean_distance_world_max": .003, "normal_angle_p95_degrees_max": 2.5,
                        "scope": "smooth-vase analytic contract only; other families require independent tessellation/noise qualification before acceptance",
                        "derivation": "vase max r=.8, 192 radial sectors: chord error <=.8*(1-cos(pi/192))=.000108; 65 input sections over2.6 and |r''|<=.2*(2*pi/2.6)^2 yield linear section error <=.000247; directional face-normal discretization <=sqrt(.9375^2+1.36^2) degrees. Budgets include declared approximation allowance, not candidate measurements"},
            "rounded_triangle": {"support_directions": 24, "support_distance_world_max": .005,
                                 "thickness_world_error_max": .005, "surface_status": "requires reference tessellation/noise qualification"},
            "topology": "one closed oriented component for each required solid; cavities/features are separate gates",
            "editability": "explicit artist-edit response receipt required; absence is blocked",
            "aggregate": "any required-case failure or unavailable required metric blocks acceptance",
        },
        "cases": [{"name": name, "parameters": params, "status": "unmeasured"} for name, params in cases],
    }


def rounded_triangle_mesh(parameters: dict | None = None):
    """Preserved frozen reference, shared with the editable production primitive."""
    from primitives.rounded_triangle import rounded_triangle_arrays
    if parameters is None:
        parameters = quality_workload()["cases"][-1]["parameters"]
    return rounded_triangle_arrays(parameters)


def triangle_preservation(vertices, parameters=None) -> dict:
    """A circularized or flattened control cannot pass this authored-shape gate."""
    if parameters is None:
        parameters = quality_workload()["cases"][-1]["parameters"]
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("candidate vertices must be nonempty finite XYZ coordinates")
    angles = np.arange(24) * 2 * math.pi / 24
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    expected = (np.asarray(parameters["vertices_xy"]) @ directions.T).max(axis=0) + parameters["corner_radius"]
    observed = (vertices[:, :2] @ directions.T).max(axis=0)
    error = float(np.abs(observed - expected).max())
    thickness_error = abs(float(np.ptp(vertices[:, 2])) - float(parameters["thickness"]))
    return {"support_error_world": error, "thickness_error_world": thickness_error,
            "passed": error <= .005 and thickness_error <= .005,
            "surface_verdict": "independent; not implied by support/thickness"}
