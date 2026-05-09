"""Analytic SDF and occupancy helpers for pure-Python fixtures."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .specs import SyntheticShapeSpec


class OptionalDependencyUnavailable(RuntimeError):
    """Raised when an optional array dependency is required for generation."""


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise OptionalDependencyUnavailable(
            "NumPy is required for analytic SDF grids and NPZ synthetic artifacts."
        ) from exc
    return np


def grid_points(resolution: int, bounds: tuple[float, float] = (-1.5, 1.5)) -> Any:
    np = require_numpy()
    if resolution <= 1:
        raise ValueError("resolution must be > 1")
    axis = np.linspace(bounds[0], bounds[1], resolution, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([x, y, z], axis=-1)


def signed_distance(spec: SyntheticShapeSpec, points: Any) -> Any:
    primitive = str(spec.parameters.get("primitive", spec.parameters.get("blockout_kind", "")))
    params = spec.parameters
    if primitive == "box":
        dims = _dimensions(params.get("dimensions", spec.known_dimensions), ["width", "depth", "height"])
        return sdf_box(points, (dims["width"], dims["depth"], dims["height"]))
    if primitive == "sphere":
        return sdf_sphere(points, float(params["radius"]))
    if primitive == "ellipsoid":
        return sdf_ellipsoid(points, tuple(float(x) for x in params["radii"]))
    if primitive == "cylinder":
        return sdf_cylinder(points, float(params["radius"]), float(params["height"]))
    if primitive in ("cone", "frustum"):
        return sdf_frustum(
            points,
            float(params.get("radius_bottom", params.get("radius", 0.5))),
            float(params.get("radius_top", 0.0)),
            float(params["height"]),
        )
    if primitive == "capsule":
        return sdf_capsule(points, float(params["radius"]), float(params["segment_height"]))
    if primitive == "torus":
        return sdf_torus(points, float(params["major_radius"]), float(params["minor_radius"]))
    if primitive == "rounded_box":
        dims = _dimensions(params["dimensions"], ["width", "depth", "height"])
        return sdf_rounded_box(points, (dims["width"], dims["depth"], dims["height"]), float(params["radius"]))
    if primitive == "superquadric":
        return sdf_superquadric(
            points,
            tuple(float(x) for x in params["radii"]),
            tuple(float(x) for x in params["exponents"]),
            float(params.get("taper", 0.0)),
        )
    raise ValueError(f"SDF is not available for primitive {primitive!r}")


def occupancy_grid(
    spec: SyntheticShapeSpec,
    resolution: int = 64,
    bounds: tuple[float, float] = (-1.5, 1.5),
) -> Any:
    np = require_numpy()
    sdf = signed_distance(spec, grid_points(resolution, bounds))
    return np.asarray(sdf <= 0.0, dtype=np.bool_)


def sdf_samples(
    spec: SyntheticShapeSpec,
    resolution: int = 64,
    bounds: tuple[float, float] = (-1.5, 1.5),
) -> dict[str, Any]:
    points = grid_points(resolution, bounds)
    sdf = signed_distance(spec, points)
    return {"points": points, "sdf": sdf}


def sdf_box(points: Any, dimensions: tuple[float, float, float]) -> Any:
    np = require_numpy()
    half = np.asarray(dimensions, dtype=np.float32) / 2.0
    q = np.abs(points) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum.reduce(q, axis=-1), 0.0)
    return outside + inside


def sdf_sphere(points: Any, radius: float) -> Any:
    np = require_numpy()
    return np.linalg.norm(points, axis=-1) - radius


def sdf_ellipsoid(points: Any, radii: tuple[float, float, float]) -> Any:
    np = require_numpy()
    r = np.asarray(radii, dtype=np.float32)
    k0 = np.linalg.norm(points / r, axis=-1)
    k1 = np.linalg.norm(points / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-8)


def sdf_cylinder(points: Any, radius: float, height: float) -> Any:
    np = require_numpy()
    radial = np.linalg.norm(points[..., :2], axis=-1) - radius
    z = np.abs(points[..., 2]) - height / 2.0
    outside = np.linalg.norm(np.stack([np.maximum(radial, 0.0), np.maximum(z, 0.0)], axis=-1), axis=-1)
    inside = np.minimum(np.maximum(radial, z), 0.0)
    return outside + inside


def sdf_frustum(points: Any, radius_bottom: float, radius_top: float, height: float) -> Any:
    np = require_numpy()
    z01 = np.clip((points[..., 2] / height) + 0.5, 0.0, 1.0)
    radius_at_z = radius_bottom + (radius_top - radius_bottom) * z01
    radial = np.linalg.norm(points[..., :2], axis=-1) - radius_at_z
    cap = np.abs(points[..., 2]) - height / 2.0
    outside = np.linalg.norm(np.stack([np.maximum(radial, 0.0), np.maximum(cap, 0.0)], axis=-1), axis=-1)
    inside = np.minimum(np.maximum(radial, cap), 0.0)
    return outside + inside


def sdf_capsule(points: Any, radius: float, segment_height: float) -> Any:
    np = require_numpy()
    half = segment_height / 2.0
    q = points.copy()
    q[..., 2] = np.clip(q[..., 2], -half, half)
    return np.linalg.norm(points - q, axis=-1) - radius


def sdf_torus(points: Any, major_radius: float, minor_radius: float) -> Any:
    np = require_numpy()
    qx = np.linalg.norm(points[..., :2], axis=-1) - major_radius
    q = np.stack([qx, points[..., 2]], axis=-1)
    return np.linalg.norm(q, axis=-1) - minor_radius


def sdf_rounded_box(points: Any, dimensions: tuple[float, float, float], radius: float) -> Any:
    shrunk = tuple(max(1e-6, value - 2.0 * radius) for value in dimensions)
    return sdf_box(points, shrunk) - radius


def sdf_superquadric(
    points: Any,
    radii: tuple[float, float, float],
    exponents: tuple[float, float],
    taper: float = 0.0,
) -> Any:
    np = require_numpy()
    r = np.asarray(radii, dtype=np.float32)
    e_xy, e_z = exponents
    z_norm = np.clip(points[..., 2] / max(r[2], 1e-6), -1.0, 1.0)
    taper_scale = np.maximum(0.15, 1.0 + taper * z_norm)
    px = np.abs(points[..., 0] / (r[0] * taper_scale))
    py = np.abs(points[..., 1] / (r[1] * taper_scale))
    pz = np.abs(points[..., 2] / r[2])
    xy = (px ** (2.0 / e_xy) + py ** (2.0 / e_xy)) ** (e_xy / e_z)
    field = (xy + pz ** (2.0 / e_z)) ** (e_z / 2.0)
    dist_scale = min(r)
    return (field - 1.0) * dist_scale


def point_signed_distance(spec: SyntheticShapeSpec, point: tuple[float, float, float]) -> float:
    np = require_numpy()
    value = signed_distance(spec, np.asarray(point, dtype=np.float32).reshape(1, 3))
    return float(value.reshape(-1)[0])


def analytic_metadata(spec: SyntheticShapeSpec, resolution: int) -> dict[str, object]:
    primitive = str(spec.parameters.get("primitive", "unknown"))
    topology = {
        "box": {"watertight": True, "genus": 0, "component_count": 1},
        "sphere": {"watertight": True, "genus": 0, "component_count": 1},
        "ellipsoid": {"watertight": True, "genus": 0, "component_count": 1},
        "cylinder": {"watertight": True, "genus": 0, "component_count": 1},
        "frustum": {"watertight": True, "genus": 0, "component_count": 1},
        "capsule": {"watertight": True, "genus": 0, "component_count": 1},
        "torus": {"watertight": True, "genus": 1, "component_count": 1},
        "rounded_box": {"watertight": True, "genus": 0, "component_count": 1},
        "superquadric": {"watertight": True, "genus": 0, "component_count": 1},
    }.get(primitive, {"watertight": None, "genus": None, "component_count": None})
    return {
        "ground_truth_level": "analytic_exact",
        "resolution": resolution,
        "primitive": primitive,
        "topology": topology,
        "sdf_formula": _formula_name(primitive),
    }


def _dimensions(value: object, keys: list[str]) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("dimensions must be a mapping")
    return {key: float(value[key]) for key in keys}


def _formula_name(primitive: str) -> str:
    if primitive == "frustum":
        return "linear-radius capped cone approximation"
    if primitive == "superquadric":
        return "implicit superquadric approximation"
    if primitive == "torus":
        return "axis-aligned torus SDF"
    if primitive == "rounded_box":
        return "box SDF offset by corner radius"
    return f"{primitive} SDF"
