"""
Shared primitive protocols and mesh containers.

These types intentionally stay Blender-free so primitive fitting, tests, and
research backends can run in plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Sequence, Tuple, TypeVar

import numpy as np


@dataclass(frozen=True)
class MeshData:
    """Plain mesh data suitable for Blender import or OBJ-style export."""

    vertices: np.ndarray
    faces: Tuple[Tuple[int, ...], ...]
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        object.__setattr__(self, "vertices", vertices)

        if self.normals is not None:
            normals = np.asarray(self.normals, dtype=np.float64)
            if normals.shape != vertices.shape:
                raise ValueError("normals must match vertices shape")
            object.__setattr__(self, "normals", normals)

        clean_faces = tuple(tuple(int(i) for i in face) for face in self.faces)
        object.__setattr__(self, "faces", clean_faces)


class FittablePrimitive(Protocol):
    """Protocol consumed by ResFit modules."""

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """Return signed distance values for points with shape (N, 3)."""

    def sample_surface(self, n: int) -> np.ndarray:
        """Return deterministic surface samples with shape (n, 3)."""

    def to_mesh_data(self, resolution: int = 32) -> MeshData:
        """Return a Blender-free mesh approximation."""

    def to_mesh(self, resolution: int = 32) -> MeshData:
        """Compatibility alias for callers that expect to_mesh."""

    def to_dict(self) -> Dict[str, object]:
        """Serialize primitive parameters."""


PrimitiveT = TypeVar("PrimitiveT", bound=FittablePrimitive)


def normalize_rotation(rotation: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a right-handed orthonormal 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def unit_sphere_samples(n: int) -> np.ndarray:
    """Deterministic approximately even samples on the unit sphere."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    indices = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), z],
        axis=1,
    )


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    """Apply sign(x) * abs(x) ** exponent with stable zero handling."""
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.abs(values) < 1e-12, 0.0, values)
    return np.sign(values) * np.power(np.abs(values), exponent)
