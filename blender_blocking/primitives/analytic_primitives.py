"""
Blender-free analytic primitives for primitive fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

try:
    from .primitive_protocol import MeshData, normalize_rotation, signed_power, unit_sphere_samples
except ImportError:  # pragma: no cover - supports direct script execution.
    from primitive_protocol import MeshData, normalize_rotation, signed_power, unit_sphere_samples


def _as_vec3(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return arr


def _positive_vec3(values: object, name: str, minimum: float = 1e-6) -> np.ndarray:
    arr = _as_vec3(values, name)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return np.maximum(arr, minimum)


def _surface_mesh_from_parametric(
    vertices_grid: np.ndarray,
    closed_u: bool,
    closed_v: bool = False,
) -> MeshData:
    if closed_u and not closed_v:
        capped = _surface_mesh_with_collapsed_poles(vertices_grid)
        if capped is not None:
            return capped

    rows, cols, _ = vertices_grid.shape
    faces = []
    row_stop = rows if closed_v else rows - 1
    col_stop = cols if closed_u else cols - 1
    for i in range(row_stop):
        ni = (i + 1) % rows
        for j in range(col_stop):
            nj = (j + 1) % cols
            faces.append(
                (
                    i * cols + j,
                    i * cols + nj,
                    ni * cols + nj,
                    ni * cols + j,
                )
            )
    return MeshData(vertices=vertices_grid.reshape((-1, 3)), faces=tuple(faces))


def _surface_mesh_with_collapsed_poles(vertices_grid: np.ndarray) -> MeshData | None:
    rows, cols, _ = vertices_grid.shape
    if rows < 3 or cols < 3:
        return None

    collapsed_first = _row_collapsed(vertices_grid[0])
    collapsed_last = _row_collapsed(vertices_grid[-1])
    if not collapsed_first and not collapsed_last:
        return None

    vertices: list[np.ndarray] = []
    row_offsets: list[tuple[int, int]] = []
    for row_index in range(rows):
        row = vertices_grid[row_index]
        collapsed = (
            (row_index == 0 and collapsed_first)
            or (row_index == rows - 1 and collapsed_last)
        )
        row_offsets.append((len(vertices), 1 if collapsed else cols))
        if collapsed:
            vertices.append(np.asarray(row[0], dtype=np.float64))
        else:
            vertices.extend(np.asarray(vertex, dtype=np.float64) for vertex in row)

    def vertex_index(row_index: int, col_index: int) -> int:
        offset, count = row_offsets[row_index]
        if count == 1:
            return offset
        return offset + (col_index % cols)

    faces: list[tuple[int, ...]] = []
    for row_index in range(rows - 1):
        lower_count = row_offsets[row_index][1]
        upper_count = row_offsets[row_index + 1][1]
        if lower_count == 1 and upper_count == 1:
            continue
        for col_index in range(cols):
            next_col = (col_index + 1) % cols
            if lower_count == 1:
                faces.append(
                    (
                        vertex_index(row_index, col_index),
                        vertex_index(row_index + 1, col_index),
                        vertex_index(row_index + 1, next_col),
                    )
                )
            elif upper_count == 1:
                faces.append(
                    (
                        vertex_index(row_index, col_index),
                        vertex_index(row_index, next_col),
                        vertex_index(row_index + 1, col_index),
                    )
                )
            else:
                faces.append(
                    (
                        vertex_index(row_index, col_index),
                        vertex_index(row_index, next_col),
                        vertex_index(row_index + 1, next_col),
                        vertex_index(row_index + 1, col_index),
                    )
                )

    return MeshData(vertices=np.asarray(vertices, dtype=np.float64), faces=tuple(faces))


def _row_collapsed(row: np.ndarray) -> bool:
    row = np.asarray(row, dtype=np.float64)
    if len(row) <= 1:
        return True
    scale = max(1.0, float(np.linalg.norm(row[0])))
    spread = np.linalg.norm(row - row[0][None, :], axis=1)
    return bool(float(np.max(spread)) <= 1e-8 * scale)


@dataclass
class EllipsoidPrimitive:
    """Editable ellipsoid primitive with approximate SDF support."""

    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    density: float = 1.0
    confidence: float = 1.0

    def __init__(
        self,
        center: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        radii: Tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
        rotation: np.ndarray | None = None,
        density: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        self.center = _as_vec3(center, "center")
        self.radii = _positive_vec3(radii, "radii")
        self.rotation = normalize_rotation(
            np.eye(3, dtype=np.float64) if rotation is None else rotation
        )
        self.density = float(density)
        self.confidence = float(confidence)

    def _to_local(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return (points - self.center[None, :]) @ self.rotation

    def covariance(self) -> np.ndarray:
        """Return the ellipsoid axis covariance used by projected renderers."""
        return self.rotation @ np.diag(self.radii * self.radii) @ self.rotation.T

    def profile_width_at_world_z(self, z_world: float) -> float:
        """Approximate horizontal profile width at a world-space height."""
        local_z = float((np.asarray((0.0, 0.0, z_world)) - self.center) @ self.rotation[:, 2])
        rz = max(float(self.radii[2]), 1e-9)
        normalized = abs(local_z) / rz
        if normalized >= 1.0:
            return 0.0
        return float(2.0 * self.radii[0] * np.sqrt(max(0.0, 1.0 - normalized * normalized)))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """Approximate signed distance for an ellipsoid."""
        local = self._to_local(points)
        if local.size == 0:
            return np.zeros((0,), dtype=np.float64)
        q = local / self.radii[None, :]
        k0 = np.linalg.norm(q, axis=1)
        k1 = np.linalg.norm(local / (self.radii[None, :] ** 2), axis=1)
        inside_center = k1 <= 1e-12
        safe_k1 = np.where(inside_center, 1.0, k1)
        sdf = k0 * (k0 - 1.0) / safe_k1
        sdf[inside_center] = -float(np.min(self.radii))
        return sdf

    def sample_surface(self, n: int) -> np.ndarray:
        sphere = unit_sphere_samples(n)
        local = sphere * self.radii[None, :]
        return local @ self.rotation.T + self.center[None, :]

    def to_mesh_data(self, resolution: int = 32) -> MeshData:
        resolution = max(6, int(resolution))
        rows = max(4, resolution // 2 + 1)
        cols = resolution
        v = np.linspace(-np.pi / 2.0, np.pi / 2.0, rows)
        u = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
        vv, uu = np.meshgrid(v, u, indexing="ij")
        local = np.stack(
            [
                self.radii[0] * np.cos(vv) * np.cos(uu),
                self.radii[1] * np.cos(vv) * np.sin(uu),
                self.radii[2] * np.sin(vv),
            ],
            axis=2,
        )
        world = local.reshape((-1, 3)) @ self.rotation.T + self.center[None, :]
        return _surface_mesh_from_parametric(world.reshape((rows, cols, 3)), True)

    def to_mesh(self, resolution: int = 32) -> MeshData:
        return self.to_mesh_data(resolution)

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "ellipsoid",
            "center": self.center.tolist(),
            "radii": self.radii.tolist(),
            "rotation": self.rotation.tolist(),
            "density": self.density,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> "EllipsoidPrimitive":
        return cls(
            center=params["center"],
            radii=params["radii"],
            rotation=np.asarray(params.get("rotation", np.eye(3)), dtype=np.float64),
            density=float(params.get("density", 1.0)),
            confidence=float(params.get("confidence", 1.0)),
        )


@dataclass
class SuperquadricPrimitive:
    """Superquadric primitive using Solina-style inside-outside function."""

    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray
    epsilon1: float = 1.0
    epsilon2: float = 1.0
    density: float = 1.0
    confidence: float = 1.0

    def __init__(
        self,
        center: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        radii: Tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
        rotation: np.ndarray | None = None,
        epsilon1: float = 1.0,
        epsilon2: float = 1.0,
        density: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        self.center = _as_vec3(center, "center")
        self.radii = _positive_vec3(radii, "radii")
        self.rotation = normalize_rotation(
            np.eye(3, dtype=np.float64) if rotation is None else rotation
        )
        self.epsilon1 = float(np.clip(epsilon1, 0.05, 4.0))
        self.epsilon2 = float(np.clip(epsilon2, 0.05, 4.0))
        self.density = float(density)
        self.confidence = float(confidence)

    def _to_local(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return (points - self.center[None, :]) @ self.rotation

    def inside_outside(self, points: np.ndarray) -> np.ndarray:
        local = self._to_local(points)
        q = np.abs(local / self.radii[None, :])
        e1 = max(self.epsilon1, 1e-6)
        e2 = max(self.epsilon2, 1e-6)
        xy = np.power(q[:, 0], 2.0 / e2) + np.power(q[:, 1], 2.0 / e2)
        return np.power(
            np.power(xy, e2 / e1) + np.power(q[:, 2], 2.0 / e1),
            e1 / 2.0,
        )

    def profile_width_at_world_z(self, z_world: float) -> float:
        """Approximate horizontal profile width at a world-space height."""
        local_z = float((np.asarray((0.0, 0.0, z_world)) - self.center) @ self.rotation[:, 2])
        rz = max(float(self.radii[2]), 1e-9)
        normalized = abs(local_z) / rz
        if normalized >= 1.0:
            return 0.0
        e1 = max(float(self.epsilon1), 1e-6)
        exponent = 2.0 / e1
        remaining = max(0.0, 1.0 - normalized**exponent)
        return float(2.0 * self.radii[0] * remaining ** (1.0 / exponent))

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        if np.asarray(points).size == 0:
            return np.zeros((0,), dtype=np.float64)
        return (self.inside_outside(points) - 1.0) * float(np.min(self.radii))

    def sample_surface(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        sphere = unit_sphere_samples(n)
        eta = np.arcsin(np.clip(sphere[:, 2], -1.0, 1.0))
        omega = np.arctan2(sphere[:, 1], sphere[:, 0])
        local = self._parametric(eta, omega)
        return local @ self.rotation.T + self.center[None, :]

    def _parametric(self, eta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        ce = signed_power(np.cos(eta), self.epsilon1)
        se = signed_power(np.sin(eta), self.epsilon1)
        co = signed_power(np.cos(omega), self.epsilon2)
        so = signed_power(np.sin(omega), self.epsilon2)
        return np.stack(
            [
                self.radii[0] * ce * co,
                self.radii[1] * ce * so,
                self.radii[2] * se,
            ],
            axis=-1,
        )

    def to_mesh_data(self, resolution: int = 32) -> MeshData:
        resolution = max(8, int(resolution))
        rows = max(5, resolution // 2 + 1)
        cols = resolution
        eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, rows)
        omega = np.linspace(-np.pi, np.pi, cols, endpoint=False)
        ee, oo = np.meshgrid(eta, omega, indexing="ij")
        local = self._parametric(ee, oo)
        world = local.reshape((-1, 3)) @ self.rotation.T + self.center[None, :]
        return _surface_mesh_from_parametric(world.reshape((rows, cols, 3)), True)

    def to_mesh(self, resolution: int = 32) -> MeshData:
        return self.to_mesh_data(resolution)

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "superquadric",
            "center": self.center.tolist(),
            "radii": self.radii.tolist(),
            "rotation": self.rotation.tolist(),
            "epsilon1": self.epsilon1,
            "epsilon2": self.epsilon2,
            "density": self.density,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> "SuperquadricPrimitive":
        return cls(
            center=params["center"],
            radii=params["radii"],
            rotation=np.asarray(params.get("rotation", np.eye(3)), dtype=np.float64),
            epsilon1=float(params.get("epsilon1", 1.0)),
            epsilon2=float(params.get("epsilon2", 1.0)),
            density=float(params.get("density", 1.0)),
            confidence=float(params.get("confidence", 1.0)),
        )


@dataclass
class AnisotropicGaussianPrimitive:
    """Anisotropic Gaussian occupancy primitive."""

    center: np.ndarray
    covariance: np.ndarray
    opacity: float = 1.0
    color: Tuple[float, float, float] | None = None
    semantic_role: str | None = None
    confidence: float = 1.0

    def __init__(
        self,
        center: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        covariance: np.ndarray | None = None,
        opacity: float = 1.0,
        color: Tuple[float, float, float] | None = None,
        semantic_role: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        self.center = _as_vec3(center, "center")
        cov = np.eye(3, dtype=np.float64) if covariance is None else np.asarray(covariance, dtype=np.float64)
        if cov.shape != (3, 3):
            raise ValueError("covariance must have shape (3, 3)")
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        self.covariance = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.color = color
        self.semantic_role = semantic_role
        self.confidence = float(confidence)

    def mahalanobis_squared(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        delta = points - self.center[None, :]
        inv_cov = np.linalg.pinv(self.covariance)
        return np.einsum("ni,ij,nj->n", delta, inv_cov, delta)

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        if np.asarray(points).size == 0:
            return np.zeros((0,), dtype=np.float64)
        eigvals = np.linalg.eigvalsh(self.covariance)
        radius_scale = float(np.sqrt(np.min(np.maximum(eigvals, 1e-8))))
        return (np.sqrt(self.mahalanobis_squared(points)) - 1.0) * radius_scale

    def occupancy_batch(self, points: np.ndarray) -> np.ndarray:
        mahal = self.mahalanobis_squared(points)
        return self.opacity * np.exp(-0.5 * mahal)

    def to_ellipsoid(self, sigma: float = 1.0) -> EllipsoidPrimitive:
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        radii = sigma * np.sqrt(np.maximum(eigvals, 1e-8))
        return EllipsoidPrimitive(
            center=self.center,
            radii=radii,
            rotation=eigvecs,
            density=self.opacity,
            confidence=self.confidence,
        )

    def sample_surface(self, n: int) -> np.ndarray:
        return self.to_ellipsoid().sample_surface(n)

    def to_mesh_data(self, resolution: int = 32) -> MeshData:
        return self.to_ellipsoid().to_mesh_data(resolution)

    def to_mesh(self, resolution: int = 32) -> MeshData:
        return self.to_mesh_data(resolution)

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "anisotropic_gaussian",
            "center": self.center.tolist(),
            "covariance": self.covariance.tolist(),
            "opacity": self.opacity,
            "color": self.color,
            "semantic_role": self.semantic_role,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> "AnisotropicGaussianPrimitive":
        color = params.get("color")
        return cls(
            center=params["center"],
            covariance=np.asarray(params["covariance"], dtype=np.float64),
            opacity=float(params.get("opacity", 1.0)),
            color=tuple(color) if color is not None else None,
            semantic_role=params.get("semantic_role"),
            confidence=float(params.get("confidence", 1.0)),
        )
