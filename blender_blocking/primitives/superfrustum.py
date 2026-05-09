"""
SuperFrustum primitive with differentiable SDF.

Based on arXiv:2512.09201 - Residual Primitive Fitting of 3D Shapes with SuperFrusta
Implementation uses signed distance fields (SDF) from Inigo Quilez's distance functions.

References:
- https://arxiv.org/abs/2512.09201
- https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
import math

try:
    from .primitive_protocol import MeshData
except ImportError:  # pragma: no cover - supports direct script execution.
    from primitive_protocol import MeshData


class SuperFrustum:
    """
    SuperFrustum primitive with 8 parameters and differentiable SDF.

    Parameters (8 total):
        - position (x, y, z): 3 parameters
        - orientation (theta, phi): 2 parameters (spherical coordinates for axis direction)
        - radius_top: 1 parameter
        - radius_bottom: 1 parameter
        - height: 1 parameter

    Can represent:
        - Cylinder: radius_top = radius_bottom
        - Cone: radius_top = 0 OR radius_bottom = 0
        - Sphere: radius_top = radius_bottom = height/2 (approximately)
        - Tapered frustum: radius_top ≠ radius_bottom
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: Tuple[float, float] = (0.0, 0.0),  # (theta, phi) in radians
        radius_bottom: float = 1.0,
        radius_top: float = 0.5,
        height: float = 2.0,
    ) -> None:
        """
        Initialize SuperFrustum.

        Args:
            position: (x, y, z) center position
            orientation: (theta, phi) axis direction in spherical coordinates
                        theta: azimuthal angle (rotation around Z)
                        phi: polar angle (tilt from Z axis)
            radius_bottom: Radius at bottom cap
            radius_top: Radius at top cap
            height: Height along axis
        """
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)
        self.radius_bottom = float(radius_bottom)
        self.radius_top = float(radius_top)
        self.height = float(height)

    def get_axis_vector(self) -> np.ndarray:
        """
        Compute axis direction vector from spherical coordinates.

        Returns:
            Unit vector (3D) representing axis direction
        """
        theta, phi = self.orientation
        # Convert spherical to Cartesian
        # Default axis is +Z (0, 0, 1) when theta=0, phi=0
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z], dtype=np.float64)

    def _axis_rotation_matrix(self) -> np.ndarray:
        """Cached rotation from world-space to the frustum's local z-aligned frame."""
        axis = self.get_axis_vector()
        return self._rotation_matrix_to_z(axis)

    def radius_at_world_z(self, z_world: float) -> float:
        """Return the interpolated profile radius at a world-space z coordinate.

        The radius is derived in local frustum coordinates and clipped to the
        nearest end-cap interpolation range for numerical stability.
        """
        axis_position = np.array([self.position[0], self.position[1], float(z_world)])
        rot = self._axis_rotation_matrix()
        local = np.asarray((axis_position - self.position) @ rot.T, dtype=np.float64)
        local_z = float(local[2])
        half_height = max(self.height * 0.5, 1e-9)
        t = np.clip((local_z + half_height) / (2.0 * half_height), 0.0, 1.0)
        return float((1.0 - t) * self.radius_bottom + t * self.radius_top)

    def profile_width_at_world_z(self, z_world: float) -> float:
        """Return an approximate world-space profile width from the local model."""
        return float(2.0 * self.radius_at_world_z(z_world))

    @staticmethod
    def _rotation_matrix_to_z(axis: np.ndarray) -> np.ndarray:
        """Return a rotation matrix that aligns the given axis with +Z."""
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        if np.allclose(axis, z_axis):
            return np.eye(3, dtype=np.float64)

        if np.allclose(axis, -z_axis):
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            )

        rot_axis = np.cross(axis, z_axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        cos_angle = np.clip(np.dot(axis, z_axis), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        kx, ky, kz = rot_axis
        k_mat = np.array(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            dtype=np.float64,
        )
        eye = np.eye(3, dtype=np.float64)
        return eye + math.sin(angle) * k_mat + (1.0 - math.cos(angle)) * (k_mat @ k_mat)

    def sdf_batch(self, points: np.ndarray) -> np.ndarray:
        """Compute SDF for a batch of points (Nx3)."""
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.zeros((0,), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        p = points - self.position[None, :]
        axis = self.get_axis_vector()
        rot = self._rotation_matrix_to_z(axis)
        p_local = p @ rot.T

        r1 = self.radius_bottom
        r2 = self.radius_top
        h = self.height / 2.0

        p_cone = p_local + np.array([0.0, 0.0, h], dtype=np.float64)
        q0 = np.linalg.norm(p_cone[:, :2], axis=1)
        q1 = p_cone[:, 2]
        q = np.stack([q0, q1], axis=1)

        k1 = np.array([r2, self.height], dtype=np.float64)
        k2 = np.array([r2 - r1, 2.0 * self.height], dtype=np.float64)

        r_edge = np.where(q1 < 0.0, r1, r2)
        ca0 = q0 - np.minimum(q0, r_edge)
        ca1 = np.abs(q1) - self.height
        ca = np.stack([ca0, ca1], axis=1)

        dot_k2 = float(np.dot(k2, k2))
        t = np.clip(((k1 - q) @ k2) / dot_k2, 0.0, 1.0)
        cb = q - k1 + t[:, None] * k2

        s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
        dist = np.sqrt(np.minimum(np.sum(ca * ca, axis=1), np.sum(cb * cb, axis=1)))
        return s * dist

    def gradient_batch(
        self, points: np.ndarray, epsilon: float = 1e-5
    ) -> Dict[str, np.ndarray]:
        """Compute SDF gradients for a batch of points (Nx3)."""
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return {
                "sdf": np.zeros((0,), dtype=np.float64),
                "position": np.zeros((0, 3), dtype=np.float64),
                "orientation": np.zeros((0, 2), dtype=np.float64),
                "radius_bottom": np.zeros((0,), dtype=np.float64),
                "radius_top": np.zeros((0,), dtype=np.float64),
                "height": np.zeros((0,), dtype=np.float64),
            }

        f0 = self.sdf_batch(points)
        grads: Dict[str, np.ndarray] = {"sdf": f0}

        old_pos = self.position.copy()
        grad_pos = np.zeros((len(points), 3), dtype=np.float64)
        for i in range(3):
            pos_plus = old_pos.copy()
            pos_plus[i] += epsilon
            self.position = pos_plus
            f_plus = self.sdf_batch(points)
            grad_pos[:, i] = (f_plus - f0) / epsilon
        self.position = old_pos
        grads["position"] = grad_pos

        old_orient = self.orientation.copy()
        grad_orient = np.zeros((len(points), 2), dtype=np.float64)
        for i in range(2):
            orient_plus = old_orient.copy()
            orient_plus[i] += epsilon
            self.orientation = orient_plus
            f_plus = self.sdf_batch(points)
            grad_orient[:, i] = (f_plus - f0) / epsilon
        self.orientation = old_orient
        grads["orientation"] = grad_orient

        old_rb = self.radius_bottom
        self.radius_bottom = old_rb + epsilon
        f_plus = self.sdf_batch(points)
        grads["radius_bottom"] = (f_plus - f0) / epsilon
        self.radius_bottom = old_rb

        old_rt = self.radius_top
        self.radius_top = old_rt + epsilon
        f_plus = self.sdf_batch(points)
        grads["radius_top"] = (f_plus - f0) / epsilon
        self.radius_top = old_rt

        old_h = self.height
        self.height = old_h + epsilon
        f_plus = self.sdf_batch(points)
        grads["height"] = (f_plus - f0) / epsilon
        self.height = old_h

        return grads

    def sdf(self, point: np.ndarray) -> float:
        """
        Compute signed distance field at a point.

        Uses Inigo Quilez's capped cone/frustum SDF formula.

        Args:
            point: 3D point (x, y, z) as numpy array

        Returns:
            Signed distance (negative inside, positive outside, zero on surface)
        """
        # Transform point to local coordinate system
        # 1. Translate to origin
        p = point - self.position

        # 2. Rotate to align axis with Z
        axis = self.get_axis_vector()
        p_local = self._rotate_to_z_axis(p, axis)

        # 3. Apply capped cone SDF (from Inigo Quilez)
        # Cone is centered at origin, extends from -h/2 to +h/2 along Z
        r1 = self.radius_bottom
        r2 = self.radius_top
        h = self.height / 2.0  # Half-height for centered cone

        # Translate to cone's coordinate system (base at origin, extends upward)
        p_cone = p_local + np.array([0, 0, h])

        # Capped cone SDF (exact formula from IQ)
        q = np.array([np.linalg.norm(p_cone[:2]), p_cone[2]])
        k1 = np.array([r2, self.height])
        k2 = np.array([r2 - r1, 2.0 * self.height])

        ca = np.array(
            [q[0] - min(q[0], r1 if q[1] < 0.0 else r2), abs(q[1]) - self.height]
        )

        cb = q - k1 + k2 * np.clip(np.dot(k1 - q, k2) / np.dot(k2, k2), 0.0, 1.0)

        s = -1.0 if (cb[0] < 0.0 and ca[1] < 0.0) else 1.0

        return s * np.sqrt(min(np.dot(ca, ca), np.dot(cb, cb)))

    def _rotate_to_z_axis(self, point: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """
        Rotate point so that given axis aligns with Z axis.

        Args:
            point: 3D point to rotate
            axis: Current axis direction (unit vector)

        Returns:
            Rotated point
        """
        # Target axis is Z (0, 0, 1)
        z_axis = np.array([0.0, 0.0, 1.0])

        # If axis is already aligned with Z, no rotation needed
        if np.allclose(axis, z_axis):
            return point

        # If axis is opposite to Z, rotate 180 degrees around X
        if np.allclose(axis, -z_axis):
            return np.array([point[0], -point[1], -point[2]])

        # Compute rotation axis (cross product)
        rot_axis = np.cross(axis, z_axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)

        # Compute rotation angle
        cos_angle = np.dot(axis, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Apply Rodrigues' rotation formula
        return self._rodrigues_rotation(point, rot_axis, angle)

    def _rodrigues_rotation(
        self, point: np.ndarray, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """
        Rotate point around axis by angle using Rodrigues' formula.

        Args:
            point: 3D point
            axis: Rotation axis (unit vector)
            angle: Rotation angle in radians

        Returns:
            Rotated point
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rodrigues' rotation formula
        # v_rot = v*cos(a) + (k×v)*sin(a) + k*(k·v)*(1-cos(a))
        return (
            point * cos_a
            + np.cross(axis, point) * sin_a
            + axis * np.dot(axis, point) * (1.0 - cos_a)
        )

    def gradient(
        self, point: np.ndarray, epsilon: float = 1e-5
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients of SDF with respect to all 8 parameters using finite differences.

        Args:
            point: 3D point where to compute gradients
            epsilon: Small perturbation for finite differences

        Returns:
            Dictionary with gradients for each parameter:
                - 'position': (3,) array - gradient w.r.t. position
                - 'orientation': (2,) array - gradient w.r.t. orientation
                - 'radius_bottom': float - gradient w.r.t. bottom radius
                - 'radius_top': float - gradient w.r.t. top radius
                - 'height': float - gradient w.r.t. height
        """
        grads = {}

        # Current SDF value
        f0 = self.sdf(point)

        # Gradient w.r.t. position (3 parameters)
        grad_pos = np.zeros(3)
        for i in range(3):
            pos_plus = self.position.copy()
            pos_plus[i] += epsilon
            old_pos = self.position.copy()
            self.position = pos_plus
            f_plus = self.sdf(point)
            self.position = old_pos
            grad_pos[i] = (f_plus - f0) / epsilon
        grads["position"] = grad_pos

        # Gradient w.r.t. orientation (2 parameters)
        grad_orient = np.zeros(2)
        for i in range(2):
            orient_plus = self.orientation.copy()
            orient_plus[i] += epsilon
            old_orient = self.orientation.copy()
            self.orientation = orient_plus
            f_plus = self.sdf(point)
            self.orientation = old_orient
            grad_orient[i] = (f_plus - f0) / epsilon
        grads["orientation"] = grad_orient

        # Gradient w.r.t. radius_bottom
        old_rb = self.radius_bottom
        self.radius_bottom += epsilon
        f_plus = self.sdf(point)
        self.radius_bottom = old_rb
        grads["radius_bottom"] = (f_plus - f0) / epsilon

        # Gradient w.r.t. radius_top
        old_rt = self.radius_top
        self.radius_top += epsilon
        f_plus = self.sdf(point)
        self.radius_top = old_rt
        grads["radius_top"] = (f_plus - f0) / epsilon

        # Gradient w.r.t. height
        old_h = self.height
        self.height += epsilon
        f_plus = self.sdf(point)
        self.height = old_h
        grads["height"] = (f_plus - f0) / epsilon

        return grads

    def sample_surface(self, n: int) -> np.ndarray:
        """Return deterministic samples on side and cap surfaces."""
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        side_count = max(1, int(round(n * 0.7)))
        cap_count = max(0, n - side_count)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        side_idx = np.arange(side_count, dtype=np.float64)
        z = -self.height / 2.0 + self.height * ((side_idx + 0.5) / side_count)
        t = (z + self.height / 2.0) / max(self.height, 1e-12)
        radius = (1.0 - t) * self.radius_bottom + t * self.radius_top
        theta = side_idx * golden_angle
        local = [
            np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
        ]

        if cap_count > 0:
            bottom_count = cap_count // 2
            top_count = cap_count - bottom_count
            for count, cap_z, cap_radius in (
                (bottom_count, -self.height / 2.0, self.radius_bottom),
                (top_count, self.height / 2.0, self.radius_top),
            ):
                if count <= 0:
                    continue
                idx = np.arange(count, dtype=np.float64)
                theta = idx * golden_angle
                disc_radius = cap_radius * np.sqrt((idx + 0.5) / count)
                local.append(
                    np.stack(
                        [
                            disc_radius * np.cos(theta),
                            disc_radius * np.sin(theta),
                            np.full(count, cap_z, dtype=np.float64),
                        ],
                        axis=1,
                    )
                )

        local_points = np.concatenate(local, axis=0)[:n]
        axis = self.get_axis_vector()
        rot = self._rotation_matrix_to_z(axis)
        return local_points @ rot + self.position[None, :]

    def to_mesh_data(self, resolution: int = 32) -> MeshData:
        """Return a Blender-free mesh approximation of the frustum."""
        resolution = max(6, int(resolution))
        ring_count = max(2, resolution // 4 + 1)
        theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        z_values = np.linspace(-self.height / 2.0, self.height / 2.0, ring_count)

        vertices = []
        for z in z_values:
            t = (z + self.height / 2.0) / max(self.height, 1e-12)
            radius = (1.0 - t) * self.radius_bottom + t * self.radius_top
            vertices.append(
                np.stack(
                    [
                        radius * np.cos(theta),
                        radius * np.sin(theta),
                        np.full(resolution, z, dtype=np.float64),
                    ],
                    axis=1,
                )
            )

        local_vertices = np.concatenate(vertices, axis=0)
        bottom_center = len(local_vertices)
        top_center = bottom_center + 1
        local_vertices = np.vstack(
            [
                local_vertices,
                np.array(
                    [
                        [0.0, 0.0, -self.height / 2.0],
                        [0.0, 0.0, self.height / 2.0],
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        faces = []
        for ring in range(ring_count - 1):
            base = ring * resolution
            next_base = (ring + 1) * resolution
            for j in range(resolution):
                nj = (j + 1) % resolution
                faces.append((base + j, base + nj, next_base + nj, next_base + j))

        top_base = (ring_count - 1) * resolution
        for j in range(resolution):
            nj = (j + 1) % resolution
            faces.append((bottom_center, nj, j))
            faces.append((top_center, top_base + j, top_base + nj))

        axis = self.get_axis_vector()
        rot = self._rotation_matrix_to_z(axis)
        world_vertices = local_vertices @ rot + self.position[None, :]
        return MeshData(vertices=world_vertices, faces=tuple(faces))

    def to_mesh(self, resolution: int = 32) -> MeshData:
        """Compatibility alias for protocol consumers."""
        return self.to_mesh_data(resolution)

    def to_dict(self) -> Dict[str, object]:
        """
        Export parameters as dictionary.

        Returns:
            Dictionary with all 8 parameters
        """
        return {
            "type": "superfrustum",
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "radius_bottom": self.radius_bottom,
            "radius_top": self.radius_top,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, object]) -> "SuperFrustum":
        """
        Create SuperFrustum from parameter dictionary.

        Args:
            params: Dictionary with parameters

        Returns:
            SuperFrustum instance
        """
        return cls(
            position=tuple(params["position"]),
            orientation=tuple(params["orientation"]),
            radius_bottom=params["radius_bottom"],
            radius_top=params["radius_top"],
            height=params["height"],
        )

    def __repr__(self) -> str:
        """Return a compact, readable summary for debugging/logging."""
        return (
            f"SuperFrustum(pos={self.position}, "
            f"orient={self.orientation}, "
            f"r_bot={self.radius_bottom:.3f}, "
            f"r_top={self.radius_top:.3f}, "
            f"h={self.height:.3f})"
        )
