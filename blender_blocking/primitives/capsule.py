"""Connected editable capsule with welded poles and no Boolean assembly."""
from __future__ import annotations
import math
import numpy as np
from .primitive_protocol import MeshData, normalize_rotation


class CapsulePrimitive:
    def __init__(self, *, radius=.4, segment_height=1.2, center=(0., 0., 0.), rotation=None):
        self.radius, self.segment_height = float(radius), float(segment_height)
        self.center = np.asarray(center, float).copy()
        matrix = np.eye(3) if rotation is None else np.asarray(rotation, float)
        if (not np.isfinite([self.radius, self.segment_height]).all() or self.radius <= 0 or
                self.segment_height < 0 or self.center.shape != (3,) or not np.isfinite(self.center).all() or
                matrix.shape != (3, 3) or not np.isfinite(matrix).all()):
            raise ValueError("capsule requires finite pose, positive radius and nonnegative segment height")
        self.rotation = normalize_rotation(matrix)

    def sdf_batch(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("capsule distance points must be finite Nx3")
        local = (points - self.center) @ self.rotation
        local[:, 2] -= np.clip(local[:, 2], -self.segment_height / 2, self.segment_height / 2)
        return np.linalg.norm(local, axis=1) - self.radius

    def to_mesh_data(self, resolution=48):
        if isinstance(resolution, bool) or not isinstance(resolution, int) or not 8 <= resolution <= 256:
            raise ValueError("capsule resolution must be an integer in [8, 256]")
        n, hemisphere_steps = resolution, max(4, resolution // 4)
        rings = []
        for theta in np.linspace(-math.pi / 2, 0, hemisphere_steps + 1)[1:]:
            rings.append((self.radius * math.cos(theta), -self.segment_height / 2 + self.radius * math.sin(theta)))
        if self.segment_height > 0:
            rings.append((self.radius, self.segment_height / 2))
        for theta in np.linspace(0, math.pi / 2, hemisphere_steps + 1)[1:-1]:
            rings.append((self.radius * math.cos(theta), self.segment_height / 2 + self.radius * math.sin(theta)))
        vertices = [(0., 0., -self.segment_height / 2 - self.radius)]
        for radius, z in rings:
            vertices.extend((radius * math.cos(2 * math.pi * i / n),
                             radius * math.sin(2 * math.pi * i / n), z) for i in range(n))
        top = len(vertices)
        vertices.append((0., 0., self.segment_height / 2 + self.radius))
        faces = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)]
        for row in range(len(rings) - 1):
            a, b = 1 + row * n, 1 + (row + 1) * n
            for i in range(n):
                j = (i + 1) % n
                faces.extend(((a + i, a + j, b + j), (a + i, b + j, b + i)))
        a = 1 + (len(rings) - 1) * n
        faces.extend((a + i, a + (i + 1) % n, top) for i in range(n))
        return MeshData(np.asarray(vertices) @ self.rotation.T + self.center, tuple(faces))

    def to_mesh(self, resolution=48):
        return self.to_mesh_data(resolution)

    def sample_surface(self, n):
        from blender_blocking.evaluation.comparable_geometry import sample_surface
        if n <= 0:
            return np.empty((0, 3))
        data = self.to_mesh_data()
        return sample_surface(data.vertices, np.asarray(data.faces), count=n, seed=61007)[0]

    def to_dict(self):
        return {"type": "capsule", "radius": self.radius, "segment_height": self.segment_height,
                "center": self.center.tolist(), "rotation": self.rotation.tolist()}

    @classmethod
    def from_dict(cls, parameters):
        return cls(radius=parameters["radius"], segment_height=parameters["segment_height"],
                   center=parameters.get("center", (0., 0., 0.)), rotation=parameters.get("rotation"))

    @classmethod
    def from_program_parameters(cls, parameters, *, world=True):
        from blender_blocking.reconstruction.program_transforms import position_vector, rotation_matrix
        radius = float(parameters.get("radius_world", parameters.get("width_world", .8) / 2))
        segment = parameters.get("segment_height_world", float(parameters.get("height_world", 2.)) - 2 * radius)
        return cls(radius=radius, segment_height=segment,
                   center=position_vector(parameters) if world else (0., 0., 0.),
                   rotation=rotation_matrix(parameters) if world else None)
