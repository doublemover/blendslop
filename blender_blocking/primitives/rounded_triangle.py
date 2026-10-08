"""Editable rounded triangular lens with an authored offset outline.

The outline is a triangle Minkowski-summed with a circular corner radius.
Front/back domes scale that outline by sin(phi); depths may be asymmetric.
The membership field is a signed zero-set field, not Euclidean distance.
"""
from __future__ import annotations

import math
import numpy as np

from .primitive_protocol import MeshData, normalize_rotation

DEFAULT_VERTICES = ((0., .9), (-.7794228634059948, -.45), (.7794228634059948, -.45))


def rounded_triangle_arrays(parameters: dict):
    """Closed lens-shaped pebble around a rounded triangular Minkowski outline."""
    points = np.asarray(parameters["vertices_xy"], dtype=float)
    radius = float(parameters["corner_radius"])
    thickness = float(parameters["thickness"])
    n = parameters["corner_segments"]
    m = parameters["dome_segments"]
    if (points.shape != (3, 2) or not np.isfinite(points).all() or
            not math.isfinite(radius + thickness) or radius <= 0 or thickness <= 0 or
            isinstance(n, bool) or not isinstance(n, int) or
            isinstance(m, bool) or not isinstance(m, int) or
            not 4 <= n <= 128 or not 8 <= m <= 256):
        raise ValueError("invalid rounded triangular reference parameters")
    a, b = points[1] - points[0], points[2] - points[0]
    cross = a[0] * b[1] - a[1] * b[0]
    if cross <= 0:
        raise ValueError("triangle vertices must be counterclockwise and non-collinear")
    center = points.mean(axis=0)
    outline = []
    for i, p in enumerate(points):
        previous = p - points[(i - 1) % 3]
        following = points[(i + 1) % 3] - p
        normals = [np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
                   for edge in (previous, following)]
        start = math.atan2(normals[0][1], normals[0][0])
        stop = math.atan2(normals[1][1], normals[1][0])
        while stop <= start:
            stop += 2 * math.pi
        for angle in np.linspace(start, stop, n + 1):
            outline.append(p + radius * np.array([math.cos(angle), math.sin(angle)]))
    outline = np.asarray(outline)
    count = len(outline)
    front_depth = float(parameters.get("front_depth", thickness / 2))
    back_depth = float(parameters.get("back_depth", thickness / 2))
    if (not np.isfinite([front_depth, back_depth]).all() or
            min(front_depth, back_depth) <= 0 or
            not math.isclose(front_depth + back_depth, thickness, rel_tol=1e-12)):
        raise ValueError("front/back depths must be positive and sum to thickness")
    vertices = [(*center, front_depth)]
    for phi in np.linspace(0, math.pi, m + 1)[1:-1]:
        xy = center + (outline - center) * math.sin(phi)
        vertices.extend((float(x), float(y), (front_depth if phi <= math.pi / 2 else back_depth) * math.cos(phi)) for x, y in xy)
    bottom = len(vertices)
    vertices.append((*center, -back_depth))
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((0, 1 + i, 1 + j))
    for row in range(m - 2):
        a = 1 + row * count
        b = a + count
        for i in range(count):
            j = (i + 1) % count
            faces.extend(((a + i, b + i, b + j), (a + i, b + j, a + j)))
    a = 1 + (m - 2) * count
    for i in range(count):
        j = (i + 1) % count
        faces.append((a + i, bottom, a + j))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)



class RoundedTrianglePrimitive:
    def __init__(self, vertices_xy=DEFAULT_VERTICES, *, corner_radius=.16,
                 thickness=.48, front_fraction=.5, corner_segments=32,
                 dome_segments=64, center=(0., 0., 0.), rotation=None,
                 scale_xy=(1., 1.)):
        self.vertices_xy = np.asarray(vertices_xy, float).copy()
        self.corner_radius = float(corner_radius)
        self.thickness = float(thickness)
        self.front_fraction = float(front_fraction)
        self.corner_segments = corner_segments
        self.dome_segments = dome_segments
        self.center = np.asarray(center, float).copy()
        matrix = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.scale_xy = np.asarray(scale_xy, float).copy()
        if (self.center.shape != (3,) or not np.isfinite(self.center).all() or
                matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
                self.scale_xy.shape != (2,) or not np.isfinite(self.scale_xy).all() or
                (self.scale_xy <= 0).any() or not 0 < self.front_fraction < 1):
            raise ValueError("rounded triangle requires finite pose and positive scales/depths")
        self.rotation = normalize_rotation(matrix)
        # Validate geometric parameters before the compiler creates any objects.
        rounded_triangle_arrays(self.geometry_parameters())

    def geometry_parameters(self):
        return {"vertices_xy": self.vertices_xy.tolist(), "corner_radius": self.corner_radius,
                "thickness": self.thickness, "front_depth": self.thickness * self.front_fraction,
                "back_depth": self.thickness * (1 - self.front_fraction),
                "corner_segments": self.corner_segments, "dome_segments": self.dome_segments}

    def to_mesh_data(self, resolution=None):
        parameters = self.geometry_parameters()
        if resolution is not None:
            if isinstance(resolution, bool) or not isinstance(resolution, int) or not 8 <= resolution <= 128:
                raise ValueError("rounded triangle resolution must be an integer in [8, 128]")
            parameters.update(corner_segments=resolution, dome_segments=min(256, 2 * resolution))
        vertices, faces = rounded_triangle_arrays(parameters)
        vertices[:, :2] *= self.scale_xy
        return MeshData(vertices @ self.rotation.T + self.center, tuple(map(tuple, faces)))

    def to_mesh(self, resolution=None):
        return self.to_mesh_data(resolution)

    def sdf_batch(self, points):
        """Exact sign/zero set for the domed offset outline; distance is a proxy."""
        from .polygon_extrusion import polygon_signed_distance
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("triangle field points must be finite Nx3")
        local = (points - self.center) @ self.rotation
        xy = local[:, :2] / self.scale_xy
        depth = self.thickness * np.where(local[:, 2] >= 0, self.front_fraction, 1 - self.front_fraction)
        axial = np.abs(local[:, 2]) - depth
        scale = np.sqrt(np.maximum(0., 1 - (local[:, 2] / depth) ** 2))
        center = self.vertices_xy.mean(axis=0)
        # Inverse scale is evaluated only away from poles. At the poles the
        # domed outline collapses to its centroid without division by zero.
        active = scale > 1e-12
        profile = np.linalg.norm(xy - center, axis=1)
        if active.any():
            unscaled = center + (xy[active] - center) / scale[active, None]
            profile[active] = (polygon_signed_distance(unscaled, [self.vertices_xy]) - self.corner_radius) * scale[active]
        return np.maximum(profile, axial)

    def sample_surface(self, n):
        from blender_blocking.evaluation.comparable_geometry import sample_surface
        data = self.to_mesh_data()
        if n <= 0:
            return np.empty((0, 3))
        return sample_surface(data.vertices, np.asarray(data.faces), count=n, seed=61007)[0]

    def to_dict(self):
        return {"type": "rounded_triangle", **self.geometry_parameters(),
                "front_fraction": self.front_fraction, "center": self.center.tolist(),
                "rotation": self.rotation.tolist(), "scale_xy": self.scale_xy.tolist()}

    @classmethod
    def from_dict(cls, parameters):
        return cls(parameters["vertices_xy"], corner_radius=parameters["corner_radius"],
                   thickness=parameters["thickness"], front_fraction=parameters.get("front_fraction", .5),
                   corner_segments=parameters.get("corner_segments", 32),
                   dome_segments=parameters.get("dome_segments", 64),
                   center=parameters.get("center", (0., 0., 0.)), rotation=parameters.get("rotation"),
                   scale_xy=parameters.get("scale_xy", (1., 1.)))

    @classmethod
    def from_program_parameters(cls, parameters, *, world=True):
        from blender_blocking.reconstruction.program_transforms import position_vector, rotation_matrix
        return cls(parameters.get("vertices_xy", DEFAULT_VERTICES),
                   corner_radius=parameters.get("corner_radius_world", parameters.get("corner_radius", .16)),
                   thickness=parameters.get("height_world", parameters.get("thickness", .48)),
                   front_fraction=parameters.get("front_fraction", .5),
                   corner_segments=parameters.get("corner_segments", 32),
                   dome_segments=parameters.get("dome_segments", 64),
                   center=position_vector(parameters) if world else (0., 0., 0.),
                   rotation=rotation_matrix(parameters) if world else None,
                   scale_xy=parameters.get("scale_xy", (1., 1.)))
