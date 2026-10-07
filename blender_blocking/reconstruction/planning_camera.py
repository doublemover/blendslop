"""General orthographic calibration for read-only capture suggestions.

These cameras are not accepted as new ReconstructionTarget observations yet.
The production fitters still explicitly require canonical front/side/top input.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanningOrthographicCamera:
    name: str
    origin: tuple
    right: tuple
    up: tuple
    backward: tuple
    viewport: tuple
    resolution: tuple = (32, 32)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('planning camera requires a nonempty name')
        origin = np.asarray(self.origin, float)
        frame = np.column_stack([self.right, self.up, self.backward]).astype(float)
        bounds = np.asarray(self.viewport, float)
        if (origin.shape != (3,) or frame.shape != (3, 3)
                or not np.isfinite(origin).all() or not np.isfinite(frame).all()
                or not np.allclose(frame.T@frame, np.eye(3), atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.det(frame), 1., atol=1e-12, rtol=0)):
            raise ValueError('planning camera requires a finite proper orthonormal frame')
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]):
            raise ValueError('planning camera requires increasing finite viewport intervals')
        if (len(self.resolution) != 2
                or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in self.resolution)):
            raise ValueError('planning resolution must contain two positive integers')
        for field in ('origin', 'right', 'up', 'backward', 'viewport'):
            object.__setattr__(self, field, tuple(map(float, getattr(self, field))))
        object.__setattr__(self, 'resolution', tuple(map(int, self.resolution)))

    @property
    def frame(self):
        return np.column_stack([self.right, self.up, self.backward])

    @classmethod
    def from_direction(cls, name, *, azimuth_deg, elevation_deg, origin,
                       viewport=(-1., 1., -1., 1.), resolution=(32, 32)):
        angles = np.asarray([azimuth_deg, elevation_deg], float)
        if not np.isfinite(angles).all() or not -90 <= angles[1] <= 90:
            raise ValueError('planning angles must be finite; elevation must be -90..90')
        azimuth, elevation = np.deg2rad(angles)
        backward = np.array([np.sin(azimuth)*np.cos(elevation),
                             -np.cos(azimuth)*np.cos(elevation), np.sin(elevation)])
        reference = np.array([0., 0., 1.])
        if abs(np.dot(reference, backward)) > .999:
            reference = np.array([0., 1., 0.])
        right = np.cross(reference, backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        return cls(name, tuple(origin), tuple(right), tuple(up), tuple(backward),
                   tuple(viewport), tuple(resolution))

    def project_world(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('planning world queries must be finite Nx3 points')
        local = (points-np.asarray(self.origin))@self.frame
        width, height = self.resolution
        u0, u1, v0, v1 = self.viewport
        pixels = np.column_stack([(local[:, 0]-u0)/(u1-u0)*width-.5,
                                  (v1-local[:, 1])/(v1-v0)*height-.5])
        return pixels, local[:, 2]

    def backproject(self, pixels, depth):
        pixels = np.asarray(pixels, float)
        depth = np.asarray(depth, float)
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all()
                or depth.shape not in ((), (len(pixels),)) or not np.isfinite(depth).all()):
            raise ValueError('planning backprojection requires finite Nx2 pixels and scalar/N depth')
        width, height = self.resolution
        u0, u1, v0, v1 = self.viewport
        local = np.column_stack([u0+(pixels[:, 0]+.5)/width*(u1-u0),
                                 v1-(pixels[:, 1]+.5)/height*(v1-v0),
                                 np.broadcast_to(depth, len(pixels))])
        return local@self.frame.T+np.asarray(self.origin)

    def pixel_rays(self, *, depth):
        width, height = self.resolution
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.column_stack([xx.ravel(), yy.ravel()])
        origins = self.backproject(pixels, depth)
        directions = np.broadcast_to(-np.asarray(self.backward), origins.shape).copy()
        return origins, directions

    def to_dict(self):
        return {'name': self.name, 'projection': 'orthographic', 'world_units': 'metres',
                'origin_world': list(self.origin), 'right_world': list(self.right),
                'up_world': list(self.up), 'camera_backward_world': list(self.backward),
                'viewport_world_relative_to_origin': list(self.viewport),
                'resolution': list(self.resolution), 'pixel_convention': 'cell centers; top-down rows',
                'scope': 'read-only planning calibration; no real observation added'}
