"""Bounded local taper/bend sharing one field/surface/mesh deformation.

The signed function preserves the inverse-mapped zero set and material sign. It
is an approximate distance proxy, not a Euclidean SDF or native solid certificate.
Analytic Jacobian bounds apply to the continuous map on the base support; finite
triangle contacts and native output qualification remain separate.
"""
from typing import Mapping

import numpy as np

from .analytic_primitives import SuperquadricPrimitive


class DeformedSuperquadricPrimitive(SuperquadricPrimitive):
    MAXIMUM_TAPER = .35
    MAXIMUM_BEND_ANGLE = .75
    MAXIMUM_BEND_RADIAL_RATIO = .45

    def __init__(self, *, taper_x=0., taper_y=0., bend_angle=0., deformation_fit_enabled=False, **base):
        sizes = np.asarray(base.get('radii', (1.,)*3), float)
        if sizes.shape != (3,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError('deformed superquadric sizes must be finite and strictly positive')
        frame = np.asarray(base.get('rotation', np.eye(3)), float)
        if (frame.shape != (3, 3) or not np.isfinite(frame).all()
                or not np.allclose(frame.T@frame, np.eye(3), atol=1e-8, rtol=0.)
                or not np.isclose(np.linalg.det(frame), 1., atol=1e-8, rtol=0.)):
            raise ValueError('deformed superquadric requires a proper input pose; automatic reflection repair is unsupported')
        super().__init__(**base)
        self.taper_x, self.taper_y, self.bend_angle = float(taper_x), float(taper_y), float(bend_angle)
        if not isinstance(deformation_fit_enabled, bool):
            raise ValueError('deformation fitting release must be an explicit Boolean')
        self.deformation_fit_enabled = deformation_fit_enabled
        self.validate_deformation()

    def maximum_bend_angle(self):
        transverse = float(self.radii[0])*(1.+abs(self.taper_x))
        return min(self.MAXIMUM_BEND_ANGLE,
                   self.MAXIMUM_BEND_RADIAL_RATIO*float(self.radii[2])/transverse)

    def validate_deformation(self):
        scalars = [self.taper_x, self.taper_y, self.bend_angle, self.epsilon1,
                   self.epsilon2, self.density, self.confidence]
        if (not np.isfinite(scalars).all() or self.center.shape != (3,) or self.radii.shape != (3,)
                or self.rotation.shape != (3, 3) or not np.isfinite(self.center).all()
                or not np.isfinite(self.radii).all() or np.any(self.radii <= 0)
                or not np.isfinite(self.rotation).all()
                or not isinstance(self.deformation_fit_enabled, bool)
                or not .05 <= self.epsilon1 <= 4. or not .05 <= self.epsilon2 <= 4.
                or abs(self.taper_x) > self.MAXIMUM_TAPER or abs(self.taper_y) > self.MAXIMUM_TAPER):
            raise ValueError('deformed superquadric requires finite positive sizes and bounded taper')
        if (not np.allclose(self.rotation.T@self.rotation, np.eye(3), atol=1e-8, rtol=0.)
                or not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-8, rtol=0.)):
            raise ValueError('deformed superquadric pose must remain a proper orthonormal frame')
        if self.bend_angle and not np.isfinite(float(self.radii[2])/abs(self.bend_angle)):
            raise ValueError('bend radius is numerically unresolved')
        if abs(self.bend_angle) > self.maximum_bend_angle()+1e-14:
            raise ValueError('bend violates the bounded angle/radius Jacobian contract')

    def _points(self, points):
        points = np.asarray(points, float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('deformation points must have a finite final xyz axis')
        return points

    def forward_local(self, points):
        self.validate_deformation()
        points = self._points(points)
        result = points.copy()
        normalized = np.clip(points[..., 2]/self.radii[2], -1., 1.)
        result[..., 0] *= 1.+self.taper_x*normalized
        result[..., 1] *= 1.+self.taper_y*normalized
        angle = abs(self.bend_angle)
        if angle:
            sign = np.sign(self.bend_angle)
            radius = self.radii[2]/angle
            if not np.isfinite(radius):
                raise ValueError('bend radius is numerically unresolved')
            phase = angle*result[..., 2]/self.radii[2]
            xx = sign*result[..., 0]
            # Avoid cancellation in 1-cos at the rigid limit.
            result[..., 0] = sign*(xx*np.cos(phase)+radius*2.*np.sin(phase*.5)**2)
            result[..., 2] = (radius-xx)*np.sin(phase)
        if not np.isfinite(result).all():
            raise ValueError('deformation world coordinates are unresolved')
        return result

    def inverse_local(self, points):
        self.validate_deformation()
        points = self._points(points)
        result = points.copy()
        angle = abs(self.bend_angle)
        if angle:
            sign = np.sign(self.bend_angle)
            radius = self.radii[2]/angle
            if not np.isfinite(radius):
                raise ValueError('bend radius is numerically unresolved')
            xx, zz = sign*points[..., 0], points[..., 2]
            phase = np.arctan2(zz, radius-xx)
            radial = np.hypot(radius-xx, zz)
            # Rationalized R-hypot prevents loss of transverse coordinates at
            # a tiny bend angle. Far queries use the direct, stable branch.
            near = (np.abs(xx/radius) < .25) & (np.abs(zz/radius) < .25)
            numerator = 2.*(xx/radius)-(xx/radius)**2-(zz/radius)**2
            small = radius*numerator/(1.+radial/radius)
            result[..., 0] = sign*np.where(near, small, radius-radial)
            result[..., 2] = phase*(self.radii[2]/angle)
        normalized = np.clip(result[..., 2]/self.radii[2], -1., 1.)
        result[..., 0] /= 1.+self.taper_x*normalized
        result[..., 1] /= 1.+self.taper_y*normalized
        if not np.isfinite(result).all():
            raise ValueError('inverse deformation coordinates are unresolved')
        return result

    def inside_outside(self, points):
        local = self.inverse_local(self._to_local(points))
        q = np.abs(local/self.radii)
        e1, e2 = self.epsilon1, self.epsilon2
        xy = q[:, 0]**(2./e2)+q[:, 1]**(2./e2)
        return (xy**(e2/e1)+q[:, 2]**(2./e1))**(e1*.5)

    def profile_width_at_world_z(self, z_world):
        raise NotImplementedError('deformed world-z profile requires actual local-frame mesh sections; base width proxy is unsupported')

    def _parametric(self, eta, omega):
        return self.forward_local(super()._parametric(eta, omega))

    def deformation_report(self):
        self.validate_deformation()
        ratio = (abs(self.bend_angle)*float(self.radii[0])*(1.+abs(self.taper_x))
                 /float(self.radii[2]))
        lower = (1.-abs(self.taper_x))*(1.-abs(self.taper_y))*(1.-ratio)
        upper = (1.+abs(self.taper_x))*(1.+abs(self.taper_y))*(1.+ratio)
        return {'deformation': 'local_xy_linear_taper_then_xz_circular_bend_v1',
                'maximum_bend_angle': self.maximum_bend_angle(),
                'bend_radial_ratio_bound': ratio, 'jacobian_determinant_lower_bound': lower,
                'jacobian_determinant_upper_bound': upper,
                'bound_scope': 'continuous map on the base superquadric support; finite mesh contacts remain separate',
                'field_semantics': 'inverse-mapped signed zero-set proxy; not Euclidean distance',
                'projection_convexity': 'not assumed; project the actual triangle union',
                'native_qualification': False}

    def to_dict(self):
        self.validate_deformation()
        values = super().to_dict()
        values.update(type='deformed_superquadric', taper_x=self.taper_x, taper_y=self.taper_y,
                      bend_angle=self.bend_angle, deformation_fit_enabled=self.deformation_fit_enabled,
                      deformation_contract='local_xy_linear_taper_then_xz_circular_bend_v1')
        return values

    @classmethod
    def from_dict(cls, params: Mapping[str, object]):
        return cls(center=params.get('center', (0.,)*3), radii=params.get('radii', (1.,)*3),
                   rotation=params.get('rotation', np.eye(3)), epsilon1=params.get('epsilon1', 1.),
                   epsilon2=params.get('epsilon2', 1.), density=params.get('density', 1.),
                   confidence=params.get('confidence', 1.), taper_x=params.get('taper_x', 0.),
                   taper_y=params.get('taper_y', 0.), bend_angle=params.get('bend_angle', 0.),
                   deformation_fit_enabled=params.get('deformation_fit_enabled', False))

    @classmethod
    def from_program_parameters(cls, params, *, world=True):
        from blender_blocking.reconstruction.program_transforms import position_vector, rotation_matrix, DIMENSION_KEYS
        radii = np.asarray([params.get(key, 1.) for key in DIMENSION_KEYS], float)*.5
        return cls(center=position_vector(params) if world else np.zeros(3), radii=radii,
                   rotation=rotation_matrix(params) if world else np.eye(3),
                   epsilon1=params.get('epsilon1', 1.), epsilon2=params.get('epsilon2', 1.),
                   taper_x=params.get('taper_x', 0.), taper_y=params.get('taper_y', 0.),
                   bend_angle=params.get('bend_angle', 0.),
                   deformation_fit_enabled=params.get('deformation_fit_enabled', False))
