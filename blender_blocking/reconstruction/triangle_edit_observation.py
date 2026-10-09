"""Physical corner edit observation for the frozen triangular pebble recipe."""
from __future__ import annotations

import math
import numpy as np

from primitives.rounded_triangle import DEFAULT_VERTICES, RoundedTrianglePrimitive


def triangle_corner_response(before, after, matrix_world, parameters, *, multiplier=1.1):
    """Observe XY support growth with unchanged depth, independently of identity.

The frozen equilateral32/64 outline contains the four cardinal support points.
A radius increase therefore expands both local XY spans by twice the radius
change, multiplied by their existing physical scales. World pose is undone only
for this edit observation; surface comparisons retain their unchanged frame.
"""
    matrix = np.asarray(matrix_world, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or
            abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12 or
            isinstance(multiplier, bool) or not math.isfinite(multiplier) or not 1 < multiplier <= 1.2):
        raise ValueError('triangle physical edit needs a finite invertible pose and bounded radius increase')
    part = RoundedTrianglePrimitive.from_program_parameters(parameters)
    if (part.corner_segments != 32 or part.dome_segments != 64 or
            not np.array_equal(part.vertices_xy, np.asarray(DEFAULT_VERTICES))):
        raise ValueError('corner response is qualified only for the frozen32/64 equilateral outline')
    inverse = np.linalg.inv(matrix[:3, :3])
    local_before = (before.vertices - matrix[:3, 3]) @ inverse.T
    local_after = (after.vertices - matrix[:3, 3]) @ inverse.T
    if not np.isfinite(local_before).all() or not np.isfinite(local_after).all():
        raise ValueError('triangle physical observation is nonfinite')
    old_span, new_span = np.ptp(local_before, axis=0), np.ptp(local_after, axis=0)
    expected = 2 * part.corner_radius * (multiplier - 1) * part.scale_xy
    tolerance = 1e-5
    checks = {'xy_corner_support_response': bool(np.all(np.abs(new_span[:2]-old_span[:2]-expected) <= tolerance)),
              'thickness_preserved': bool(abs(float(new_span[2]-old_span[2])) <= tolerance)}
    return {'protocol':'frozen_triangle_corner_edit_response_v1',
            'passed':all(checks.values()), 'checks':checks,
            'observed_xy_span_delta_world':(new_span[:2]-old_span[:2]).tolist(),
            'expected_xy_span_delta_world':expected.tolist(),
            'thickness_delta_world':float(new_span[2]-old_span[2]),
            'physical_observation_tolerance_world':tolerance,
            'scope':'same frozen cardinal outline under explicit local corner edit; geometry identity/restoration remain exact and independent'}
