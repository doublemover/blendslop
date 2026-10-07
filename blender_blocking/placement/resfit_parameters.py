"""Family-specific, dimensionless updates for editable fitting primitives.

Scalar references remain tuples so optimizer records and callers can retain them.
Updates use local rotations, log-positive sizes, and Gaussian Cholesky factors.
Only NumPy is needed; no Blender scene state is stored here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

ParameterRef = Tuple[int, str, int | None]
CHOLESKY_ENTRIES = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
POSITIVE_ATTRIBUTES = frozenset((
    "radii", "radius_bottom", "radius_top", "height", "epsilon1", "epsilon2", "section_exponent",
))


def discover_primitive_parameters(primitives: Sequence[object]) -> list[ParameterRef]:
    """Expose geometry parameters without redundant Gaussian rotation variables."""
    refs: list[ParameterRef] = []
    for index, primitive in enumerate(primitives):
        for attr in ("center", "position", "radii"):
            if hasattr(primitive, attr):
                refs.extend((index, attr, axis) for axis in range(3))
        for attr in ("radius_bottom", "radius_top", "height", "epsilon1", "epsilon2", "section_exponent"):
            if hasattr(primitive, attr):
                refs.append((index, attr, None))
        if getattr(primitive, 'deformation_fit_enabled', False) is True:
            refs.extend((index, name, None) for name in ('taper_x', 'taper_y', 'bend_angle')
                        if hasattr(primitive, name))
        if hasattr(primitive, "rotation"):
            refs.extend((index, "rotation", axis) for axis in range(3))
        if hasattr(primitive, "orientation"):
            refs.extend((index, "orientation", axis) for axis in range(2))
        covariance = getattr(primitive, "covariance", None)
        if covariance is not None and not callable(covariance):
            refs.extend((index, "covariance_cholesky", axis) for axis in range(6))
        if hasattr(primitive, "opacity"):
            refs.append((index, "opacity", None))
        if hasattr(primitive,'section_knots_normalized'):
            for row in range(len(primitive.section_knots_normalized)):
                refs.extend((index,'section_knots_normalized',5*row+column) for column in (1,2,3,4))
    return refs


def primitive_length_scale(primitive: object) -> float:
    """A translation step is relative to this part's characteristic radius."""
    lengths = []
    if hasattr(primitive, "radii"):
        lengths.extend(np.asarray(primitive.radii, dtype=float).reshape(-1))
    for attr in ("radius_bottom", "radius_top"):
        if hasattr(primitive, attr):
            lengths.append(float(getattr(primitive, attr)))
    if hasattr(primitive, "height"):
        lengths.append(float(primitive.height) * 0.5)
    covariance = getattr(primitive, "covariance", None)
    if covariance is not None and not callable(covariance):
        lengths.extend(np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0)))
    valid = [float(value) for value in lengths if np.isfinite(value) and value > 0.0]
    return max(1e-6, float(np.median(valid))) if valid else 1.0


def parameter_scale(primitive: object, attr: str, axis: int | None = None) -> float:
    """Physical scale for additive parameters; log sizes/angles are dimensionless."""
    if attr in ("center", "position"):
        return primitive_length_scale(primitive)
    if attr == "covariance_cholesky":
        row, col = CHOLESKY_ENTRIES[int(axis)]
        if row != col:
            return primitive_length_scale(primitive)
    return 1.0


def _rotation_increment(axis: int, angle: float) -> np.ndarray:
    vector = np.zeros(3, dtype=float)
    vector[axis] = 1.0
    x, y, z = vector
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
    antisymmetric = np.array((rotation[2, 1] - rotation[1, 2],
                             rotation[0, 2] - rotation[2, 0],
                             rotation[1, 0] - rotation[0, 1]))
    if angle < 1e-7:
        return antisymmetric * 0.5
    if np.pi - angle < 1e-6:
        _, vectors = np.linalg.eigh((rotation + rotation.T) * 0.5)
        axis = vectors[:, -1]
        if np.dot(axis, antisymmetric) < 0.0:
            axis = -axis
        return axis * angle
    return antisymmetric * (angle / (2.0 * np.sin(angle)))


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def orientation_tangents(primitive: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axis and two orthogonal tilt directions, including at either pole."""
    theta, phi = np.asarray(primitive.orientation, dtype=float)
    axis = np.array((np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)))
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(reference, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return axis, first, second


def _cholesky(primitive: object) -> np.ndarray:
    covariance = np.asarray(primitive.covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("Gaussian covariance must be a finite 3x3 matrix")
    return np.linalg.cholesky((covariance + covariance.T) * 0.5)


def _set_cholesky(primitive: object, factor: np.ndarray, bounds: object) -> None:
    covariance = factor @ factor.T
    # Match the Gaussian primitive's representable floor, and bound principal sizes.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(1e-8, float(bounds.min_radius) ** 2)
    ceiling = float(bounds.max_radius) ** 2
    if eigenvalues.min() < floor or eigenvalues.max() > ceiling:
        covariance = (eigenvectors * np.clip(eigenvalues, floor, ceiling)) @ eigenvectors.T
    primitive.covariance = (covariance + covariance.T) * 0.5


def clip_parameter_value(attr: str, value: float, bounds: object) -> float:
    if not np.isfinite(value):
        raise ValueError("parameter values must be finite")
    if attr in ("radii", "radius_bottom", "radius_top"):
        return float(np.clip(value, max(1e-12, bounds.min_radius), bounds.max_radius))
    if attr == "height":
        return float(np.clip(value, max(1e-12, bounds.min_height), bounds.max_height))
    if attr in ("epsilon1", "epsilon2"):
        return float(np.clip(value, max(1e-12, bounds.min_exponent), bounds.max_exponent))
    if attr == 'section_exponent':
        return float(np.clip(value,.15,2.))
    if attr == "opacity":
        return float(np.clip(value, bounds.min_opacity, bounds.max_opacity))
    return float(value)


def parameter_value(primitive: object, attr: str, axis: int | None) -> float:
    """Absolute scalar view for callers; fitting uses local increment updates below."""
    if attr == "rotation":
        return float(_rotation_vector(primitive.rotation)[int(axis)])
    if attr == "covariance_cholesky":
        row, col = CHOLESKY_ENTRIES[int(axis)]
        value = float(_cholesky(primitive)[row, col])
        return float(np.log(value)) if row == col else value
    if attr == 'section_knots_normalized':
        return float(primitive.section_knots_normalized.ravel()[int(axis)])
    value = getattr(primitive, attr)
    return float(value) if axis is None else float(np.asarray(value)[axis])


def set_parameter_value(primitive: object, attr: str, axis: int | None,
                        value: float, bounds: object) -> None:
    """Set an absolute scalar view, retaining a valid matrix for coupled parameters."""
    value = clip_parameter_value(attr, value, bounds)
    if attr in ('taper_x', 'taper_y', 'bend_angle'):
        if getattr(primitive, 'deformation_fit_enabled', False) is not True:
            raise ValueError('deformation controls require explicit fitting release after coarse pose')
        old = getattr(primitive, attr)
        setattr(primitive, attr, value)
        try:
            primitive.validate_deformation()
        except Exception:
            setattr(primitive, attr, old)
            raise
        return
    if attr == 'section_knots_normalized':
        row,column=divmod(int(axis),5)
        if column not in (1,2,3,4):
            raise ValueError('axial knots are fixed to preserve sweep topology/order')
        primitive.section_knots_normalized[row,column]=np.clip(value,1e-4,4.) if column>=3 else np.clip(value,-4.,4.)
        return
    if attr == "rotation":
        vector = _rotation_vector(primitive.rotation)
        vector[int(axis)] = value
        primitive.rotation = _rotation_from_vector(vector)
        return
    if attr == "covariance_cholesky":
        factor = _cholesky(primitive)
        row, col = CHOLESKY_ENTRIES[int(axis)]
        if row == col:
            low = np.log(max(1e-4, float(bounds.min_radius)))
            factor[row, col] = np.exp(np.clip(value, low, np.log(bounds.max_radius)))
        else:
            factor[row, col] = np.clip(value, -bounds.max_radius, bounds.max_radius)
        _set_cholesky(primitive, factor, bounds)
        return
    original = getattr(primitive, attr)
    original = original.copy() if isinstance(original, np.ndarray) else original
    if axis is None:
        setattr(primitive, attr, value)
    else:
        vector = np.asarray(getattr(primitive, attr), dtype=float).copy()
        vector[int(axis)] = value
        setattr(primitive, attr, vector)
    validator = getattr(primitive, 'validate_deformation', None)
    if validator is not None:
        try:
            validator()
        except Exception:
            setattr(primitive, attr, original)
            raise


def apply_parameter_increment(primitives: Sequence[object], ref: ParameterRef,
                              delta: float, bounds: object) -> None:
    """Apply a dimensionless local step to the current primitive.

    Callers must restore their baseline before competing trials. Rotation axes
    are local to the current frame, orientation axes are tangent tilt directions,
    and Cholesky diagonal increments are log multipliers.
    """
    if not np.isfinite(delta):
        raise ValueError("parameter increment must be finite")
    if delta == 0.0:
        return
    index, attr, axis = ref
    primitive = primitives[index]
    if attr == 'section_knots_normalized':
        current=parameter_value(primitive,attr,axis)
        column=int(axis)%5
        value=current*np.exp(np.clip(delta,-20.,20.)) if column>=3 else current+.25*delta
        set_parameter_value(primitive,attr,axis,value,bounds)
        return
    if attr == "rotation":
        primitive.rotation = np.asarray(primitive.rotation) @ _rotation_increment(int(axis), delta)
        return
    if attr == "orientation":
        direction, first, second = orientation_tangents(primitive)
        tangent = (first, second)[int(axis)]
        direction = np.cos(delta) * direction + np.sin(delta) * tangent
        primitive.orientation = np.array((np.arctan2(direction[1], direction[0]),
                                           np.arccos(np.clip(direction[2], -1.0, 1.0))))
        return
    if attr == "covariance_cholesky":
        factor = _cholesky(primitive)
        row, col = CHOLESKY_ENTRIES[int(axis)]
        if row == col:
            log_value = np.log(factor[row, col]) + delta
            factor[row, col] = np.exp(np.clip(log_value,
                np.log(max(1e-4, float(bounds.min_radius))), np.log(bounds.max_radius)))
        else:
            factor[row, col] = np.clip(factor[row, col] + delta * parameter_scale(primitive, attr, axis),
                                      -bounds.max_radius, bounds.max_radius)
        _set_cholesky(primitive, factor, bounds)
        return
    value = parameter_value(primitive, attr, axis)
    if attr in POSITIVE_ATTRIBUTES:
        value = np.exp(np.clip(np.log(max(value, 1e-12)) + delta, -700.0, 700.0))
    else:
        value += delta * parameter_scale(primitive, attr, axis)
    set_parameter_value(primitive, attr, axis, value, bounds)


@dataclass(frozen=True)
class PrimitiveParameterJacobian:
    """World-space ellipse/Gaussian derivatives for one local fitting coordinate.

    Covariance is the 3D matrix projected by the ellipse renderer; opacity includes
    confidence, matching its rendered alpha. The map is differentiable away from
    fitting bounds. Bound projection belongs to the optimizer's trial application.
    """

    center: np.ndarray
    covariance: np.ndarray
    opacity: float = 0.0


def parameter_jacobian(primitive: object, attr: str,
                       axis: int | None = None) -> PrimitiveParameterJacobian:
    """Analytic derivative with respect to a dimensionless local increment.

    Superquadric exponent changes have no effect on this ellipse approximation.
    Frustum tilt/size is not represented by that renderer and is rejected here.
    At alpha saturation endpoints we retain the inward derivative; projected
    fitting steps enforce the valid interval rather than freezing opaque seeds.
    """
    center = np.zeros(3, dtype=float)
    covariance = np.zeros((3, 3), dtype=float)
    opacity = 0.0
    if attr in ("center", "position"):
        center[int(axis)] = parameter_scale(primitive, attr, axis)
    elif attr == "radii":
        rotation = np.asarray(getattr(primitive, "rotation", np.eye(3)), dtype=float)
        direction = rotation[:, int(axis)]
        radius = float(np.asarray(primitive.radii)[int(axis)])
        covariance = 2.0 * radius * radius * np.outer(direction, direction)
    elif attr == "rotation":
        rotation = np.asarray(primitive.rotation, dtype=float)
        x, y, z = np.eye(3)[int(axis)]
        skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        diagonal = np.diag(np.asarray(primitive.radii, dtype=float) ** 2)
        covariance = rotation @ (skew @ diagonal - diagonal @ skew) @ rotation.T
    elif attr == "covariance_cholesky":
        factor = _cholesky(primitive)
        row, col = CHOLESKY_ENTRIES[int(axis)]
        derivative = np.zeros((3, 3), dtype=float)
        derivative[row, col] = (factor[row, col] if row == col else
                                parameter_scale(primitive, attr, axis))
        covariance = derivative @ factor.T + factor @ derivative.T
    elif attr == "opacity":
        confidence = float(getattr(primitive, "confidence", 1.0))
        alpha = float(primitive.opacity) * confidence
        opacity = confidence if 0.0 <= alpha <= 1.0 else 0.0
    elif attr not in ("epsilon1", "epsilon2"):
        raise ValueError(f"ellipse renderer does not represent parameter {attr!r}")
    return PrimitiveParameterJacobian(center, covariance, opacity)


def parameter_jacobians(primitives: Sequence[object]) -> dict[ParameterRef, PrimitiveParameterJacobian]:
    """Return the complete adapter Jacobian map for ellipse-supported parts."""
    return {ref: parameter_jacobian(primitives[ref[0]], ref[1], ref[2])
            for ref in discover_primitive_parameters(primitives)}


def pullback_render_gradients(primitives: Sequence[object], center_gradients: np.ndarray,
                              covariance_gradients: np.ndarray,
                              opacity_gradients: np.ndarray) -> dict[ParameterRef, float]:
    """Chain world-space renderer derivatives into dimensionless fitter coordinates."""
    centers = np.asarray(center_gradients, dtype=float)
    covariances = np.asarray(covariance_gradients, dtype=float)
    opacities = np.asarray(opacity_gradients, dtype=float)
    count = len(primitives)
    if centers.shape != (count, 3) or covariances.shape != (count, 3, 3) or opacities.shape != (count,):
        raise ValueError("render gradient shapes must be (parts,3), (parts,3,3), and (parts,)")
    if not all(np.isfinite(values).all() for values in (centers, covariances, opacities)):
        raise ValueError("render gradients must be finite")
    return {ref: float(np.dot(centers[ref[0]], jacobian.center)
                       + np.sum(covariances[ref[0]] * jacobian.covariance)
                       + opacities[ref[0]] * jacobian.opacity)
            for ref, jacobian in parameter_jacobians(primitives).items()}
