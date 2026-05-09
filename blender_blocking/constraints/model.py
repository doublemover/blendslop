"""Human correction constraint data contracts.

The constraints package is intentionally independent from Blender and from the
new reconstruction modules.  It stores validated, JSON-safe constraint records
and exposes helpers that downstream stages can consume through duck-typed APIs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, Literal, Optional, Tuple, Union


SUPPORTED_VERSION = 1

ScribbleKind = Literal["foreground", "background", "unknown"]
SymmetryPlane = Literal["xz", "yz", "xy", "custom"]
PolarityHint = Literal["foreground_dark", "foreground_light", "auto"]
ViewRole = Literal["front", "side", "top", "custom"]
PlaneKind = Literal["ground", "contact"]
WorldAxis = Literal["x", "y", "z"]
ComponentHintKind = Literal[
    "preserve_holes",
    "preserve_components",
    "force_separate_parts",
    "ignore_component",
    "ignore_crop",
]

_VIEW_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SCRIBBLE_KINDS = {"foreground", "background", "unknown"}
_SYMMETRY_PLANES = {"xz", "yz", "xy", "custom"}
_POLARITY_HINTS = {"foreground_dark", "foreground_light", "auto"}
_VIEW_ROLES = {"front", "side", "top", "custom"}
_PLANE_KINDS = {"ground", "contact"}
_WORLD_AXES = {"x", "y", "z"}
_COMPONENT_HINTS = {
    "preserve_holes",
    "preserve_components",
    "force_separate_parts",
    "ignore_component",
    "ignore_crop",
}
_DIMENSION_AXIS = {
    "width": 0,
    "x": 0,
    "depth": 1,
    "y": 1,
    "height": 2,
    "z": 2,
}


def _ensure_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ensure_nonnegative(value: Any, name: str) -> float:
    result = _ensure_finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _ensure_positive(value: Any, name: str) -> float:
    result = _ensure_finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def _ensure_confidence(value: Any, name: str = "confidence") -> float:
    result = _ensure_finite(value, name)
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _ensure_bool_or_none(value: Optional[bool], name: str = "hard") -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool or None")
    return value


def _ensure_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ensure_view(value: Any, name: str = "view") -> str:
    view = _ensure_text(value, name)
    if not _VIEW_RE.match(view):
        raise ValueError(
            f"{name} may contain only letters, numbers, underscore, dot, colon, or dash"
        )
    return view


def _ensure_choice(value: Any, choices: set[str], name: str) -> str:
    text = _ensure_text(value, name)
    if text not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return text


def _ensure_points2(
    points: Sequence[Sequence[Any]], name: str = "points_px"
) -> Tuple[Tuple[float, float], ...]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of 2D points")
    result = []
    for index, point in enumerate(points):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError(f"{name}[{index}] must be a 2D point")
        if len(point) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two values")
        result.append(
            (
                _ensure_finite(point[0], f"{name}[{index}][0]"),
                _ensure_finite(point[1], f"{name}[{index}][1]"),
            )
        )
    if not result:
        raise ValueError(f"{name} must contain at least one point")
    return tuple(result)


def _ensure_tuple3(values: Sequence[Any], name: str) -> Tuple[float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a 3D vector")
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(_ensure_finite(values[i], f"{name}[{i}]") for i in range(3))
    return result  # type: ignore[return-value]


def _ensure_bbox(values: Sequence[Any], name: str = "bbox_px") -> Tuple[float, float, float, float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a 4-value bbox")
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    x0, y0, x1, y1 = (
        _ensure_finite(values[0], f"{name}[0]"),
        _ensure_finite(values[1], f"{name}[1]"),
        _ensure_finite(values[2], f"{name}[2]"),
        _ensure_finite(values[3], f"{name}[3]"),
    )
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"{name} max values must be greater than min values")
    return (x0, y0, x1, y1)


def _json_safe(value: Any) -> Any:
    """Return a manifest-safe JSON value without importing project helpers."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def stable_constraint_id(payload: Mapping[str, Any]) -> str:
    """Build a deterministic short ID from a constraint payload."""
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ScribbleConstraint:
    view: str
    kind: ScribbleKind
    points_px: Tuple[Tuple[float, float], ...]
    brush_radius_px: float
    confidence: float
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _ensure_view(self.view))
        object.__setattr__(self, "kind", _ensure_choice(self.kind, _SCRIBBLE_KINDS, "kind"))
        object.__setattr__(self, "points_px", _ensure_points2(self.points_px))
        object.__setattr__(
            self, "brush_radius_px", _ensure_nonnegative(self.brush_radius_px, "brush_radius_px")
        )
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "scribble",
            "view": self.view,
            "kind": self.kind,
            "points_px": [list(point) for point in self.points_px],
            "brush_radius_px": self.brush_radius_px,
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class AxisConstraint:
    axis_name: str
    direction_world: Tuple[float, float, float]
    confidence: float
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_name", _ensure_text(self.axis_name, "axis_name"))
        direction = _ensure_tuple3(self.direction_world, "direction_world")
        length = math.sqrt(sum(v * v for v in direction))
        if length <= 0:
            raise ValueError("direction_world must be non-zero")
        object.__setattr__(self, "direction_world", tuple(v / length for v in direction))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "axis",
            "axis_name": self.axis_name,
            "direction_world": list(self.direction_world),
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class DimensionConstraint:
    name: str
    value_u: float
    tolerance_u: float
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _ensure_text(self.name, "name").lower())
        object.__setattr__(self, "value_u", _ensure_positive(self.value_u, "value_u"))
        object.__setattr__(self, "tolerance_u", _ensure_nonnegative(self.tolerance_u, "tolerance_u"))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    @property
    def axis_index(self) -> Optional[int]:
        return _DIMENSION_AXIS.get(self.name)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "dimension",
            "name": self.name,
            "value_u": self.value_u,
            "tolerance_u": self.tolerance_u,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class SymmetryConstraint:
    plane: SymmetryPlane
    confidence: float
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane", _ensure_choice(self.plane, _SYMMETRY_PLANES, "plane"))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "symmetry",
            "plane": self.plane,
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class CenterlineConstraint:
    view: str
    points_px: Tuple[Tuple[float, float], ...]
    tolerance_px: float
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _ensure_view(self.view))
        object.__setattr__(self, "points_px", _ensure_points2(self.points_px))
        object.__setattr__(self, "tolerance_px", _ensure_nonnegative(self.tolerance_px, "tolerance_px"))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "centerline",
            "view": self.view,
            "points_px": [list(point) for point in self.points_px],
            "tolerance_px": self.tolerance_px,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class BBoxConstraint:
    view: str
    bbox_px: Tuple[float, float, float, float]
    confidence: float = 1.0
    tolerance_px: float = 0.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _ensure_view(self.view))
        object.__setattr__(self, "bbox_px", _ensure_bbox(self.bbox_px))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "tolerance_px", _ensure_nonnegative(self.tolerance_px, "tolerance_px"))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "bbox",
            "view": self.view,
            "bbox_px": list(self.bbox_px),
            "confidence": self.confidence,
            "tolerance_px": self.tolerance_px,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class PolarityConstraint:
    view: str
    polarity: PolarityHint
    confidence: float = 1.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _ensure_view(self.view))
        object.__setattr__(self, "polarity", _ensure_choice(self.polarity, _POLARITY_HINTS, "polarity"))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "polarity",
            "view": self.view,
            "polarity": self.polarity,
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class ViewRoleConstraint:
    view: str
    role: ViewRole
    confidence: float = 1.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _ensure_view(self.view))
        object.__setattr__(self, "role", _ensure_choice(self.role, _VIEW_ROLES, "role"))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "view_role",
            "view": self.view,
            "role": self.role,
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class PlaneConstraint:
    kind: PlaneKind
    axis: WorldAxis = "z"
    value_u: float = 0.0
    tolerance_u: float = 0.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _ensure_choice(self.kind, _PLANE_KINDS, "kind"))
        object.__setattr__(self, "axis", _ensure_choice(self.axis, _WORLD_AXES, "axis"))
        object.__setattr__(self, "value_u", _ensure_finite(self.value_u, "value_u"))
        object.__setattr__(self, "tolerance_u", _ensure_nonnegative(self.tolerance_u, "tolerance_u"))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    @property
    def axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.axis]

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "plane",
            "kind": self.kind,
            "axis": self.axis,
            "value_u": self.value_u,
            "tolerance_u": self.tolerance_u,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class ComponentConstraint:
    kind: ComponentHintKind
    view: Optional[str] = None
    label: Optional[str] = None
    crop_px: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 1.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _ensure_choice(self.kind, _COMPONENT_HINTS, "kind"))
        if self.view is not None:
            object.__setattr__(self, "view", _ensure_view(self.view))
        if self.label is not None:
            object.__setattr__(self, "label", _ensure_text(self.label, "label"))
        if self.crop_px is not None:
            object.__setattr__(self, "crop_px", _ensure_bbox(self.crop_px, "crop_px"))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "component",
            "kind": self.kind,
            "confidence": self.confidence,
        }
        if self.view is not None:
            data["view"] = self.view
        if self.label is not None:
            data["label"] = self.label
        if self.crop_px is not None:
            data["crop_px"] = list(self.crop_px)
        if self.hard is not None:
            data["hard"] = self.hard
        return data


@dataclass(frozen=True)
class PrimitiveFamilyConstraint:
    family: str
    confidence: float = 1.0
    hard: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _ensure_text(self.family, "family"))
        object.__setattr__(self, "confidence", _ensure_confidence(self.confidence))
        object.__setattr__(self, "hard", _ensure_bool_or_none(self.hard))

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "primitive_family",
            "family": self.family,
            "confidence": self.confidence,
        }
        if self.hard is not None:
            data["hard"] = self.hard
        return data


Constraint = Union[
    ScribbleConstraint,
    AxisConstraint,
    DimensionConstraint,
    SymmetryConstraint,
    CenterlineConstraint,
    BBoxConstraint,
    PolarityConstraint,
    ViewRoleConstraint,
    PlaneConstraint,
    ComponentConstraint,
    PrimitiveFamilyConstraint,
]


def constraint_type(constraint: Constraint) -> str:
    return str(constraint.to_dict()["type"])


def is_hard_constraint(constraint: Constraint) -> bool:
    explicit = getattr(constraint, "hard", None)
    if explicit is not None:
        return bool(explicit)
    if isinstance(constraint, ScribbleConstraint):
        return constraint.kind in {"foreground", "background"}
    if isinstance(constraint, (DimensionConstraint, BBoxConstraint, PlaneConstraint)):
        return True
    if isinstance(constraint, ComponentConstraint):
        return constraint.kind != "ignore_crop"
    return False


def constraint_id(constraint: Constraint) -> str:
    return stable_constraint_id(constraint.to_dict())


@dataclass(frozen=True)
class ConstraintSatisfaction:
    constraint_id: str
    constraint_type: str
    satisfied: bool
    hard: bool
    score: float
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", _ensure_text(self.constraint_id, "constraint_id"))
        object.__setattr__(self, "constraint_type", _ensure_text(self.constraint_type, "constraint_type"))
        if not isinstance(self.satisfied, bool):
            raise ValueError("satisfied must be a bool")
        if not isinstance(self.hard, bool):
            raise ValueError("hard must be a bool")
        object.__setattr__(self, "score", _ensure_confidence(self.score, "score"))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "details", _json_safe(dict(self.details)))

    @property
    def required(self) -> bool:
        return self.hard

    @property
    def passed(self) -> bool:
        return self.satisfied

    @property
    def reason(self) -> str:
        return self.message

    @property
    def penalty(self) -> float:
        return max(0.0, 1.0 - self.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "satisfied": self.satisfied,
            "passed": self.passed,
            "pass": self.passed,
            "hard": self.hard,
            "required": self.required,
            "score": self.score,
            "penalty": self.penalty,
            "message": self.message,
            "reason": self.reason,
            "details": _json_safe(dict(self.details)),
        }


def satisfaction_for_constraint(
    constraint: Constraint,
    *,
    satisfied: bool,
    score: float,
    message: str = "",
    details: Optional[Mapping[str, Any]] = None,
) -> ConstraintSatisfaction:
    return ConstraintSatisfaction(
        constraint_id=constraint_id(constraint),
        constraint_type=constraint_type(constraint),
        satisfied=bool(satisfied),
        hard=is_hard_constraint(constraint),
        score=score,
        message=message,
        details=details or {},
    )


@dataclass(frozen=True)
class ConstraintSet:
    scribbles: Tuple[ScribbleConstraint, ...] = ()
    axes: Tuple[AxisConstraint, ...] = ()
    dimensions: Tuple[DimensionConstraint, ...] = ()
    symmetries: Tuple[SymmetryConstraint, ...] = ()
    centerlines: Tuple[CenterlineConstraint, ...] = ()
    bboxes: Tuple[BBoxConstraint, ...] = ()
    polarities: Tuple[PolarityConstraint, ...] = ()
    view_roles: Tuple[ViewRoleConstraint, ...] = ()
    planes: Tuple[PlaneConstraint, ...] = ()
    components: Tuple[ComponentConstraint, ...] = ()
    primitive_families: Tuple[PrimitiveFamilyConstraint, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scribbles", tuple(self.scribbles))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "symmetries", tuple(self.symmetries))
        object.__setattr__(self, "centerlines", tuple(self.centerlines))
        object.__setattr__(self, "bboxes", tuple(self.bboxes))
        object.__setattr__(self, "polarities", tuple(self.polarities))
        object.__setattr__(self, "view_roles", tuple(self.view_roles))
        object.__setattr__(self, "planes", tuple(self.planes))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "primitive_families", tuple(self.primitive_families))
        object.__setattr__(self, "metadata", _json_safe(dict(self.metadata)))

    @classmethod
    def from_constraints(
        cls, constraints: Iterable[Constraint], metadata: Optional[Mapping[str, Any]] = None
    ) -> "ConstraintSet":
        buckets: Dict[str, list[Constraint]] = {
            "scribbles": [],
            "axes": [],
            "dimensions": [],
            "symmetries": [],
            "centerlines": [],
            "bboxes": [],
            "polarities": [],
            "view_roles": [],
            "planes": [],
            "components": [],
            "primitive_families": [],
        }
        for constraint in constraints:
            if isinstance(constraint, ScribbleConstraint):
                buckets["scribbles"].append(constraint)
            elif isinstance(constraint, AxisConstraint):
                buckets["axes"].append(constraint)
            elif isinstance(constraint, DimensionConstraint):
                buckets["dimensions"].append(constraint)
            elif isinstance(constraint, SymmetryConstraint):
                buckets["symmetries"].append(constraint)
            elif isinstance(constraint, CenterlineConstraint):
                buckets["centerlines"].append(constraint)
            elif isinstance(constraint, BBoxConstraint):
                buckets["bboxes"].append(constraint)
            elif isinstance(constraint, PolarityConstraint):
                buckets["polarities"].append(constraint)
            elif isinstance(constraint, ViewRoleConstraint):
                buckets["view_roles"].append(constraint)
            elif isinstance(constraint, PlaneConstraint):
                buckets["planes"].append(constraint)
            elif isinstance(constraint, ComponentConstraint):
                buckets["components"].append(constraint)
            elif isinstance(constraint, PrimitiveFamilyConstraint):
                buckets["primitive_families"].append(constraint)
            else:
                raise TypeError(f"Unsupported constraint: {constraint!r}")
        return cls(
            scribbles=tuple(buckets["scribbles"]),  # type: ignore[arg-type]
            axes=tuple(buckets["axes"]),  # type: ignore[arg-type]
            dimensions=tuple(buckets["dimensions"]),  # type: ignore[arg-type]
            symmetries=tuple(buckets["symmetries"]),  # type: ignore[arg-type]
            centerlines=tuple(buckets["centerlines"]),  # type: ignore[arg-type]
            bboxes=tuple(buckets["bboxes"]),  # type: ignore[arg-type]
            polarities=tuple(buckets["polarities"]),  # type: ignore[arg-type]
            view_roles=tuple(buckets["view_roles"]),  # type: ignore[arg-type]
            planes=tuple(buckets["planes"]),  # type: ignore[arg-type]
            components=tuple(buckets["components"]),  # type: ignore[arg-type]
            primitive_families=tuple(buckets["primitive_families"]),  # type: ignore[arg-type]
            metadata=metadata or {},
        )

    def all_constraints(self) -> Tuple[Constraint, ...]:
        return (
            *self.scribbles,
            *self.axes,
            *self.dimensions,
            *self.symmetries,
            *self.centerlines,
            *self.bboxes,
            *self.polarities,
            *self.view_roles,
            *self.planes,
            *self.components,
            *self.primitive_families,
        )

    def sorted_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(
            sorted(
                self.all_constraints(),
                key=lambda item: json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")),
            )
        )

    def is_empty(self) -> bool:
        return len(self.all_constraints()) == 0

    def for_view(self, view: str) -> "ConstraintSet":
        view = _ensure_view(view)
        scoped: list[Constraint] = []
        for constraint in self.all_constraints():
            constraint_view = getattr(constraint, "view", None)
            if constraint_view is None or constraint_view == view:
                scoped.append(constraint)
        return ConstraintSet.from_constraints(scoped, metadata=self.metadata)

    def merge(self, *others: "ConstraintSet") -> "ConstraintSet":
        constraints = list(self.all_constraints())
        metadata = dict(self.metadata)
        for other in others:
            constraints.extend(other.all_constraints())
            metadata.update(dict(other.metadata))
        return ConstraintSet.from_constraints(constraints, metadata=metadata)

    def scribbles_for_view(
        self, view: str, kind: Optional[ScribbleKind] = None
    ) -> Tuple[ScribbleConstraint, ...]:
        view = _ensure_view(view)
        return tuple(
            scribble
            for scribble in self.scribbles
            if scribble.view == view and (kind is None or scribble.kind == kind)
        )

    def foreground_scribbles(self, view: str) -> Tuple[ScribbleConstraint, ...]:
        return self.scribbles_for_view(view, "foreground")

    def background_scribbles(self, view: str) -> Tuple[ScribbleConstraint, ...]:
        return self.scribbles_for_view(view, "background")

    def unknown_scribbles(self, view: str) -> Tuple[ScribbleConstraint, ...]:
        return self.scribbles_for_view(view, "unknown")

    def bbox_for_view(self, view: str) -> Optional[BBoxConstraint]:
        view = _ensure_view(view)
        matches = [bbox for bbox in self.bboxes if bbox.view == view]
        if not matches:
            return None
        return max(matches, key=lambda bbox: (bbox.confidence, -bbox.tolerance_px))

    def dimension_lookup(self, name: str) -> Optional[DimensionConstraint]:
        name = _ensure_text(name, "name").lower()
        matches = [dimension for dimension in self.dimensions if dimension.name == name]
        if not matches:
            return None
        return min(matches, key=lambda dimension: dimension.tolerance_u)

    def dimensions_by_name(self) -> Dict[str, DimensionConstraint]:
        result: Dict[str, DimensionConstraint] = {}
        for dimension in self.dimensions:
            current = result.get(dimension.name)
            if current is None or dimension.tolerance_u < current.tolerance_u:
                result[dimension.name] = dimension
        return result


def summarize_satisfaction(
    satisfaction: Iterable[ConstraintSatisfaction],
) -> Dict[str, Any]:
    entries = tuple(satisfaction)
    hard_unsatisfied = [entry for entry in entries if entry.hard and not entry.satisfied]
    soft_unsatisfied = [entry for entry in entries if not entry.hard and not entry.satisfied]
    penalties = [entry.penalty for entry in entries]
    if entries:
        score = sum(entry.score for entry in entries) / float(len(entries))
        penalty = sum(penalties) / float(len(entries))
    else:
        score = 1.0
        penalty = 0.0
    passed = not hard_unsatisfied and not soft_unsatisfied
    first_failed = next((entry for entry in entries if not entry.satisfied), None)
    return {
        "score": score,
        "penalty": penalty,
        "satisfied": passed,
        "passed": passed,
        "pass": passed,
        "required": any(entry.hard for entry in entries),
        "reason": "" if first_failed is None else first_failed.message,
        "hard_failed": bool(hard_unsatisfied),
        "hard_penalty": sum(entry.penalty for entry in entries if entry.hard),
        "soft_penalty": sum(entry.penalty for entry in entries if not entry.hard),
        "total": len(entries),
        "satisfied_count": sum(1 for entry in entries if entry.satisfied),
        "unsatisfied_count": sum(1 for entry in entries if not entry.satisfied),
        "hard_unsatisfied": [entry.to_dict() for entry in hard_unsatisfied],
        "soft_unsatisfied": [entry.to_dict() for entry in soft_unsatisfied],
        "entries": [entry.to_dict() for entry in entries],
    }
