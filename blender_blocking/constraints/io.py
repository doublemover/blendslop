"""JSON IO for human correction constraints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .model import (
    AxisConstraint,
    BBoxConstraint,
    CenterlineConstraint,
    ComponentConstraint,
    Constraint,
    ConstraintSet,
    DimensionConstraint,
    PlaneConstraint,
    PolarityConstraint,
    PrimitiveFamilyConstraint,
    ScribbleConstraint,
    SUPPORTED_VERSION,
    SymmetryConstraint,
    ViewRoleConstraint,
    stable_constraint_id,
)


class ConstraintIOError(ValueError):
    """Raised when a constraint JSON payload is malformed."""


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintIOError(f"{name} must be an object")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], type_name: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise ConstraintIOError(
            f"{type_name} constraint has unknown fields: {sorted(unknown)}"
        )


def _make_scribble(data: Mapping[str, Any]) -> ScribbleConstraint:
    _reject_unknown(
        data,
        {"type", "view", "kind", "points_px", "brush_radius_px", "confidence", "hard"},
        "scribble",
    )
    return ScribbleConstraint(
        view=data["view"],
        kind=data["kind"],
        points_px=data["points_px"],
        brush_radius_px=data["brush_radius_px"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_axis(data: Mapping[str, Any]) -> AxisConstraint:
    _reject_unknown(
        data, {"type", "axis_name", "direction_world", "confidence", "hard"}, "axis"
    )
    return AxisConstraint(
        axis_name=data["axis_name"],
        direction_world=data["direction_world"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_dimension(data: Mapping[str, Any]) -> DimensionConstraint:
    _reject_unknown(data, {"type", "name", "value_u", "tolerance_u", "hard"}, "dimension")
    return DimensionConstraint(
        name=data["name"],
        value_u=data["value_u"],
        tolerance_u=data.get("tolerance_u", 0.0),
        hard=data.get("hard"),
    )


def _make_symmetry(data: Mapping[str, Any]) -> SymmetryConstraint:
    _reject_unknown(data, {"type", "plane", "confidence", "hard"}, "symmetry")
    return SymmetryConstraint(
        plane=data["plane"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_centerline(data: Mapping[str, Any]) -> CenterlineConstraint:
    _reject_unknown(data, {"type", "view", "points_px", "tolerance_px", "hard"}, "centerline")
    return CenterlineConstraint(
        view=data["view"],
        points_px=data["points_px"],
        tolerance_px=data.get("tolerance_px", 0.0),
        hard=data.get("hard"),
    )


def _make_bbox(data: Mapping[str, Any]) -> BBoxConstraint:
    _reject_unknown(
        data, {"type", "view", "bbox_px", "confidence", "tolerance_px", "hard"}, "bbox"
    )
    return BBoxConstraint(
        view=data["view"],
        bbox_px=data["bbox_px"],
        confidence=data.get("confidence", 1.0),
        tolerance_px=data.get("tolerance_px", 0.0),
        hard=data.get("hard"),
    )


def _make_polarity(data: Mapping[str, Any]) -> PolarityConstraint:
    _reject_unknown(data, {"type", "view", "polarity", "confidence", "hard"}, "polarity")
    return PolarityConstraint(
        view=data["view"],
        polarity=data["polarity"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_view_role(data: Mapping[str, Any]) -> ViewRoleConstraint:
    _reject_unknown(data, {"type", "view", "role", "confidence", "hard"}, "view_role")
    return ViewRoleConstraint(
        view=data["view"],
        role=data["role"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_plane(data: Mapping[str, Any]) -> PlaneConstraint:
    _reject_unknown(
        data, {"type", "kind", "axis", "value_u", "tolerance_u", "hard"}, "plane"
    )
    return PlaneConstraint(
        kind=data["kind"],
        axis=data.get("axis", "z"),
        value_u=data.get("value_u", 0.0),
        tolerance_u=data.get("tolerance_u", 0.0),
        hard=data.get("hard"),
    )


def _make_component(data: Mapping[str, Any]) -> ComponentConstraint:
    _reject_unknown(
        data,
        {"type", "kind", "view", "label", "crop_px", "confidence", "hard"},
        "component",
    )
    return ComponentConstraint(
        kind=data["kind"],
        view=data.get("view"),
        label=data.get("label"),
        crop_px=data.get("crop_px"),
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


def _make_primitive_family(data: Mapping[str, Any]) -> PrimitiveFamilyConstraint:
    _reject_unknown(data, {"type", "family", "confidence", "hard"}, "primitive_family")
    return PrimitiveFamilyConstraint(
        family=data["family"],
        confidence=data.get("confidence", 1.0),
        hard=data.get("hard"),
    )


_CONSTRAINT_FACTORIES: Dict[str, Callable[[Mapping[str, Any]], Constraint]] = {
    "scribble": _make_scribble,
    "axis": _make_axis,
    "dimension": _make_dimension,
    "symmetry": _make_symmetry,
    "centerline": _make_centerline,
    "bbox": _make_bbox,
    "polarity": _make_polarity,
    "view_role": _make_view_role,
    "plane": _make_plane,
    "component": _make_component,
    "primitive_family": _make_primitive_family,
}


def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    data = _expect_mapping(data, "constraint")
    type_name = data.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ConstraintIOError("constraint type must be a non-empty string")
    factory = _CONSTRAINT_FACTORIES.get(type_name)
    if factory is None:
        raise ConstraintIOError(f"Unsupported constraint type: {type_name}")
    try:
        return factory(data)
    except KeyError as exc:
        raise ConstraintIOError(
            f"{type_name} constraint missing required field: {exc.args[0]}"
        ) from exc
    except ValueError as exc:
        raise ConstraintIOError(f"{type_name} constraint is invalid: {exc}") from exc


def constraint_to_dict(constraint: Constraint) -> Dict[str, Any]:
    return constraint.to_dict()


def constraint_set_to_payload(
    constraint_set: ConstraintSet, *, deterministic: bool = True
) -> Dict[str, Any]:
    constraints = (
        constraint_set.sorted_constraints()
        if deterministic
        else constraint_set.all_constraints()
    )
    payload: Dict[str, Any] = {
        "version": SUPPORTED_VERSION,
        "constraints": [constraint_to_dict(constraint) for constraint in constraints],
    }
    if constraint_set.metadata:
        payload["metadata"] = dict(constraint_set.metadata)
    return payload


def constraint_set_from_payload(payload: Mapping[str, Any]) -> ConstraintSet:
    payload = _expect_mapping(payload, "payload")
    _reject_unknown(payload, {"version", "constraints", "metadata"}, "constraint file")
    version = payload.get("version")
    if version != SUPPORTED_VERSION:
        raise ConstraintIOError(
            f"Unsupported constraint file version {version!r}; expected {SUPPORTED_VERSION}"
        )
    raw_constraints = payload.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ConstraintIOError("constraints must be a list")
    constraints = [constraint_from_dict(item) for item in raw_constraints]
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ConstraintIOError("metadata must be an object when provided")
    return ConstraintSet.from_constraints(constraints, metadata=metadata)


def dumps_constraint_set(
    constraint_set: ConstraintSet,
    *,
    deterministic: bool = True,
    indent: int = 2,
) -> str:
    payload = constraint_set_to_payload(constraint_set, deterministic=deterministic)
    return json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False) + "\n"


def loads_constraint_set(text: str) -> ConstraintSet:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConstraintIOError(f"Invalid constraint JSON: {exc}") from exc
    return constraint_set_from_payload(payload)


def load_constraint_file(path: str | Path) -> ConstraintSet:
    return loads_constraint_set(Path(path).read_text(encoding="utf-8"))


def save_constraint_file(
    path: str | Path,
    constraint_set: ConstraintSet,
    *,
    deterministic: bool = True,
) -> None:
    Path(path).write_text(
        dumps_constraint_set(constraint_set, deterministic=deterministic),
        encoding="utf-8",
    )


def load_constraint_files(paths: Iterable[str | Path]) -> ConstraintSet:
    result = ConstraintSet()
    for path in paths:
        result = result.merge(load_constraint_file(path))
    return result


def constraint_set_hash(
    constraint_set: ConstraintSet, *, deterministic: bool = True
) -> str:
    payload = constraint_set_to_payload(constraint_set, deterministic=deterministic)
    return stable_constraint_id(payload)


def make_constraint_payload(
    constraints: Iterable[Constraint],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return constraint_set_to_payload(
        ConstraintSet.from_constraints(constraints, metadata=metadata),
        deterministic=True,
    )
