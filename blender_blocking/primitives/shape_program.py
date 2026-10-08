"""Editable shape-program data model.

Shape programs are a Blender-friendly alternative to dense mesh-only output:
they describe named primitives, operations, constraints, and residual patches
that can compile to editable objects and modifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ShapeConstraint:
    kind: str
    target_nodes: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "target_nodes": list(self.target_nodes),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ShapeNode:
    node_id: str
    operation: str
    primitive_type: str | None
    parameters: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    children: tuple[str, ...] = ()
    name: str = ""
    editable: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "primitive_type": self.primitive_type,
            "parameters": dict(self.parameters),
            "children": list(self.children),
            "name": self.name or self.node_id,
            "editable": self.editable,
        }


@dataclass(frozen=True)
class ResidualPatch:
    patch_id: str
    category: str
    source_view: str
    confidence: float = 0.0
    suggested_node: ShapeNode | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "category": self.category,
            "source_view": self.source_view,
            "confidence": self.confidence,
            "suggested_node": None if self.suggested_node is None else self.suggested_node.to_dict(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ShapeProgram:
    schema_version: str
    program_id: str
    root_nodes: tuple[ShapeNode, ...]
    constraints: tuple[ShapeConstraint, ...] = ()
    residual_patches: tuple[ResidualPatch, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def node_count(self) -> int:
        return len(self.root_nodes)

    def residual_patch_count(self) -> int:
        return len(self.residual_patches)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "root_nodes": [node.to_dict() for node in self.root_nodes],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "residual_patches": [patch.to_dict() for patch in self.residual_patches],
            "metadata": dict(self.metadata),
        }


PRIMITIVE_TYPES = {
    "rounded_triangle",
    "convex_hull",
    "polygon_extrusion",
    "generalized_sweep",
    "deformed_superquadric",
    "box",
    "rounded_box",
    "cylinder",
    "cone",
    "frustum",
    "capsule",
    "sphere",
    "ellipsoid",
    "superquadric",
    "torus",
    "lathe_profile",
    "loft_profile",
    "plane_patch",
    "residual_mesh_patch",
}


OPERATIONS = {
    "subtract",
    "intersect",
    "add",
    "union",
    "difference",
    "intersection",
    "mirror",
    "radial_array",
    "linear_array",
    "attach",
    "align",
    "bevel",
    "solidify",
    "smooth",
    "subdivision",
}


def validate_shape_program(program: ShapeProgram) -> tuple[str, ...]:
    errors: list[str] = []
    node_ids = set()
    for node in program.root_nodes:
        if node.node_id in node_ids:
            errors.append(f"duplicate node_id {node.node_id!r}")
        node_ids.add(node.node_id)
        if node.operation not in OPERATIONS:
            errors.append(f"unknown operation {node.operation!r} on {node.node_id}")
        if node.primitive_type is not None and node.primitive_type not in PRIMITIVE_TYPES:
            errors.append(f"unknown primitive_type {node.primitive_type!r} on {node.node_id}")
    for constraint in program.constraints:
        for target in constraint.target_nodes:
            if target not in node_ids:
                errors.append(f"constraint {constraint.kind!r} targets missing node {target!r}")
    return tuple(errors)



COMPILED_PRIMITIVE_TYPES = PRIMITIVE_TYPES - {"capsule"}
COMPILED_OPERATIONS = {"add", "union", "subtract", "difference", "intersect", "intersection"}


def validate_compilable_program(program: ShapeProgram) -> tuple[str, ...]:
    """Validate actual compiler capabilities separately from the authoring schema."""
    errors = list(validate_shape_program(program))
    for node in program.root_nodes:
        if node.operation not in COMPILED_OPERATIONS:
            errors.append(f"unsupported compiler operation {node.operation!r} on {node.node_id}")
        if node.primitive_type not in COMPILED_PRIMITIVE_TYPES:
            errors.append(f"unsupported compiler primitive {node.primitive_type!r} on {node.node_id}")
        if node.children:
            errors.append(f"unsupported compiler child references on {node.node_id}: {node.children!r}")
        if node.primitive_type == "rounded_triangle":
            try:
                from .rounded_triangle import RoundedTrianglePrimitive
                RoundedTrianglePrimitive.from_program_parameters(node.parameters)
            except Exception as exc:
                errors.append(f"invalid rounded triangle on {node.node_id}: {exc}")
        if node.primitive_type == 'deformed_superquadric':
            try:
                from .deformed_superquadric import DeformedSuperquadricPrimitive
                DeformedSuperquadricPrimitive.from_program_parameters(node.parameters)
            except Exception as exc:
                errors.append(f'invalid deformed superquadric on {node.node_id}: {exc}')
        if node.primitive_type == 'generalized_sweep':
            try:
                from .generalized_sweep import GeneralizedSweepPrimitive
                GeneralizedSweepPrimitive.from_program_parameters(node.parameters)
            except Exception as exc:
                errors.append(f'invalid generalized sweep on {node.node_id}: {exc}')
    return tuple(errors)
