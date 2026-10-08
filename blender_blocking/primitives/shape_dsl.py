"""Text DSL helpers for editable shape programs."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .shape_program import ResidualPatch, ShapeConstraint, ShapeNode, ShapeProgram


def program_to_dsl(program: ShapeProgram) -> str:
    """Serialize a shape program into a stable, line-oriented DSL."""

    lines = [
        f"program {program.program_id} schema={program.schema_version}",
    ]
    for node in program.root_nodes:
        lines.append(
            "node {node_id} op={operation} primitive={primitive} editable={editable} "
            "name={name} params={params} children={children}".format(
                node_id=node.node_id,
                operation=node.operation,
                primitive=node.primitive_type or "none",
                editable=json.dumps(bool(node.editable)),
                name=json.dumps(node.name or node.node_id),
                params=_json(node.parameters),
                children=_json(list(node.children)),
            )
        )
    for constraint in program.constraints:
        lines.append(
            "constraint {kind} targets={targets} params={params}".format(
                kind=constraint.kind,
                targets=_json(list(constraint.target_nodes)),
                params=_json(constraint.parameters),
            )
        )
    for patch in program.residual_patches:
        suggested = (
            patch.suggested_node.to_dict() if patch.suggested_node is not None else None
        )
        lines.append(
            "residual {patch_id} category={category} view={view} confidence={confidence} "
            "suggested={suggested} notes={notes}".format(
                patch_id=patch.patch_id,
                category=patch.category,
                view=patch.source_view,
                confidence=float(patch.confidence),
                suggested=_json(suggested),
                notes=_json(list(patch.notes)),
            )
        )
    if program.metadata:
        lines.append(f"metadata {_json(program.metadata)}")
    return "\n".join(lines) + "\n"


def program_from_dsl(text: str) -> ShapeProgram:
    """Parse DSL emitted by :func:`program_to_dsl`."""

    program_id = ""
    schema = "shape-program-v1"
    nodes: list[ShapeNode] = []
    constraints: list[ShapeConstraint] = []
    residuals: list[ResidualPatch] = []
    metadata: Mapping[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("program "):
            tokens = line.split(maxsplit=2)
            program_id = tokens[1]
            if len(tokens) > 2 and tokens[2].startswith("schema="):
                schema = tokens[2].split("=", 1)[1]
        elif line.startswith("node "):
            nodes.append(_parse_node(line))
        elif line.startswith("constraint "):
            constraints.append(_parse_constraint(line))
        elif line.startswith("residual "):
            residuals.append(_parse_residual(line))
        elif line.startswith("metadata "):
            metadata = json.loads(line.split(" ", 1)[1])
    if not program_id:
        raise ValueError("shape DSL is missing program header")
    return ShapeProgram(
        schema_version=schema,
        program_id=program_id,
        root_nodes=tuple(nodes),
        constraints=tuple(constraints),
        residual_patches=tuple(residuals),
        metadata=metadata,
    )


def _parse_node(line: str) -> ShapeNode:
    node_id, rest = line.split(maxsplit=2)[1:]
    fields = _parse_fields(rest)
    primitive = fields["primitive"]
    return ShapeNode(
        node_id=node_id,
        operation=str(fields["op"]),
        primitive_type=None if primitive == "none" else str(primitive),
        editable=bool(fields["editable"]),
        name=str(fields["name"]),
        parameters=dict(fields["params"]),
        children=tuple(str(item) for item in fields["children"]),
    )


def _parse_constraint(line: str) -> ShapeConstraint:
    kind, rest = line.split(maxsplit=2)[1:]
    fields = _parse_fields(rest)
    return ShapeConstraint(
        kind=kind,
        target_nodes=tuple(str(item) for item in fields["targets"]),
        parameters=dict(fields["params"]),
    )


def _parse_residual(line: str) -> ResidualPatch:
    patch_id, rest = line.split(maxsplit=2)[1:]
    fields = _parse_fields(rest)
    suggested_payload = fields.get("suggested")
    suggested = (
        ShapeNode(
            node_id=str(suggested_payload["node_id"]),
            operation=str(suggested_payload["operation"]),
            primitive_type=suggested_payload.get("primitive_type"),
            parameters=dict(suggested_payload.get("parameters", {})),
            children=tuple(suggested_payload.get("children", ())),
            name=str(suggested_payload.get("name", "")),
            editable=bool(suggested_payload.get("editable", True)),
        )
        if isinstance(suggested_payload, Mapping)
        else None
    )
    return ResidualPatch(
        patch_id=patch_id,
        category=str(fields["category"]),
        source_view=str(fields["view"]),
        confidence=float(fields["confidence"]),
        suggested_node=suggested,
        notes=tuple(str(item) for item in fields["notes"]),
    )


def _parse_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    index = 0
    while index < len(text):
        equals = text.find("=", index)
        if equals < 0:
            break
        key_start = text.rfind(" ", index, equals)
        key_start = index if key_start < index else key_start + 1
        key = text[key_start:equals]
        value_start = equals + 1
        next_key = _find_next_key(text, value_start)
        raw_value = text[value_start:next_key].strip()
        fields[key] = _parse_value(raw_value)
        index = next_key + 1
    return fields


def _find_next_key(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string and char in "[{":
            depth += 1
        elif not in_string and char in "]}":
            depth = max(0, depth - 1)
        elif not in_string and depth == 0 and char == " ":
            probe = text.find("=", index + 1)
            if probe > index:
                key = text[index + 1 : probe]
                if key.replace("_", "").isalnum():
                    return index
    return len(text)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
