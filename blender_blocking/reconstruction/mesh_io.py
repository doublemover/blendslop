"""Backend-neutral mesh and primitive artifact IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from metrics.topology import mesh_topology_report

from .artifacts import stable_json_dumps
from .types import MeshData as ReconstructionMeshData


def mesh_arrays_from_object(mesh: Any) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Accept protocol MeshData, reconstruction MeshData, or raw mappings."""
    if isinstance(mesh, ReconstructionMeshData):
        return (
            np.asarray(mesh.vertices, dtype=float),
            tuple(tuple(int(i) for i in face) for face in mesh.faces),
        )
    if isinstance(mesh, Mapping):
        return (
            np.asarray(mesh.get("vertices", ()), dtype=float),
            tuple(tuple(int(i) for i in face) for face in mesh.get("faces", ())),
        )
    vertices = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if vertices is None or faces is None:
        raise TypeError(f"unsupported mesh object: {type(mesh)!r}")
    return (
        np.asarray(vertices, dtype=float),
        tuple(tuple(int(i) for i in face) for face in faces),
    )


def write_obj(
    path: str | Path,
    mesh: Any,
    *,
    header: Optional[Sequence[str]] = None,
) -> Path:
    """Write a plain OBJ mesh artifact from backend-neutral arrays."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = mesh_arrays_from_object(mesh)
    lines = []
    for item in header or ():
        lines.append(f"# {item}")
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}")
    for face in faces:
        # OBJ is 1-indexed.
        lines.append("f " + " ".join(str(index + 1) for index in face))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_mesh_json(path: str | Path, mesh: Any, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write mesh arrays and topology metadata as deterministic JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = mesh_arrays_from_object(mesh)
    report = mesh_topology_report(vertices, faces)
    payload = {
        "vertices": vertices.tolist(),
        "faces": [list(face) for face in faces],
        "topology": report.to_dict(),
        "metadata": dict(metadata or {}),
    }
    output.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    return output


def write_primitive_set(
    path: str | Path,
    primitives: Iterable[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a JSON primitive set artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    primitive_payload = []
    for primitive in primitives:
        if hasattr(primitive, "to_dict"):
            primitive_payload.append(primitive.to_dict())
        else:
            primitive_payload.append({"type": type(primitive).__name__, "repr": repr(primitive)})
    payload = {
        "primitive_count": len(primitive_payload),
        "primitives": primitive_payload,
        "metadata": dict(metadata or {}),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return output


def combine_primitive_meshes(
    primitives: Sequence[Any],
    *,
    resolution: int = 24,
) -> ReconstructionMeshData:
    """Concatenate primitive mesh proxies into one neutral mesh."""
    vertices_blocks = []
    faces_out: list[tuple[int, ...]] = []
    offset = 0
    for primitive in primitives:
        if not hasattr(primitive, "to_mesh_data"):
            continue
        vertices, faces = mesh_arrays_from_object(primitive.to_mesh_data(resolution))
        vertices_blocks.append(vertices)
        faces_out.extend(tuple(index + offset for index in face) for face in faces)
        offset += len(vertices)
    if vertices_blocks:
        vertices_all = np.vstack(vertices_blocks)
    else:
        vertices_all = np.empty((0, 3), dtype=float)
    return ReconstructionMeshData(
        vertices=tuple(tuple(float(v) for v in row) for row in vertices_all),
        faces=tuple(faces_out),
        metadata={"source": "primitive_proxy", "resolution": int(resolution)},
    )
