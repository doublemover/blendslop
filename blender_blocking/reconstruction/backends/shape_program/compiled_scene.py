from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...artifacts import write_json
from ...backend import BackendBudget, BackendCapabilities, BaseBackend
from ...types import CandidateMetrics, CandidateRequest, CandidateResult, ProfileBand, ReconstructionTarget

try:  # Package import when called as blender_blocking.*
    from blender_blocking.primitives.shape_program import (
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from blender_blocking.primitives.grammar import default_shape_program_grammar
    from blender_blocking.primitives.program_search import search_shape_program_candidates
    from blender_blocking.primitives.shape_dsl import program_to_dsl
except Exception:  # pragma: no cover - legacy script import path
    from primitives.shape_program import (  # type: ignore
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from primitives.grammar import default_shape_program_grammar  # type: ignore
    from primitives.program_search import search_shape_program_candidates  # type: ignore
    from primitives.shape_dsl import program_to_dsl  # type: ignore

_ROOT_STRATEGIES = {"profile_lathe", "bounds_box", "hybrid_profile_bounds"}
_RESIDUAL_POLICIES = {"ignore", "report", "suggest_patches"}


def _compiled_scene_summary(compiled: Any) -> dict[str, Any]:
    if compiled is None:
        return {
            "compiled": False,
            "object_count": 0,
            "mesh_object_count": 0,
            "vertex_count": 0,
            "face_count": 0,
            "modifier_count": 0,
            "material_count": 0,
        }
    objects = tuple(getattr(compiled, "objects", ()) or ())
    mesh_objects = [
        obj
        for obj in objects
        if getattr(getattr(obj, "data", None), "vertices", None) is not None
    ]
    vertex_count = 0
    face_count = 0
    modifier_count = 0
    material_count = 0
    for obj in objects:
        data = getattr(obj, "data", None)
        vertices = getattr(data, "vertices", ()) or ()
        polygons = getattr(data, "polygons", ()) or ()
        materials = getattr(data, "materials", ()) or ()
        modifiers = getattr(obj, "modifiers", ()) or ()
        vertex_count += len(vertices)
        face_count += len(polygons)
        material_count += len(materials)
        modifier_count += len(modifiers)
    return {
        "compiled": True,
        "object_count": len(objects),
        "mesh_object_count": len(mesh_objects),
        "residual_marker_count": len(getattr(compiled, "residual_markers", ()) or ()),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "modifier_count": modifier_count,
        "material_count": material_count,
        "object_names": list(compiled.object_names()) if hasattr(compiled, "object_names") else [],
        "marker_names": list(compiled.marker_names()) if hasattr(compiled, "marker_names") else [],
    }

def _compiled_topology_summary(compiled: Any) -> dict[str, Any]:
    if compiled is None:
        return {
            "status": "not_applicable",
            "reason": "shape program has not been compiled",
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for obj in tuple(getattr(compiled, "objects", ()) or ()):
        data = getattr(obj, "data", None)
        obj_vertices = getattr(data, "vertices", None)
        obj_polygons = getattr(data, "polygons", None)
        if obj_vertices is None or obj_polygons is None:
            continue
        offset = len(vertices)
        matrix = getattr(obj, "matrix_world", None)
        for vertex in obj_vertices:
            co = getattr(vertex, "co", None)
            if co is None:
                continue
            if matrix is not None:
                try:
                    co = matrix @ co
                except Exception:
                    pass
            vertices.append((float(co[0]), float(co[1]), float(co[2])))
        for polygon in obj_polygons:
            indices = tuple(int(index) + offset for index in getattr(polygon, "vertices", ()) or ())
            if len(indices) >= 3:
                faces.append(indices)
    if not vertices or not faces:
        return {
            "status": "skipped",
            "reason": "compiled shape program produced no mesh faces",
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }
    try:
        from metrics.topology import mesh_topology_report

        report = mesh_topology_report(
            np.asarray(vertices, dtype=float),
            tuple(faces),
        ).to_dict()
        report["status"] = "measured"
        report["source"] = "shape_program_compiler"
        return report
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"compiled shape program topology report failed: {exc}",
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "topology_score": 0.0,
            "penalty": 1.0,
            "source": "shape_program_compiler",
        }
