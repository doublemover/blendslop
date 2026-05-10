from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from metrics.topology import mesh_topology_report
from placement.resfit_initialization import (
    PrimitiveInitializationConfig,
    initialize_ellipsoids_from_points,
    initialize_gaussians_from_points,
)
from reconstruction.artifacts import write_json
from reconstruction.mesh_io import (
    combine_primitive_meshes,
    mesh_arrays_from_object,
    write_obj,
    write_primitive_set,
)
from reconstruction.point_cloud import target_surface_points
from reconstruction.types import CandidateMetrics, CandidateResult

try:
    from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from primitives.proxy_distillation import distill_proxy_field, write_proxy_field_npz
except ImportError:  # pragma: no cover - package import path
    from ..shape_program import ShapeNode, ShapeProgram, validate_shape_program
    from ..proxy_distillation import distill_proxy_field, write_proxy_field_npz

from .scoring import _ellipsoid_proxy_terms


def _editable_proxy_program_from_primitives(
    primitives: tuple[object, ...],
    *,
    family: str,
    program_id: str,
    sigma: float,
) -> ShapeProgram:
    nodes = [
        _editable_proxy_node(
            primitive,
            index=index,
            family=family,
            sigma=sigma,
        )
        for index, primitive in enumerate(primitives)
    ]
    return ShapeProgram(
        schema_version="shape-program-v1",
        program_id=program_id,
        root_nodes=tuple(nodes),
        constraints=(),
        residual_patches=(),
        metadata={
            "source": "gaussian_ellipsoid_proxy_distillation",
            "family": family,
            "primitive_count": len(primitives),
            "sigma": sigma,
            "editable_output": True,
        },
    )


def _editable_proxy_node(
    primitive: object,
    *,
    index: int,
    family: str,
    sigma: float,
) -> ShapeNode:
    center, radii, rotation, density, confidence, source_type = _ellipsoid_proxy_terms(
        primitive,
        sigma=sigma,
    )
    return ShapeNode(
        node_id=f"editable_proxy_{index:03d}",
        operation="add",
        primitive_type="ellipsoid",
        name=f"{source_type.replace('_', ' ')} editable proxy {index:03d}",
        editable=True,
        parameters={
            "source_family": family,
            "source_primitive_type": source_type,
            "source_primitive_index": index,
            "location_x": float(center[0]),
            "location_y": float(center[1]),
            "location_z": float(center[2]),
            "width_world": float(2.0 * radii[0]),
            "depth_world": float(2.0 * radii[1]),
            "height_world": float(2.0 * radii[2]),
            "radius_x_world": float(radii[0]),
            "radius_y_world": float(radii[1]),
            "radius_z_world": float(radii[2]),
            "density": float(density),
            "opacity": float(density),
            "confidence": float(confidence),
            "rotation_row_major": [float(value) for value in rotation.reshape(-1)],
            "distillation": "gaussian_to_editable_ellipsoid",
        },
    )


def _editable_proxy_summary(
    program: ShapeProgram,
    *,
    validation_errors: Sequence[str],
    source_family: str,
) -> dict[str, Any]:
    valid = not validation_errors
    node_count = program.node_count()
    score = 0.0 if not valid else min(0.92, 0.72 + 0.20 * node_count / float(node_count + 4))
    component_sanity_score = 0.0 if not valid else min(0.88, 0.68 + 0.20 / max(1.0, node_count / 12.0))
    return {
        "source": "gaussian_ellipsoid_proxy_distillation",
        "source_family": source_family,
        "shape_program": program.to_dict(),
        "node_count": node_count,
        "validation_errors": list(validation_errors),
        "editability_score": float(score),
        "component_sanity_score": float(component_sanity_score),
        "topology_interpretation": (
            "editable primitive-set topology score; dense proxy mesh may remain "
            "multi-component and non-watertight"
        ),
        "editable_primitives": [
            node.primitive_type for node in program.root_nodes if node.editable
        ],
    }
