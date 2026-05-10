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


def primitive_payload_summary(primitives: tuple[object, ...]) -> Mapping[str, object]:
    """Return a compact manifest-safe primitive summary."""
    return {
        "primitive_count": len(primitives),
        "types": [type(primitive).__name__ for primitive in primitives],
    }
