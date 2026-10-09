"""Candidate-blind authored reference tessellation observations.

Measured discretization error does not declare acceptable reconstruction error.
The existing vase tolerance is deliberately absent from this contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from blender_blocking.synthetic.quality_contracts import quality_workload

REFERENCE_NOISE_FAMILIES = ("rounded_triangle_dot", "sphere")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def reference_noise_workload():
    """Fix authored parameters, tessellations and sample budget before measuring."""
    authored = {row["name"]: row for row in quality_workload()["cases"]}
    families = {}
    for name in REFERENCE_NOISE_FAMILIES:
        levels = ({"baseline": {"corner_segments": 32, "dome_segments": 64},
                   "dense": {"corner_segments": 64, "dome_segments": 128},
                   "finer": {"corner_segments": 128, "dome_segments": 256}}
                  if name == "rounded_triangle_dot" else
                  {"baseline": {"segments": 64, "ring_count": 32},
                   "dense": {"segments": 128, "ring_count": 64},
                   "finer": {"segments": 256, "ring_count": 128}})
        families[name] = {"authored_parameters": deepcopy(authored[name]["parameters"]),
                          "tessellations": levels, "authored_case_sha256": canonical_digest(authored[name])}
    return {"protocol": "candidate_blind_reference_noise_v1", "status": "frozen_unmeasured",
            "families": families, "comparisons": [["baseline", "dense"], ["dense", "finer"]],
            "metric": {"protocol": "shared_world_area_sample_to_triangle_v1",
                       "sample_count_per_direction": 4096, "seed": 61007,
                       "normal_orientation": "oriented geometric face normals", "frame": "unchanged authored world coordinates"},
            "limits": {"wall_seconds": 120, "max_rss_bytes": 8589934592, "native_threads": 2,
                       "max_generated_bytes": 67108864},
            "candidate_access": "forbidden; only authored-reference receipt/workload/geometry input",
            "renders": "unrun", "boundary_qualification": "unrun",
            "family_acceptance_contract": None,
            "family_acceptance_status": "unqualified: no independently justified reconstruction tolerance",
            "error_floor_scope": "observed reference discretization against finer tessellations; not a rigorous analytic upper bound"}


def validate_reference_workload(retained_workload, frozen):
    """Reject a reference whose declared authored geometry differs from the freeze."""
    if retained_workload.get("protocol") != "quality-coverage-references-v1":
        raise ValueError("input must be the authored coverage-reference workload")
    retained = {row["name"]: row for row in retained_workload["cases"]}
    for name, declaration in frozen["families"].items():
        if retained.get(name, {}).get("parameters") != declaration["authored_parameters"]:
            raise ValueError("authored reference parameters differ: " + name)


def build_reference_tessellation(family, level, frozen):
    """Build one newly owned native reference and release its transient object."""
    import bpy
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays
    declaration = frozen["families"][family]
    parameters = deepcopy(declaration["authored_parameters"])
    options = declaration["tessellations"][level]
    if family == "rounded_triangle_dot":
        from blender_blocking.synthetic.quality_contracts import rounded_triangle_mesh
        parameters.update(options)
        vertices, faces = rounded_triangle_mesh(parameters)
        mesh = bpy.data.meshes.new("ReferenceNoise_" + family + "_" + level)
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
    elif family == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(parameters["radius"]), **options)
        obj = bpy.context.active_object
        mesh = obj.data
    else:
        raise ValueError("unsupported independently declared reference-noise family")
    try:
        bpy.context.view_layer.update()
        return evaluated_arrays(obj)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def oriented_surface_identity(data):
    """Exact coordinate/triangle identity independent of native index ordering.

    Preserve every vertex and oriented triangle; neither floating-point rounding,
    changed diagonals nor opposite normals can become equivalent here.
    """
    import numpy as np
    vertices = np.asarray(data.vertices, dtype="<f8")
    triangles = vertices[data.faces]
    starts = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)[:, 0]
    cycles = (starts[:, None] + np.arange(3)) % 3
    canonical = triangles[np.arange(len(triangles))[:, None], cycles].reshape(-1, 9)
    triangle_order = np.lexsort(tuple(canonical[:, axis] for axis in range(8, -1, -1)))
    vertex_order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    digest = hashlib.sha256(b"oriented_triangle_coordinates_and_vertex_multiset_v1")
    digest.update(str((len(vertices), len(triangles))).encode("ascii"))
    digest.update(vertices[vertex_order].tobytes())
    digest.update(canonical[triangle_order].tobytes())
    return digest.hexdigest()


def reference_noise_observation(family, level_arrays, frozen):
    """Compare only declared reference levels; no candidate or pass cutoff input."""
    from blender_blocking.evaluation.canonical_artifacts import raw_surface_observation
    declaration = frozen["families"][family]
    if set(level_arrays) != set(declaration["tessellations"]):
        raise ValueError("all independently frozen reference levels are required")
    metric = frozen["metric"]
    pairs = {}
    for source, target in frozen["comparisons"]:
        key = source + "_to_" + target
        pairs[key] = raw_surface_observation(level_arrays[source], level_arrays[target],
            count=metric["sample_count_per_direction"], seed=metric["seed"])
    return {"family": family, "authored_parameters": declaration["authored_parameters"],
            "tessellations": declaration["tessellations"], "observations": pairs,
            "reference_noise_status": "measured", "noise_floor_measured": True,
            "noise_qualified": False, "family_acceptance_contract": None,
            "family_acceptance_status": frozen["family_acceptance_status"],
            "error_floor_scope": frozen["error_floor_scope"], "candidate_accessed": False,
            "reference_geometry_hashes": {name: data.content_hash for name, data in level_arrays.items()}}
