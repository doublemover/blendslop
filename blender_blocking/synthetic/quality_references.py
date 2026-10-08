"""Executable native references for the already frozen twelve-family workload.

References are authored assets. Their construction/artist-edit checks never
stand in for reconstruction acceptance or solid-boundary qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import numpy as np

from .quality_contracts import quality_workload, rounded_triangle_mesh


@dataclass
class QualityReference:
    object: object
    sources: tuple
    parameters: dict


def build_quality_reference(case, *, parameters=None):
    import bpy
    from .blender_builders import _build_analytic, _part_object
    from .specs import SyntheticShapeSpec, ShapeFamily
    from geometry.profile_models import EllipticalSlice
    from integration.blender_ops.profile_loft_mesh import create_loft_mesh_from_slices
    names = {row["name"] for row in quality_workload()["cases"]}
    if case["name"] not in names:
        raise ValueError("unknown frozen quality family")
    parameters = deepcopy(case["parameters"] if parameters is None else parameters)
    builder = parameters.get("builder")
    sources = ()
    if builder == "analytic_vase":
        h, mean, amp = (float(parameters[k]) for k in ("height", "radius_mean", "radius_amplitude"))
        slices = [EllipticalSlice(z=float(z), rx=mean + amp * math.cos(2 * math.pi * z / h),
                                 ry=mean + amp * math.cos(2 * math.pi * z / h))
                  for z in np.linspace(0, h, 257)]
        obj = create_loft_mesh_from_slices(slices, radial_segments=192, surface_mode="sharp")
        for polygon in obj.data.polygons:
            zs = [obj.data.vertices[i].co.z for i in polygon.vertices]
            polygon.use_smooth = max(zs) - min(zs) > 1e-9
    elif builder == "rounded_triangle":
        vertices, faces = rounded_triangle_mesh(parameters)
        mesh = bpy.data.meshes.new(case["name"] + "Mesh")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(case["name"], mesh)
        bpy.context.collection.objects.link(obj)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    elif builder == "compound":
        parts = parameters["parts"]
        if not parts or parts[0].get("boolean") == "subtract":
            raise ValueError("compound reference requires a positive base")
        sources = tuple(_part_object(bpy, part) for part in parts)
        obj = sources[0]
        for part, operand in zip(parts[1:], sources[1:]):
            modifier = obj.modifiers.new("Authored_" + operand.name, "BOOLEAN")
            modifier.solver = "EXACT"
            modifier.operation = "DIFFERENCE" if part.get("boolean") == "subtract" else "UNION"
            modifier.object = operand
            world = operand.matrix_world.copy()
            operand.parent = obj
            operand.matrix_world = world
            operand.hide_render = True
            operand["blendslop_export_exclude"] = True
    elif builder is None:
        spec = SyntheticShapeSpec(case["name"], ShapeFamily.ANALYTIC_PRIMITIVE.value,
                                  quality_workload()["seed"], parameters)
        obj = _build_analytic(bpy, spec)
    else:
        raise ValueError("unsupported frozen quality builder")
    obj.name = case["name"]
    obj["quality_reference_recipe_json"] = json.dumps(parameters, sort_keys=True)
    bpy.context.view_layer.update()
    return QualityReference(obj, sources or (obj,), parameters)


def coverage_fixture_contract():
    """Add explicit local feature probes; preserve the original workload file."""
    workload = quality_workload()
    workload["protocol"] = "quality-coverage-references-v1"
    workload["execution_scope"] = "authored references, live object edits and three negative controls; no reconstruction pass implied"
    workload["feature_contracts"] = {
        "thin_plate": {"depth_axis": 1, "expected_depth_world": .08, "max_error_world": .005},
        "concave_arch": {"empty_points_world": [[0, 0, -.5], [0, 0, 0]],
                         "occupied_points_world": [[-.7, 0, 0], [.7, 0, 0], [0, 0, .7]]},
        "asymmetric_multipart_solid": {"occupied_points_world": [[0, 0, -.25], [-.4, .1, .8], [.45, -.08, .35]]},
    }
    workload["negative_controls"] = {
        "incorrect_depth": {"family": "thin_plate", "depth_scale": .1},
        "filled_cavity": {"family": "concave_arch", "remove_subtract_modifier": True},
        "missing_multipart": {"family": "asymmetric_multipart_solid", "remove_union_modifiers": True},
    }
    workload["silhouette_observation"] = {"min_area_iou": .7, "boundary_radius_px": 2,
        "boundary_and_signed_distance": "measured; reconstruction thresholds must come from its declared config"}
    workload["edit_contract"] = "change source X scale by 1.05 and restore; compound upper part also edited independently"
    return workload


def quality_feature_verdict(family, data):
    """Authored depth/occupancy probes; absence of a probe is explicit."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    contract = coverage_fixture_contract()["feature_contracts"].get(family)
    if contract is None:
        return {"status": "not_applicable", "scope": "no authored local occupancy/depth probe for this family"}
    if "depth_axis" in contract:
        depth = float(np.ptp(data.vertices[:, contract["depth_axis"]]))
        error = abs(depth - contract["expected_depth_world"])
        return {"status": "passed" if error <= contract["max_error_world"] else "failed",
                "depth_world": depth, "error_world": error, "contract": contract}
    tree = BVHTree.FromPolygons(data.vertices.tolist(), data.faces.tolist(), all_triangles=True)
    direction = Vector((1., .37139, .11973)).normalized()
    def occupied(point):
        origin = Vector(point)
        hits = 0
        # Points are strictly interior/exterior probes, separated from boundaries.
        for _ in range(64):
            hit, _, _, _ = tree.ray_cast(origin, direction)
            if hit is None:
                return hits % 2 == 1
            hits += 1
            origin = hit + direction * 1e-6
        raise ValueError("feature ray exceeded its intersection bound")
    rows = [{"point": point, "expected_occupied": expected, "occupied": occupied(point)}
            for key, expected in (("empty_points_world", False), ("occupied_points_world", True))
            for point in contract.get(key, [])]
    return {"status": "passed" if all(r["occupied"] == r["expected_occupied"] for r in rows) else "failed",
            "probes": rows, "scope": "frozen local features, not a complete volume/boundary qualification"}


def coverage_acceptance(receipts):
    """Required failures/missing evidence block independently of all averages."""
    blockers = []
    workload = quality_workload()
    for case in workload["cases"]:
        name = case["name"]
        row = receipts.get(name)
        if not isinstance(row, dict):
            blockers.append(name + ": missing required case")
            continue
        for metric in ("silhouette", "surface", "topology", "editability"):
            value = row.get(metric, {})
            if value.get("status") != "passed":
                blockers.append(name + ": " + metric + " " + str(value.get("status", "missing")))
        if row.get("surface", {}).get("surface_passed") is not True or row.get("surface", {}).get("noise_qualified") is not True:
            blockers.append(name + ": surface failure or unqualified reference noise")
        if row.get("topology", {}).get("screen", {}).get("valid_solid") is not True:
            blockers.append(name + ": topology/volume screen unavailable/failed")
        if row.get("editability", {}).get("passed") is not True:
            blockers.append(name + ": artist edit response unavailable/failed")
        if row.get("topology", {}).get("boundary_qualified") is not True:
            blockers.append(name + ": boundary qualification unavailable/failed")
        views = row.get("silhouette", {}).get("views", {})
        for view in workload["views"]:
            thresholds = views.get(view, {}).get("thresholds", {})
            if (thresholds.get("min_boundary_iou") is None or
                    thresholds.get("max_signed_distance_loss") is None):
                blockers.append(name + ": boundary/signed-distance gate unavailable for " + view)
            if views.get(view, {}).get("passed") is not True:
                blockers.append(name + ": required silhouette view " + view)
        features = row.get("features", {})
        if name in coverage_fixture_contract()["feature_contracts"] and features.get("status") != "passed":
            blockers.append(name + ": required depth/cavity/part feature unavailable/failed")
    return {"status": "passed" if not blockers else "blocked", "blockers": blockers,
            "required_cases": len(workload["cases"]), "aggregate_scores_can_override": False}
