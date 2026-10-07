#!/usr/bin/env python3
"""Bounded native surface controls; independent of the completed full suite.

Run inside Blender with --background --factory-startup --python-exit-code 1
--python scripts/run_surface_quality_check.py -- --output temp/tasks/surface-quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT)]


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _object(vertices, faces, name):
    import bpy
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def _view(camera, name, lo, hi):
    from mathutils import Vector
    from integration.blender_ops.camera_framing import configure_ortho_camera_for_view
    center = (lo + hi) / 2
    scale = max(hi - lo) * 1.16
    if name in {"front", "side", "top"}:
        configure_ortho_camera_for_view(camera, name, lo, hi, margin_frac=.08, resolution=(512, 512))
    else:
        az, elevation = map(float, name.split("_")[1:])
        az, elevation = math.radians(az), math.radians(elevation)
        direction = Vector((math.cos(az) * math.cos(elevation), math.sin(az) * math.cos(elevation), math.sin(elevation)))
        camera.location = center + direction * scale * 3
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = scale


def _renders(obj, directory, bounds, views):
    import bpy
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import evaluated_arrays
    before = evaluated_arrays(obj).content_hash
    paths = {}
    with silhouette_session(target_objects=[obj], resolution=(512, 512), color_mode="BW",
                            transparent_bg=False, engine="BLENDER_EEVEE",
                            background_color=(1, 1, 1, 1), silhouette_color=(0, 0, 0, 1)) as session:
        for view in views:
            _view(session.camera, view, *bounds)
            path = directory / (view + "-mask.png")
            render_silhouette_frame(session, path)
            paths[view] = str(path)
        # Neutral inspection is a separate material/render pass in the same
        # cameras and unchanged evaluated geometry.
        session.scene.render.engine = "BLENDER_WORKBENCH"
        shading = session.scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "SINGLE"
        shading.single_color = (.65, .65, .65)
        shading.show_shadows = True
        shading.show_cavity = False
        shading.background_type = "WORLD"
        for view in views:
            _view(session.camera, view, *bounds)
            render_silhouette_frame(session, directory / (view + "-neutral.png"))
        # World-space shading normal visualization. Geometric normal-angle
        # metrics below remain independent of interpolated shading normals.
        session.scene.render.engine = "BLENDER_EEVEE"
        normal_mat = bpy.data.materials.new("SurfaceNormalInspection")
        normal_mat.use_nodes = True
        nodes, links = normal_mat.node_tree.nodes, normal_mat.node_tree.links
        nodes.clear()
        geometry = nodes.new("ShaderNodeNewGeometry")
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs[3].default_value = .5
        offset = nodes.new("ShaderNodeVectorMath")
        offset.operation = "ADD"
        offset.inputs[1].default_value = (.5, .5, .5)
        emission = nodes.new("ShaderNodeEmission")
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(geometry.outputs["Normal"], scale.inputs[0])
        links.new(scale.outputs["Vector"], offset.inputs[0])
        links.new(offset.outputs["Vector"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(normal_mat)
        for view in views:
            _view(session.camera, view, *bounds)
            render_silhouette_frame(session, directory / (view + "-normals.png"))
    after = evaluated_arrays(obj).content_hash
    if after != before:
        raise ValueError("render passes changed evaluated geometry")
    return paths


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from mathutils import Vector
    from geometry.profile_models import EllipticalSlice
    from integration.blender_ops.profile_loft_mesh import create_loft_mesh_from_slices
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.grouped_solids import solid_guard
    from evaluation.comparable_geometry import export_evaluated_object
    from evaluation.surface_quality import compare_surface_arrays
    from synthetic.quality_contracts import quality_workload, rounded_triangle_mesh, triangle_preservation

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-renders", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "temp" / "tasks"):
        raise ValueError("output must be scoped under repository temp/tasks")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("completed/partial results already exist; use a new explicitly scoped output")
    workload = quality_workload()
    workload["source_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    workload["executed_scope"] = "one affected loft test; analytic vase reference/smooth/stepped controls; sharp corners; rounded triangle reference/circularized/banded controls"
    workload["environment"] = {"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__}
    _write(output / "frozen-workload.json", workload)
    started = time.monotonic()
    receipt = {"source_head": workload["source_head"], "cases": {}, "status": "running"}
    _write(output / "results.json", receipt)
    from test_profile_loft_mesh import TestProfileLoftMesh
    test = unittest.TestSuite([TestProfileLoftMesh("test_surface_modes_are_connected_and_keep_flat_caps")])
    result = unittest.TextTestRunner(verbosity=2).run(test)
    receipt["affected_native_test"] = {"run": result.testsRun, "passed": result.wasSuccessful()}
    if not result.wasSuccessful():
        receipt["status"] = "failed_native_test"
        _write(output / "results.json", receipt)
        return 1
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    def vase(n):
        return [EllipticalSlice(z=float(z), rx=float(.6 + .2 * math.cos(2 * math.pi * z / 2.6)),
                                ry=float(.6 + .2 * math.cos(2 * math.pi * z / 2.6)))
                for z in np.linspace(0, 2.6, n)]
    reference = create_loft_mesh_from_slices(vase(257), name="analytic_vase_reference", radial_segments=192,
                                           surface_mode="sharp")
    # Reference geometry is a dense tessellation of the declared analytic curve;
    # smooth normals improve inspection but do not change geometric metrics.
    for p in reference.data.polygons:
        zs = [reference.data.vertices[i].co.z for i in p.vertices]
        p.use_smooth = max(zs) - min(zs) > 1e-9
    smooth = create_loft_mesh_from_slices(vase(65), name="smooth_vase_candidate", radial_segments=192,
                                        surface_mode="smooth", surface_subdivisions=4)
    stepped = create_loft_mesh_from_slices(vase(65), name="stepped_vase_negative", radial_segments=192,
                                         surface_mode="stepped")
    sharp = create_loft_mesh_from_slices([EllipticalSlice(z=z, rx=r, ry=r) for z, r in ((0, .8), (1.3, .4), (2.6, .8))],
                                       name="authored_sharp_control", radial_segments=192, surface_mode="sharp")
    v, f = rounded_triangle_mesh()
    triangle = _object(v, f, "rounded_triangle_reference")
    circular = v.copy()
    radial = np.linalg.norm(circular[:, :2], axis=1)
    dome_scale = np.sqrt(np.maximum(0, 1 - (circular[:, 2] / .24) ** 2))
    active = radial > 1e-9
    circular[active, :2] *= (1.06 * dome_scale[active] / radial[active])[:, None]
    circle = _object(circular, f, "circularized_triangle_negative")
    banded = v.copy()
    banded[:, 2] = np.round(banded[:, 2] / .04) * .04
    bands = _object(banded, f, "banded_triangle_negative")
    objects = [reference, smooth, stepped, sharp, triangle, circle, bands]
    ref_arrays = evaluated_arrays(reference)
    tri_arrays = evaluated_arrays(triangle)
    vase_bounds = (Vector((-.8, -.8, 0)), Vector((.8, .8, 2.6)))
    triangle_bounds = (Vector(v.min(axis=0)), Vector(v.max(axis=0)))
    mask_paths = {}
    for obj in objects:
        if time.monotonic() - started > 1200:
            raise TimeoutError("frozen native workload timeout exceeded")
        print("surface-case " + obj.name, flush=True)
        folder = output / obj.name
        folder.mkdir(exist_ok=True)
        arrays = evaluated_arrays(obj)
        mesh = export_evaluated_object(obj, folder / "evaluated.obj")
        # Preserve full binary64 arrays as well as human-readable OBJ transport.
        np.savez_compressed(folder / "evaluated-exact.npz", vertices=arrays.vertices, faces=arrays.faces)
        entry = {"geometry_hash": arrays.content_hash, "obj_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                 "topology_screen": solid_guard(arrays), "surface": None,
                 "editability": {"status": "not_measured"}, "silhouette": {"status": "pending_render"}}
        if obj in {reference, smooth, stepped}:
            entry["surface"] = compare_surface_arrays(ref_arrays, arrays)
        elif obj in {triangle, circle, bands}:
            entry["surface"] = compare_surface_arrays(tri_arrays, arrays)
            entry["authored_shape"] = triangle_preservation(arrays.vertices)
        if not args.skip_renders:
            mask_paths[obj.name] = _renders(obj, folder, triangle_bounds if obj in {triangle, circle, bands} else vase_bounds, workload["views"])
        receipt["cases"][obj.name] = entry
        _write(output / "results.json", receipt)
    for name, paths in mask_paths.items():
        ref = "rounded_triangle_reference" if "triangle" in name else "analytic_vase_reference"
        if name == "authored_sharp_control":
            receipt["cases"][name]["silhouette"] = {"status": "not_compared_to_smooth_reference", "reason": "intended sharp reference has its own shape"}
            continue
        views = {}
        for view, path in paths.items():
            a = np.asarray(Image.open(mask_paths[ref][view]).convert("L")) < 128
            b = np.asarray(Image.open(path).convert("L")) < 128
            views[view] = {"area_iou": float((a & b).sum() / max(1, (a | b).sum())),
                           "existing_gate_preserved": .700,
                           "area_passed": bool((a & b).sum() / max(1, (a | b).sum()) >= .700)}
        receipt["cases"][name]["silhouette"] = {"status": "measured_area_only", "views": views,
            "boundary_and_signed_distance": "not substituted; existing E2E gates require their own reconstruction run"}
    expected = {
        "reference_surface_pass": receipt["cases"][reference.name]["surface"]["surface_passed"],
        "smooth_surface_pass": receipt["cases"][smooth.name]["surface"]["surface_passed"],
        "stepped_surface_rejected": not receipt["cases"][stepped.name]["surface"]["surface_passed"],
        "triangle_reference_shape_pass": receipt["cases"][triangle.name]["authored_shape"]["passed"],
        "circularized_triangle_rejected": not receipt["cases"][circle.name]["authored_shape"]["passed"],
        "banded_triangle_surface_rejected": not receipt["cases"][bands.name]["surface"]["surface_passed"],
        "sharp_control_not_smoothed": not any(p.use_smooth for p in sharp.data.polygons),
    }
    receipt["control_contracts"] = expected
    receipt["status"] = "controls_passed" if all(expected.values()) else "controls_failed"
    receipt["coverage_acceptance"] = "pending twelve-family reconstruction, independent family noise qualification and editability"
    receipt["historical_vase_acceptance"] = "pending repaired reconstruction against retained observed input/evaluated mesh; analytic control pass is not historical vase acceptance"
    receipt["elapsed_seconds"] = time.monotonic() - started
    _write(output / "results.json", receipt)
    return 0 if all(expected.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
