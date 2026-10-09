#!/usr/bin/env python3
"""Bounded frozen reference/edit/negative preparation; no reconstruction claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT), str(ROOT / "scripts")]
import test_runner  # expose existing qualified dependency paths after bundled packages
from run_surface_quality_check import _object, _write, _renders


def render_masks(obj, folder, bounds, views, *, camera_records=None,
                 inspection_passes=("mask", "neutral")):
    """Retain matched shaded inspection alongside the existing mask interface."""
    import numpy as np
    from PIL import Image
    cameras = {}
    paths = _renders(obj, folder, bounds, views, camera_records=camera_records,
                     captured_cameras=cameras, inspection_passes=inspection_passes)
    masks = {view: np.asarray(Image.open(path).convert("L")) < 128 for view, path in paths.items()}
    return masks, cameras


def save_mesh(obj, folder):
    import numpy as np
    from evaluation.comparable_geometry import export_evaluated_object
    from reconstruction.native_geometry import evaluated_arrays
    folder.mkdir(parents=True, exist_ok=True)
    data = evaluated_arrays(obj)
    path = export_evaluated_object(obj, folder / "evaluated.obj")
    np.savez_compressed(folder / "evaluated-exact.npz", vertices=data.vertices, faces=data.faces)
    return data, {"geometry_hash": data.content_hash,
                  "obj_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "npz_sha256": hashlib.sha256((folder / "evaluated-exact.npz").read_bytes()).hexdigest()}


def main():
    import bpy
    import numpy as np
    from mathutils import Vector
    from reconstruction.native_geometry import evaluated_arrays, GeometryArrays
    from reconstruction.grouped_solids import solid_guard
    from synthetic.quality_references import (build_quality_reference, coverage_fixture_contract,
                                              quality_feature_verdict, coverage_acceptance)
    from evaluation.surface_quality import circular_profile_continuity
    from evaluation.canonical_artifacts import canonical_artifact_inventory, raw_surface_observation
    from evaluation.silhouette_eval import evaluate_silhouette_pair
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-renders", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "temp" / "tasks"):
        raise ValueError("output must be a new contained repository temp/tasks directory")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output must be empty; prior evidence is preserved")
    workload = coverage_fixture_contract()
    workload["source_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    workload["source_files_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "scripts/run_quality_coverage_check.py", ROOT / "blender_blocking/synthetic/quality_references.py",
                  ROOT / "blender_blocking/synthetic/quality_contracts.py", ROOT / "blender_blocking/synthetic/blender_builders.py",
                  ROOT / "scripts/run_surface_quality_check.py", ROOT / "blender_blocking/evaluation/canonical_artifacts.py")}
    workload["working_tree_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest()
    workload["environment"] = {"blender": bpy.app.version_string, "python": sys.version,
                               "numpy": np.__version__, "numpy_path": np.__file__}
    _write(output / "frozen-workload.json", workload)  # freeze before any candidate measurement
    receipt = {"status": "running", "cases": {}, "negative_controls": {},
               "scope": workload["execution_scope"]}
    started = time.monotonic()
    for case in workload["cases"]:
        if time.monotonic() - started > workload["limits"]["timeout_seconds"]:
            raise TimeoutError("frozen bounded reference workload exceeded its time budget")
        name = case["name"]
        print("coverage-reference " + name, flush=True)
        reference = build_quality_reference(case)
        obj = reference.object
        folder = output / name
        data, hashes = save_mesh(obj, folder)
        bounds = (Vector(data.vertices.min(axis=0)), Vector(data.vertices.max(axis=0)))
        screen = solid_guard(data)
        scale = obj.scale.copy()
        obj.scale.x *= 1.05
        bpy.context.view_layer.update()
        edited = evaluated_arrays(obj)
        ratio = float(np.ptp(edited.vertices[:, 0]) / np.ptp(data.vertices[:, 0]))
        obj.scale = scale
        bpy.context.view_layer.update()
        restored = evaluated_arrays(obj).content_hash == data.content_hash
        row = {**hashes, "topology": {"status": "blocked", "screen": screen,
               "boundary_qualified": False, "reason": "index/volume screen does not establish boundary intersections"},
               "editability": {"status": "passed" if abs(ratio - 1.05) < 1e-5 and restored else "failed",
                               "scope": "live object X-scale response and exact restoration", "passed": abs(ratio - 1.05) < 1e-5 and restored, "width_ratio": ratio, "restored": restored},
               "surface": {"status": "blocked", "reason": "authored reference; independent family noise qualification and reconstruction absent"},
               "features": quality_feature_verdict(name, data),
               "silhouette": {"status": "blocked", "reason": "reference masks are inputs, not reconstructed acceptance"}}
        if name == "smooth_vase":
            row["profile_continuity"] = circular_profile_continuity(data.vertices)
        cameras = {}
        if not args.skip_renders:
            reference_masks, cameras = render_masks(obj, folder, bounds, workload["views"])
            row["reference_cameras"] = cameras
        row["canonical_inspection"] = canonical_artifact_inventory(
            folder, geometry_hash=data.content_hash, camera_records=cameras,
            pass_states={"mask": "unrun" if args.skip_renders else "completed",
                         "neutral": "unrun" if args.skip_renders else "completed", "normals": "unrun"})
        _write(folder / "canonical-inspection.json", row["canonical_inspection"])
        for control, declaration in workload["negative_controls"].items():
            if declaration["family"] != name:
                continue
            if control == "incorrect_depth":
                vertices = data.vertices.copy()
                vertices[:, 1] *= declaration["depth_scale"]
                negative = GeometryArrays.capture(vertices, data.faces)
            else:
                for modifier in obj.modifiers:
                    if modifier.type == "BOOLEAN":
                        modifier.show_viewport = False
                        modifier.show_render = False
                bpy.context.view_layer.update()
                negative = evaluated_arrays(obj)
                for modifier in obj.modifiers:
                    if modifier.type == "BOOLEAN":
                        modifier.show_viewport = True
                        modifier.show_render = True
                bpy.context.view_layer.update()
            control_obj = _object(negative.vertices, negative.faces, control)
            control_folder = output / control
            negative, negative_hashes = save_mesh(control_obj, control_folder)
            feature = quality_feature_verdict(name, negative)
            control_row = {**negative_hashes, "features": feature,
                           "expected_failure_detected": feature["status"] == "failed",
                           "surface_observation": raw_surface_observation(data, negative),
                           "surface_qualification": "raw observations only; independent family gates remain unqualified"}
            control_cameras = {}
            if not args.skip_renders:
                masks, control_cameras = render_masks(control_obj, control_folder, bounds, workload["views"], camera_records=cameras)
                control_row["silhouette"] = {v: evaluate_silhouette_pair(reference_masks[v], masks[v], view=v)
                                             for v in workload["views"]}
                control_row["cameras"] = control_cameras
            control_row["canonical_inspection"] = canonical_artifact_inventory(
                control_folder, geometry_hash=negative.content_hash, camera_records=control_cameras,
                reference_camera_records=cameras if not args.skip_renders else None,
                pass_states={"mask": "unrun" if args.skip_renders else "completed",
                             "neutral": "unrun" if args.skip_renders else "completed", "normals": "unrun"})
            _write(control_folder / "canonical-inspection.json", control_row["canonical_inspection"])
            receipt["negative_controls"][control] = control_row
            mesh = control_obj.data
            bpy.data.objects.remove(control_obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        receipt["cases"][name] = row
        _write(output / "results.json", receipt)
        for source in reversed(reference.sources):
            mesh = source.data
            bpy.data.objects.remove(source, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    receipt["coverage_acceptance"] = coverage_acceptance(receipt["cases"])
    ready = (len(receipt["cases"]) == 12 and
             all(r["editability"]["status"] == "passed" and r["topology"]["screen"]["valid_solid"]
                 for r in receipt["cases"].values()) and
             len(receipt["negative_controls"]) == 3 and
             all(r["expected_failure_detected"] for r in receipt["negative_controls"].values()))
    receipt["status"] = "references_and_controls_ready" if ready else "reference_or_control_failure"
    receipt["elapsed_seconds"] = time.monotonic() - started
    _write(output / "results.json", receipt)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
