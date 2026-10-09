#!/usr/bin/env python3
"""Selected frozen family reconstructions; raw independent quality verdicts.

Run inside the authorized Blender runtime. Geometry preparation is the default;
rendering prepared rows is a separate, explicit bounded stage.
"""
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
import test_runner
from run_surface_quality_check import _write
from run_quality_coverage_check import render_masks, save_mesh


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arguments():
    from reconstruction.frozen_family import FROZEN_FAMILIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--capsule-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True, choices=FROZEN_FAMILIES)
    parser.add_argument("--coverage-transfer", type=Path,
                        help="Explicit 16-bit grayscale transfer and adjacent transfer-settings.json")
    parser.add_argument("--reuse", action="append", default=[], metavar="FAMILY=RECEIPT")
    parser.add_argument("--prepared-run", type=Path,
                        help="Existing geometry-stage results.json; inspection does not refit")
    parser.add_argument("--render", action="store_true",
                        help="Retain five mask/neutral views under the original cameras")
    parser.add_argument("--fit-evaluations", type=int, default=256)
    parser.add_argument("--fit-seconds", type=float, default=3.)
    parser.add_argument("--deadline-seconds", type=float, default=180.)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if len(set(args.families)) != len(args.families) or len(args.families) > 7:
        parser.error("select at most seven distinct families per bounded run")
    if not 0 < args.deadline_seconds <= 600:
        parser.error("deadline must be positive and at most 600 seconds")
    if args.render != (args.prepared_run is not None):
        parser.error("rendering requires prepared geometry; prepared geometry requires --render")
    return args


def _reference_inputs(args, name, reference_receipt):
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import CANONICAL_VIEWS
    folder = args.capsule_reference if name == "capsule" and args.capsule_reference else args.reference_root/name
    row = json.loads((folder/"results.json").read_text()) if folder == args.capsule_reference else reference_receipt["cases"][name]
    cameras = row["reference_cameras"]
    masks, grayscale, inputs = {}, {}, {}
    for view in CANONICAL_VIEWS:
        path = folder/(view+"-mask.png")
        digest = _sha(path)
        if cameras[view].get("png_sha256") != digest:
            raise ValueError("source mask hash differs from its frozen camera receipt: "+view)
        image = np.asarray(Image.open(path).convert("L"))
        if image.shape != (512, 512):
            raise ValueError("frozen masks must retain original 512 resolution")
        masks[view], grayscale[view] = image < 128, image.astype(float)/255
        inputs[view] = {"path": str(path.resolve()), "sha256": digest}
    return folder, cameras, masks, grayscale, inputs


def _coverage(args):
    if args.coverage_transfer is None:
        return None, None
    import numpy as np
    from PIL import Image
    path = args.coverage_transfer.resolve()
    encoded = np.asarray(Image.open(path), float).reshape(-1)/65535
    if not 64 <= len(encoded) <= 8193:
        raise ValueError("explicit coverage LUT is outside bounded sample range")
    settings = path.with_name("transfer-settings.json")
    record = {"path": str(path), "sha256": _sha(path),
              "settings": json.loads(settings.read_text()), "settings_sha256": _sha(settings),
              "scope": "declared display transfer for frozen black/white images only"}
    return (np.linspace(0, 1, len(encoded)), encoded), record


def _prepared_inputs(prepared, families):
    """Freeze actual retained archive/OBJ/recipe identities before inspection."""
    records = {}
    for name in families:
        prior = prepared["cases"][name]
        folder = Path(prior.get("artifact_directory", str(Path(prepared["run_root"])/name)))
        record = {"artifact_directory": str(folder.resolve()), "geometry_hash": prior["geometry_hash"]}
        for filename, key in (("evaluated-exact.npz", "npz_sha256"), ("evaluated.obj", "obj_sha256")):
            actual = _sha(folder/filename)
            if actual != prior[key]:
                raise ValueError("prepared source identity changed: "+name+"/"+filename)
            record[key] = actual
        record["program_sha256"] = _sha(folder/"program.json")
        origin = prior.get("origin_receipt")
        if origin is not None and _sha(origin["path"]) != origin["sha256"]:
            raise ValueError("prepared source receipt changed: "+name)
        records[name] = record
    return records


def _world_bevel(root, radius):
    """Keep a fitted world radius independent of the descendant cube scale."""
    import bpy
    from reconstruction.output_targets import output_mesh_targets
    meshes = output_mesh_targets([root])
    if len(meshes) != 1:
        raise ValueError("world bevel adapter requires one retained mesh child")
    obj = meshes[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for modifier in obj.modifiers:
        if modifier.type == "BEVEL":
            modifier.width = float(radius)
            modifier.segments = 8
    bpy.context.view_layer.update()


def _edit_response(obj, data):
    import bpy
    import numpy as np
    from reconstruction.native_geometry import evaluated_arrays
    scale = obj.scale.copy()
    obj.scale.x *= 1.05
    bpy.context.view_layer.update()
    changed = evaluated_arrays(obj)
    ratio = float(np.ptp(changed.vertices[:, 0])/np.ptp(data.vertices[:, 0]))
    obj.scale = scale
    bpy.context.view_layer.update()
    restored = evaluated_arrays(obj).content_hash == data.content_hash
    passed = bool(abs(ratio-1.05) <= 1e-5 and restored)
    return {"status": "passed" if passed else "failed", "passed": passed,
            "width_ratio": ratio, "exact_restoration": restored,
            "scope": "live object X scale and exact restored evaluated arrays",
            "semantic_artist_edit": {"status": "unrun", "reason": "object scaling does not qualify family-specific artist controls"}}


def main():
    import bpy
    import numpy as np
    from mathutils import Vector
    from reconstruction.coverage_evidence import coverage_from_grayscale
    from reconstruction.frozen_family import (SUPPORTED_FAMILIES, initial_family_rows,
                                               observed_target, fitted_family_program, retained_family_program, retained_triangle_indices)
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from reconstruction.grouped_solids import solid_guard
    from primitives.shape_program_compiler import compile_shape_program
    from evaluation.canonical_artifacts import (CANONICAL_VIEWS, canonical_artifact_inventory,
                                               raw_surface_observation)
    from evaluation.silhouette_eval import SilhouetteGateConfig, evaluate_silhouette_pair
    from synthetic.quality_references import quality_feature_verdict
    from utils.run_ownership import OwnedRun

    args = _arguments()
    parent = args.output.resolve()
    if not parent.is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("output must stay below the isolated repository temp/tasks")
    reused = {}
    for item in args.reuse:
        family, path = item.split("=", 1)
        path = Path(path).resolve()
        if family in reused or not path.is_file():
            raise ValueError("reused family receipts must be unique existing files")
        reused[family] = {"status": "reused", "source_receipt": str(path),
                          "source_sha256": _sha(path), "aggregate_accepted": False,
                          "scope": "retained previous actual result; no reconstruction rerun"}
    rows = initial_family_rows(args.families, reused)
    rows.update(reused)
    reference_receipt = json.loads((args.reference_root/"results.json").read_text())
    prepared = None if args.prepared_run is None else json.loads(args.prepared_run.read_text())
    prepared_inputs = {} if prepared is None else _prepared_inputs(prepared, args.families)
    transfer, transfer_record = _coverage(args)
    started = time.monotonic()
    owner = OwnedRun(parent, producer="frozen_family_reconstruction", max_generated_bytes=268435456,
                     shared_inputs={"reference_root": str(args.reference_root.resolve()),
                                    "capsule_reference": str(args.capsule_reference),
                                    "prepared_run": str(args.prepared_run), "reused": reused})
    with owner:
        output = owner.root
        frozen = {"protocol": "frozen_family_actual_v1", "selected_families": args.families,
                  "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "working_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                  "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in
                                     (Path(__file__), ROOT/"blender_blocking/reconstruction/frozen_family.py",
                                      ROOT/"blender_blocking/reconstruction/structured_family.py",
                                      ROOT/"blender_blocking/reconstruction/polygon_proposals.py",
                                      ROOT/"blender_blocking/evaluation/canonical_artifacts.py",
                                      ROOT/"scripts/run_quality_coverage_check.py",
                                      ROOT/"scripts/run_surface_quality_check.py",
                                      ROOT/"blender_blocking/reconstruction/oriented_support.py",
                                      ROOT/"blender_blocking/primitives/shape_program_compiler.py",
                                      ROOT/"blender_blocking/evaluation/reference_noise.py")},
                  "fit": {"active": not args.render, "evaluations_per_case": 0 if args.render else args.fit_evaluations, "seconds_per_case": 0 if args.render else args.fit_seconds,
                          "inputs": "observed masks, explicit coverage and calibrated cameras only"},
                  "candidate_tessellation": {"radial_segments": 96, "rounded_box_bevel_segments": 8},
                  "surface": {"sample_count_per_direction": 4096, "seed": 61007,
                              "acceptance_limits": None, "status": "unqualified independently per family"},
                  "silhouette": {"area_iou_min": .7, "boundary_iou_min": .8, "signed_distance_loss_max": .05},
                  "resolution": [512, 512], "canonical_views": list(CANONICAL_VIEWS),
                  "phase": "inspection" if args.render else "geometry_only", "coverage_transfer": transfer_record,
                  "render_passes": ["mask", "neutral", "normals"] if args.render else [],
                  "render_frames": len(args.families)*15 if args.render else 0,
                  "geometry_replay": "saved fitted program local mesh and pose; exact world-array identity required" if args.render else "new fitted candidate",
                  "prepared_receipt": None if args.prepared_run is None else {"path": str(args.prepared_run.resolve()), "sha256": _sha(args.prepared_run)},
                  "prepared_geometry_sources": prepared_inputs,
                  "deadline_seconds": args.deadline_seconds, "qualification_children": 0,
                  "environment": {"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__}}
        _write(output/"frozen-workload.json", frozen)
        receipt = {"protocol": frozen["protocol"], "status": "running", "cases": rows,
                   "run_root": str(output), "aggregate_accepted": False,
                   "coverage_acceptance": "blocked: independent surface, boundary and semantic-edit requirements"}
        _write(output/"results.json", receipt)
        for name in args.families:
            if name not in SUPPORTED_FAMILIES:
                rows[name].update(status="unsupported", reason="no structured observed-mask proposal in this bounded runner")
                continue
            if time.monotonic()-started >= args.deadline_seconds:
                rows[name].update(status="incomplete", reason="frozen elapsed allowance ended before row")
                continue
            print("family-"+("inspect " if args.render else "prepare ")+name, flush=True)
            folder = output/name
            folder.mkdir()
            try:
                reference_folder, cameras, masks, grayscale, input_records = _reference_inputs(args, name, reference_receipt)
                if prepared is None:
                    coverages = None if transfer is None else {view: coverage_from_grayscale(image, *transfer)
                                                               for view, image in grayscale.items()}
                    target = observed_target(masks, cameras, coverage_masks=coverages)
                    program = fitted_family_program(name, target, masks, cameras, coverage_masks=coverages,
                                                     max_evaluations=args.fit_evaluations, max_elapsed_s=args.fit_seconds)
                    # Retain fitted recipe before any reference geometry is read.
                    _write(folder/"program.json", program.to_dict())
                    compiled = compile_shape_program(program, lathe_segments=96, weighted_normals=False)
                    obj = compiled.root_object
                    if name == "rounded_box":
                        _world_bevel(obj, program.root_nodes[0].parameters["corner_radius_world"])
                    data, hashes = save_mesh(obj, folder)
                    source = np.load(reference_folder/"evaluated-exact.npz", allow_pickle=False)
                    reference = GeometryArrays.capture(source["vertices"], source["faces"])
                    observation = raw_surface_observation(reference, data, count=4096, seed=61007)
                    row = {"status": "measured", **hashes, "program": program.to_dict(),
                           "surface_observation": observation,
                           "features": quality_feature_verdict(name, data),
                           "topology": {"status": "unqualified", "screen": solid_guard(data),
                                        "boundary_qualified": False, "reason": "native solid-boundary qualification is a separate selected stage"},
                           "editability": _edit_response(obj, data), "aggregate_accepted": False}
                else:
                    prior = prepared["cases"][name]
                    if prior["reference_camera_sha256"] != hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest():
                        raise ValueError("prepared camera input changed")
                    if prior["observed_inputs"] != input_records:
                        raise ValueError("prepared observed mask input changed")
                    source_folder = Path(prior.get("artifact_directory", str(Path(prepared["run_root"])/name)))
                    if _sha(source_folder/"evaluated-exact.npz") != prior["npz_sha256"]:
                        raise ValueError("prepared exact geometry archive changed")
                    source = np.load(source_folder/"evaluated-exact.npz", allow_pickle=False)
                    captured = GeometryArrays.capture(source["vertices"], source["faces"])
                    if captured.content_hash != prior["geometry_hash"]:
                        raise ValueError("prepared geometry identity changed")
                    program = retained_family_program(prior["program"])
                    compiled = compile_shape_program(program, lathe_segments=96, weighted_normals=False)
                    if name == "rounded_box":
                        _world_bevel(compiled.root_object, program.root_nodes[0].parameters["corner_radius_world"])
                    meshes = output_mesh_targets([compiled.root_object])
                    if len(meshes) != 1:
                        raise ValueError("prepared family must retain one mesh descendant")
                    obj = meshes[0]
                    replayed = evaluated_arrays(obj)
                    if replayed.content_hash != captured.content_hash and name in {"sphere", "anisotropic_ellipsoid"}:
                        indices = retained_triangle_indices(captured, replayed)
                        local_vertices = [tuple(vertex.co) for vertex in obj.data.vertices]
                        indexed_mesh = bpy.data.meshes.new(obj.data.name+"FrozenTriangleOrder")
                        indexed_mesh.from_pydata(local_vertices, [], indices.tolist())
                        indexed_mesh.update()
                        obj.data = indexed_mesh
                        bpy.context.view_layer.update()
                    data, hashes = save_mesh(obj, folder)
                    if data.content_hash != captured.content_hash:
                        from evaluation.reference_noise import oriented_surface_identity
                        mesh = obj.data
                        mesh.calc_loop_triangles()
                        local = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], float)
                        local_faces = np.asarray([tuple(face.vertices) for face in mesh.loop_triangles], int)
                        np.savez_compressed(folder/"replay-local.npz", vertices=local, faces=local_faces)
                        _write(folder/"replay-pose.json", {"matrix_world": [list(line) for line in obj.matrix_world],
                                                         "frozen_program": prior["program"]})
                        _write(folder/"replay-identity-diagnostic.json", {
                            "required_indexed_geometry_hash": captured.content_hash,
                            "replayed_indexed_geometry_hash": data.content_hash,
                            "source_oriented_surface": oriented_surface_identity(captured),
                            "replayed_oriented_surface": oriented_surface_identity(data),
                            "vertices_exactly_equal": bool(np.array_equal(data.vertices, captured.vertices)),
                            "faces_exactly_equal": bool(np.array_equal(data.faces, captured.faces)),
                            "render_admission": "blocked; permutation diagnostic never overrides indexed hash"})
                        raise ValueError("inspection object changed prepared geometry")
                    row = {**prior, **hashes, "prepared_receipt": str(args.prepared_run.resolve()),
                           "prepared_receipt_sha256": _sha(args.prepared_run)}
                    _write(folder/"program.json", row["program"])
                row["artifact_directory"] = str(folder)
                row["observed_inputs"] = input_records
                row["reference_camera_sha256"] = hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest()
                if args.render:
                    candidate_masks, replayed = render_masks(obj, folder, (Vector(data.vertices.min(axis=0)),
                                                         Vector(data.vertices.max(axis=0))), CANONICAL_VIEWS,
                                                         camera_records=cameras,
                                                         inspection_passes=("mask", "neutral", "normals"))
                    gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
                    views = {view: evaluate_silhouette_pair(masks[view], candidate_masks[view], view=view, config=gates)
                             for view in CANONICAL_VIEWS}
                    passed = all(item["passed"] for item in views.values())
                    row["silhouette"] = {"status": "passed" if passed else "failed", "views": views}
                    row["inspection_artifacts"] = canonical_artifact_inventory(
                        folder, geometry_hash=data.content_hash, camera_records=replayed,
                        reference_camera_records=cameras,
                        pass_states={"mask": "completed", "neutral": "completed", "normals": "completed"})
                    _write(folder/"camera-snapshots.json", replayed)
                    row["canonical_passes_complete"] = all(
                        row["inspection_artifacts"]["views"][view]["artifacts"][name]["status"] == "available"
                        for view in CANONICAL_VIEWS for name in ("mask", "neutral", "normals"))
                else:
                    row["silhouette"] = {"status": "unrun", "reason": "geometry-only phase has no rendered admission"}
                    row["inspection_artifacts"] = canonical_artifact_inventory(
                        folder, geometry_hash=data.content_hash, camera_records={},
                        pass_states={"mask": "unrun", "neutral": "unrun", "normals": "unrun"})
                if row["silhouette"]["status"] == "failed":
                    row["status"] = "failed"
                elif args.render and row["inspection_artifacts"]["status"] == "complete" and row["canonical_passes_complete"]:
                    row["status"] = "measured"
                else:
                    row["status"] = "incomplete"
                rows[name] = row
            except Exception as exc:
                rows[name] = {"status": "failed", "reason": type(exc).__name__+": "+str(exc),
                              "aggregate_accepted": False}
            _write(output/"results.json", receipt)
            for path in folder.iterdir():
                if path.is_file():
                    owner.register_file(path.relative_to(output), "final_output" if path.suffix in {".npz", ".obj", ".json"} else "diagnostic")
        receipt["status"] = "actual_rows_retained"
        receipt["elapsed_seconds"] = time.monotonic()-started
        receipt["row_status_counts"] = {status: sum(row["status"] == status for row in rows.values())
                                        for status in ("measured", "reused", "unrun", "unsupported", "incomplete", "failed")}
        _write(output/"results.json", receipt)
        owner.register_file("frozen-workload.json", "diagnostic")
        owner.register_file("results.json", "final_output")
        print("FAMILY_RESULT="+str(output/"results.json"), flush=True)
    return 1 if any(rows[name]["status"] == "failed" for name in args.families) else 0


if __name__ == "__main__":
    raise SystemExit(main())
