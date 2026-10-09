#!/usr/bin/env python3
"""Authored reference neutral previews in exactly matched retained native frames.

No masks, normal passes, fitting, surface comparisons or topology qualification.
Preserve the old display-managed inspection contract and legacy evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"blender_blocking"), str(ROOT), str(ROOT/"scripts")]
import test_runner
from run_surface_quality_check import _write, _replay_orthographic_camera
from run_quality_coverage_check import save_mesh
from run_family_source_edit_check import FAMILIES, _compile_exact, _cleanup, _publish_receipt


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shading_state(obj):
    """Observe native source and evaluated polygon/normal state, without edits."""
    import bpy
    import numpy as np
    def mesh_state(mesh):
        smooth = np.asarray([p.use_smooth for p in mesh.polygons], bool)
        sharp = mesh.attributes.get("sharp_edge")
        bits = [] if sharp is None else [bool(value.value) for value in sharp.data]
        return {"vertices": len(mesh.vertices), "polygons": len(mesh.polygons),
                "smooth_polygons": int(smooth.sum()), "flat_polygons": int((~smooth).sum()),
                "polygon_style": "mixed" if smooth.any() and not smooth.all() else "smooth" if smooth.all() else "flat",
                "smooth_flags_sha256": hashlib.sha256(smooth.tobytes()).hexdigest(),
                "sharp_edge_attribute": "absent" if sharp is None else "present",
                "sharp_edges": sum(bits), "sharp_edge_flags_sha256": hashlib.sha256(bytes(bits)).hexdigest(),
                "has_custom_normals": bool(mesh.has_custom_normals)}
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    modifiers = []
    for modifier in obj.modifiers:
        record = {"name": modifier.name, "type": modifier.type,
                  "show_render": bool(modifier.show_render), "show_viewport": bool(modifier.show_viewport)}
        for key in ("width", "segments", "affect", "weight", "mode", "keep_sharp", "thresh"):
            if hasattr(modifier, key):
                value = getattr(modifier, key)
                if isinstance(value, (str, bool, int, float)):
                    record[key] = value
        modifiers.append(record)
    return {"status": "native_observed", "source_mesh": mesh_state(obj.data),
            "evaluated_mesh": mesh_state(evaluated.data), "modifiers": modifiers,
            "scope": "shading/style only; geometric-normal metrics remain separate"}


def style_comparison(source, candidate):
    a, b = source["evaluated_mesh"], candidate["evaluated_mesh"]
    def weighted_controls(state):
        return [{key: modifier.get(key) for key in ("keep_sharp", "mode", "weight", "thresh", "show_render", "show_viewport")}
                for modifier in state["modifiers"] if modifier["type"] == "WEIGHTED_NORMAL"]
    return {"status": "observed", "weighted_normal_controls_match": weighted_controls(source) == weighted_controls(candidate),
            "polygon_style_match": a["polygon_style"] == b["polygon_style"],
            "sharp_edge_count_match": a["sharp_edges"] == b["sharp_edges"],
            "custom_normal_state_match": a["has_custom_normals"] == b["has_custom_normals"],
            "modifier_type_sequence_match": [m["type"] for m in source["modifiers"]] == [m["type"] for m in candidate["modifiers"]],
            "source_polygon_style": a["polygon_style"], "candidate_polygon_style": b["polygon_style"],
            "scope": "descriptive authored/candidate shading comparison; no surface acceptance or smoothing repair implied"}


def main():
    import bpy
    import numpy as np
    from evaluation.canonical_artifacts import CANONICAL_VIEWS, camera_record, camera_frame_sha256, canonical_artifact_inventory
    from evaluation.reference_noise import oriented_surface_identity
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--capsule-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("new output must stay under isolated repository temp/tasks")
    candidates = json.loads(args.candidate_receipt.read_text())
    reference_results = json.loads((args.reference_root/"results.json").read_text())
    reference_workload = json.loads((args.reference_root/"frozen-workload.json").read_text())
    authored = {case["name"]: case for case in reference_workload["cases"]}
    inputs = {}
    for name in FAMILIES:
        candidate = candidates["cases"][name]
        candidate_folder = Path(candidate["artifact_directory"])
        source_folder = args.capsule_reference if name == "capsule" else args.reference_root/name
        source_row = json.loads((source_folder/"results.json").read_text()) if name == "capsule" else reference_results["cases"][name]
        if name == "capsule":
            case = json.loads((source_folder/"frozen-workload.json").read_text())["case"]
            if case["parameters"] != authored[name]["parameters"]:
                raise ValueError("corrected capsule recipe changed authored parameters")
        else:
            case = authored[name]
        cameras = source_row["reference_cameras"]
        original_masks = {}
        for view in CANONICAL_VIEWS:
            digest = _sha(source_folder/(view+"-mask.png"))
            if digest != cameras[view]["png_sha256"] or digest != candidate["observed_inputs"][view]["sha256"]:
                raise ValueError("frozen original mask binding changed: "+name+"/"+view)
            original_masks[view] = {"path": str((source_folder/(view+"-mask.png")).resolve()), "sha256": digest}
            current_neutral = candidate_folder/(view+"-neutral.png")
            binding = candidate["inspection_artifacts"]["views"][view]["artifacts"]["neutral"]
            if _sha(current_neutral) != binding["sha256"] or binding["geometry_binding"] != "verified_by_producer":
                raise ValueError("candidate neutral binding changed: "+name+"/"+view)
        if _sha(candidate_folder/"evaluated-exact.npz") != candidate["npz_sha256"] or _sha(candidate_folder/"evaluated.obj") != candidate["obj_sha256"]:
            raise ValueError("candidate geometry archives changed")
        source_geometry_hash = source_row["geometry_hash"]
        with np.load(source_folder/"evaluated-exact.npz", allow_pickle=False) as archive:
            arrays = GeometryArrays.capture(archive["vertices"], archive["faces"])
        if arrays.content_hash != source_geometry_hash:
            raise ValueError("authored retained geometry changed")
        inputs[name] = {"source_directory": str(source_folder.resolve()), "source_geometry_hash": source_geometry_hash,
            "source_npz_sha256": _sha(source_folder/"evaluated-exact.npz"), "source_obj_sha256": _sha(source_folder/"evaluated.obj"),
            "source_case": case, "original_cameras": cameras, "original_masks": original_masks,
            "candidate_directory": str(candidate_folder.resolve()), "candidate_geometry_hash": candidate["geometry_hash"],
            "candidate_program_sha256": _sha(candidate_folder/"program.json"),
            "candidate_camera_snapshots_sha256": _sha(candidate_folder/"camera-snapshots.json"),
            "candidate_neutral_bindings": {view: candidate["inspection_artifacts"]["views"][view]["artifacts"]["neutral"] for view in CANONICAL_VIEWS}}
    settings = {"engine": "BLENDER_WORKBENCH", "light": "STUDIO", "color_type": "SINGLE", "single_color": [.65,.65,.65],
                "shadows": True, "cavity": False, "color_mode": "BW"}
    frozen = {"protocol": "authored_reference_matched_neutral_v1", "families": list(FAMILIES), "inputs": inputs,
        "candidate_receipt": {"path": str(args.candidate_receipt.resolve()), "sha256": _sha(args.candidate_receipt)},
        "reference_workload_sha256": _sha(args.reference_root/"frozen-workload.json"),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in
          (Path(__file__), ROOT/"scripts/run_family_source_edit_check.py", ROOT/"scripts/run_surface_quality_check.py",
           ROOT/"blender_blocking/synthetic/quality_references.py", ROOT/"blender_blocking/synthetic/blender_builders.py",
           ROOT/"blender_blocking/primitives/capsule.py", ROOT/"blender_blocking/primitives/shape_program_compiler.py",
           ROOT/"blender_blocking/evaluation/reference_noise.py", ROOT/"blender_blocking/integration/blender_ops/silhouette_render.py")},
        "render_settings": settings, "resolution": [512,512], "views": list(CANONICAL_VIEWS),
        "reference_neutral_frames": 35, "candidate_renders": 0, "candidate_style_replays": 7,
        "mask_frames": 0, "normal_frames": 0, "fits": 0, "surface_comparisons": 0, "qualification_children": 0,
        "deadline_seconds": 120, "memory_limit_bytes": 8*1024**3, "threads": 2,
        "human_preview_contract": "existing display-managed neutral inspection; no coverage measurement claims",
        "source_polygon_flags": "preserve authored producer flags, sharp edges, custom normals and modifiers",
        "environment": {"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__}}
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(), producer="authored_reference_neutral_inspection", max_generated_bytes=67108864,
                     shared_inputs={"candidate_receipt": str(args.candidate_receipt.resolve()), "reference_root": str(args.reference_root.resolve())})
    with owner:
        _write(owner.root/"frozen-workload.json", frozen)
        owner.register_file("frozen-workload.json", "diagnostic")
        receipt = {"protocol": frozen["protocol"], "status": "running", "cases": {}, "aggregate_accepted": False}
        _publish_receipt(owner, receipt)
        for name in FAMILIES:
            print("source-neutral "+name, flush=True)
            folder = owner.root/name
            folder.mkdir()
            reference = compiled = None
            try:
                if time.monotonic()-started > 120:
                    raise TimeoutError("frozen source-neutral deadline exceeded")
                item = inputs[name]
                source_folder, candidate_folder = Path(item["source_directory"]), Path(item["candidate_directory"])
                with np.load(source_folder/"evaluated-exact.npz", allow_pickle=False) as archive:
                    original = GeometryArrays.capture(archive["vertices"], archive["faces"])
                reference = build_quality_reference(item["source_case"])
                rebuilt, hashes = save_mesh(reference.object, folder)
                identity = {"indexed_reference_hash": original.content_hash, "indexed_regenerated_hash": rebuilt.content_hash,
                    "reference_oriented_surface": oriented_surface_identity(original),
                    "regenerated_oriented_surface": oriented_surface_identity(rebuilt),
                    "scope": "exact binary64 oriented geometric equivalence for reference-surface comparison; no topology identity qualification"}
                identity["equivalent"] = identity["reference_oriented_surface"] == identity["regenerated_oriented_surface"]
                _write(folder/"reference-equivalence.json", identity)
                _write(folder/"source-recipe.json", item["source_case"])
                if not identity["equivalent"]:
                    raise ValueError("regenerated authored source differs from retained oriented geometry")
                with np.load(candidate_folder/"evaluated-exact.npz", allow_pickle=False) as archive:
                    captured = GeometryArrays.capture(archive["vertices"], archive["faces"])
                wire = json.loads((candidate_folder/"program.json").read_text())
                if wire != candidates["cases"][name]["program"]:
                    raise ValueError("retained candidate recipe changed")
                if name == "rounded_box":
                    wire = deepcopy(wire)
                    wire["root_nodes"][0]["parameters"]["bevel_segments"] = 8
                compiled, candidate_source, candidate_data = _compile_exact(wire, captured)
                styles = {"authored_source": shading_state(reference.object), "actual_candidate_replay": shading_state(candidate_source),
                          "candidate_indexed_geometry_hash": candidate_data.content_hash,
                          "source_indexed_geometry_hash": rebuilt.content_hash}
                styles["comparison"] = style_comparison(styles["authored_source"], styles["actual_candidate_replay"])
                _write(folder/"shading-style.json", styles)
                _cleanup(compiled)
                compiled = None
                candidate_cameras = json.loads((candidate_folder/"camera-snapshots.json").read_text())
                actual_cameras, pairs = {}, {}
                with silhouette_session(target_objects=[reference.object], resolution=(512,512), color_mode="BW",
                        transparent_bg=False, engine="BLENDER_EEVEE", background_color=(1,1,1,1), silhouette_color=(0,0,0,1)) as session:
                    session.scene.render.engine = "BLENDER_WORKBENCH"
                    shading = session.scene.display.shading
                    shading.light, shading.color_type = "STUDIO", "SINGLE"
                    shading.single_color = (.65,.65,.65)
                    shading.show_shadows, shading.show_cavity = True, False
                    shading.background_type = "WORLD"
                    view_settings = session.scene.view_settings
                    for view in CANONICAL_VIEWS:
                        _replay_orthographic_camera(session.camera, item["original_cameras"][view])
                        # render_silhouette_frame flushes the evaluated camera;
                        # snapshot only after the same native update as candidate.
                        bpy.context.view_layer.update()
                        record = camera_record(session.camera, resolution=(512,512),
                            pixel_aspect=(session.scene.render.pixel_aspect_x, session.scene.render.pixel_aspect_y))
                        frame_hash = camera_frame_sha256(record)
                        candidate_hash = camera_frame_sha256(candidate_cameras[view])
                        if frame_hash != candidate_hash:
                            raise ValueError("source/candidate actual frame differs: "+view)
                        candidate_binding = item["candidate_neutral_bindings"][view]
                        if candidate_binding["camera_sha256"] != candidate_hash or candidate_binding["render_settings"] != settings:
                            raise ValueError("candidate neutral settings/frame binding differs")
                        path = folder/(view+"-neutral.png")
                        render_silhouette_frame(session, path)
                        after_record = camera_record(session.camera, resolution=(512,512),
                            pixel_aspect=(session.scene.render.pixel_aspect_x, session.scene.render.pixel_aspect_y))
                        if camera_frame_sha256(after_record) != frame_hash:
                            raise ValueError("neutral render changed actual camera")
                        record.update(geometry_hash=rebuilt.content_hash,
                            render_passes={"neutral": settings}, executed_passes=["neutral"],
                            pass_artifacts={"neutral": {"path": path.name, "sha256": _sha(path),
                                "geometry_hash": rebuilt.content_hash, "camera_sha256": frame_hash}},
                            display_settings={"view_transform": view_settings.view_transform,
                                              "look": view_settings.look, "exposure": view_settings.exposure,
                                              "gamma": view_settings.gamma, "preview_only": True})
                        actual_cameras[view] = record
                        clips_match = all(record[key] == candidate_cameras[view][key] for key in ("clip_start", "clip_end"))
                        pairs[view] = {"actual_frame_match": True, "camera_sha256": frame_hash,
                            "actual_clipping_match": clips_match,
                            "source_neutral": {"path": str(path), "sha256": _sha(path), "geometry_hash": rebuilt.content_hash},
                            "candidate_neutral": {"path": str(candidate_folder/path.name), "sha256": candidate_binding["sha256"], "geometry_hash": captured.content_hash}}
                        if not clips_match:
                            raise ValueError("actual source/candidate clipping differs")
                unchanged = evaluated_arrays(reference.object).content_hash == rebuilt.content_hash
                if not unchanged:
                    raise ValueError("source previews changed evaluated reference geometry")
                for record in actual_cameras.values():
                    record["geometry_unchanged_after_passes"] = True
                _write(folder/"camera-snapshots.json", actual_cameras)
                inventory = canonical_artifact_inventory(folder, geometry_hash=rebuilt.content_hash,
                    camera_records=actual_cameras, reference_camera_records=item["original_cameras"],
                    pass_states={"mask": "unrun", "neutral": "completed", "normals": "unrun"})
                _write(folder/"canonical-inspection.json", inventory)
                receipt["cases"][name] = {"status": "matched_neutral_retained", "artifact_directory": str(folder),
                    **hashes, "reference_equivalence": identity, "paired_neutral_views": pairs,
                    "geometry_unchanged": unchanged, "shading_style": styles,
                    "inspection_artifacts": inventory, "canonical_source_inventory_complete": False,
                    "legacy_mask_binding": "retained unchanged; no new source mask/normal render or retroactive binding",
                    "surface_acceptance": "unqualified; human shading previews and geometric-normal metrics remain independent"}
            except Exception as exc:
                receipt["cases"][name] = {"status": "failed", "reason": type(exc).__name__+": "+str(exc)}
            finally:
                if compiled is not None:
                    _cleanup(compiled)
                if reference is not None:
                    for obj in reversed(reference.sources):
                        if obj.name in bpy.data.objects:
                            mesh = obj.data if obj.type == "MESH" else None
                            bpy.data.objects.remove(obj, do_unlink=True)
                            if mesh is not None and mesh.users == 0:
                                bpy.data.meshes.remove(mesh)
            _publish_receipt(owner, receipt)
            for path in folder.iterdir():
                if path.is_file():
                    owner.register_file(path.relative_to(owner.root), "diagnostic" if path.suffix == ".png" else "final_output")
        passed = all(row["status"] == "matched_neutral_retained" for row in receipt["cases"].values())
        receipt.update(status="matched_neutrals_retained" if passed else "failed", elapsed_seconds=time.monotonic()-started,
            run_root=str(owner.root), reference_neutral_frames=sum(len(row.get("paired_neutral_views",{})) for row in receipt["cases"].values()),
            source_mask_frames=0, source_normal_frames=0, candidate_renders=0)
        _publish_receipt(owner, receipt)
        print("REFERENCE_NEUTRAL_RESULT="+str(owner.root/"results.json"), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
