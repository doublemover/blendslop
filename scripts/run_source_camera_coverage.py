#!/usr/bin/env python3
"""Recapture ten source-only neutral frames for retained torus and concave arch."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT), str(ROOT / "scripts")]
import test_runner
from evaluation.canonical_artifacts import CANONICAL_VIEWS, camera_frame_sha256
from run_family_surface_contracts import arrays, read_bound
from run_selected_canonical_inspection import reference_equivalence, require_same_frame

FAMILIES = ("torus", "concave_arch")
PROTOCOL = "source_camera_coverage_v1"
MEMORY_BYTES = 8 * 1024 ** 3
RUNTIME_FILES = (
    "scripts/run_source_camera_coverage.py", "scripts/run_family_surface_contracts.py",
    "scripts/run_selected_canonical_inspection.py", "scripts/run_reference_neutral_inspection.py",
    "scripts/run_family_source_edit_check.py", "scripts/run_surface_quality_check.py",
    "scripts/run_quality_coverage_check.py", "blender_blocking/test_runner.py",
    "blender_blocking/synthetic/quality_references.py", "blender_blocking/synthetic/quality_contracts.py",
    "blender_blocking/synthetic/blender_builders.py", "blender_blocking/synthetic/specs.py",
    "blender_blocking/reconstruction/native_geometry.py", "blender_blocking/reconstruction/output_targets.py",
    "blender_blocking/reconstruction/mesh_io.py", "blender_blocking/metrics/topology_receipt.py",
    "blender_blocking/metrics/topology.py",
    "blender_blocking/evaluation/reference_noise.py", "blender_blocking/evaluation/canonical_artifacts.py",
    "blender_blocking/evaluation/family_surface_contracts.py", "blender_blocking/evaluation/comparable_geometry.py",
    "blender_blocking/integration/blender_ops/silhouette_render.py",
    "blender_blocking/utils/run_ownership.py", "blender_blocking/utils/json_io.py",
    "blender_blocking/integration/blender_ops/camera_framing.py",
    "blender_blocking/integration/blender_ops/profile_loft_mesh.py",
    "blender_blocking/geometry/profile_models.py",
    "scripts/run_bounded_owned_command.py", "blender_blocking/utils/owned_process_supervisor.py",
)


def bind(path):
    from run_bounded_owned_command import _file_record
    record = _file_record(path, max_bytes=16777216)
    return {"path": record["path"], "sha256": record["sha256"]}


def read_json(binding):
    return json.loads(read_bound(binding))


def require_candidate_neutral(record, geometry_hash, artifact):
    """Check a retained candidate frame and real pass bytes without adopting ownership."""
    binding = record.get("pass_artifacts", {}).get("neutral", {})
    frame = require_same_frame(record, record)
    if (record.get("geometry_hash") != geometry_hash or
            record.get("geometry_unchanged_after_passes") is not True or
            binding.get("geometry_hash") != geometry_hash or
            binding.get("camera_sha256") != frame or
            binding.get("sha256") != artifact["sha256"] or
            Path(artifact["path"]).name != binding.get("path") or
            "neutral" not in record.get("executed_passes", ()) or
            not isinstance(record.get("render_passes", {}).get("neutral"), dict)):
        raise ValueError("retained candidate neutral geometry/frame/pass binding differs")
    read_bound(artifact)
    return frame


def require_retained_producer(record, geometry_hash, artifact, producer_view):
    frame = require_candidate_neutral(record, geometry_hash, artifact)
    neutral = producer_view["artifacts"]["neutral"]
    if (producer_view["camera"]["sha256"] != frame or
            neutral.get("producer_status") != "completed" or
            neutral.get("geometry_binding") != "verified_by_producer" or
            neutral.get("geometry_hash") != geometry_hash or
            neutral.get("camera_sha256") != frame or
            neutral.get("sha256") != artifact["sha256"]):
        raise ValueError("retained candidate producer receipt differs")
    return frame


def prepare_plan(reference_root, candidate_receipt):
    """Freeze existing authored source inputs and original candidate declarations."""
    reference_root = Path(reference_root).resolve(strict=True)
    workload_binding = bind(reference_root / "frozen-workload.json")
    references_binding = bind(reference_root / "results.json")
    candidate_binding = bind(candidate_receipt)
    authored = {case["name"]: case for case in read_json(workload_binding)["cases"]}
    references, candidates = read_json(references_binding)["cases"], read_json(candidate_binding)["cases"]
    cases = {}
    for name in FAMILIES:
        source = references[name]
        candidate = candidates[name]
        candidate_folder = Path(candidate["artifact_directory"])
        source_npz = bind(reference_root / name / "evaluated-exact.npz")
        if arrays(source_npz).content_hash != source["geometry_hash"]:
            raise ValueError("retained authored source indexed geometry differs")
        candidate_npz = bind(candidate_folder / "evaluated-exact.npz")
        if (candidate_npz["sha256"] != candidate["npz_sha256"] or
                arrays(candidate_npz).content_hash != candidate["geometry_hash"]):
            raise ValueError("retained candidate indexed archive differs")
        camera_binding = bind(candidate_folder / "camera-snapshots.json")
        cameras = read_json(camera_binding)
        masks, neutral = {}, {}
        for view in CANONICAL_VIEWS:
            mask = bind(reference_root / name / (view + "-mask.png"))
            if (mask["sha256"] != source["reference_cameras"][view]["png_sha256"] or
                    candidate["observed_inputs"][view] != mask):
                raise ValueError("original source/candidate mask declaration differs")
            masks[view] = mask
            neutral[view] = bind(candidate_folder / (view + "-neutral.png"))
            require_retained_producer(cameras[view], candidate["geometry_hash"], neutral[view],
                candidate["inspection_artifacts"]["views"][view])
        cases[name] = {"source_case": authored[name], "source_npz": source_npz,
            "source_geometry_hash": source["geometry_hash"], "original_cameras": source["reference_cameras"],
            "original_masks": masks, "candidate_npz": candidate_npz,
            "candidate_geometry_hash": candidate["geometry_hash"], "candidate_camera_records": camera_binding,
            "candidate_neutral_artifacts": neutral}
    plan = {"protocol": PROTOCOL, "families": list(FAMILIES), "cases": cases,
        "reference_workload": workload_binding, "reference_receipt": references_binding,
        "candidate_receipt": candidate_binding, "views": list(CANONICAL_VIEWS),
        "resolution": [512, 512], "passes": ["neutral"], "native_frames": 10,
        "candidate_renders": 0, "candidate_replays": 0, "mask_frames": 0, "normal_frames": 0,
        "fits": 0, "raw_comparisons": 0, "qualifier_children": 0, "threads": 2,
        "work_seconds": 85, "join_seconds": 5, "rss_limit_bytes": MEMORY_BYTES,
        "committed_limit_bytes": MEMORY_BYTES,
        "clip_scope": "fresh source uses exact recorded candidate clipping; legacy original source clips remain unavailable",
        "preview_scope": "display-managed neutral inspection only; no linear coverage or acceptance claim",
        "source_style": "preserve authored polygons, sharp flags, modifiers and custom normals",
        "input_sha256": {}}
    paths = required_inputs(plan)
    plan["input_sha256"] = {str(path): bind(path)["sha256"] for path in sorted(paths)}
    validate_plan(plan)
    return plan


def required_inputs(plan):
    paths = {str((ROOT / name).resolve()) for name in RUNTIME_FILES}
    paths.update(plan[key]["path"] for key in ("reference_workload", "reference_receipt", "candidate_receipt"))
    for entry in plan["cases"].values():
        paths.update(entry[key]["path"] for key in ("source_npz", "candidate_npz", "candidate_camera_records"))
        for key in ("original_masks", "candidate_neutral_artifacts"):
            paths.update(row["path"] for row in entry[key].values())
    return paths


def validate_plan(plan):
    exact = {"protocol": PROTOCOL, "families": list(FAMILIES), "views": list(CANONICAL_VIEWS),
        "resolution": [512, 512], "passes": ["neutral"], "native_frames": 10,
        "candidate_renders": 0, "candidate_replays": 0, "mask_frames": 0, "normal_frames": 0,
        "fits": 0, "raw_comparisons": 0, "qualifier_children": 0, "threads": 2,
        "work_seconds": 85, "join_seconds": 5, "rss_limit_bytes": MEMORY_BYTES,
        "committed_limit_bytes": MEMORY_BYTES}
    if any(plan.get(key) != value for key, value in exact.items()) or set(plan.get("cases", {})) != set(FAMILIES):
        raise ValueError("source-only frozen scope differs")
    expected = plan.get("input_sha256", {})
    required = required_inputs(plan)
    if not required.issubset(expected) or not 1 <= len(expected) <= 64:
        raise ValueError("source-camera inputs are not all hash-bound")
    for filename, digest in expected.items():
        read_bound({"path": filename, "sha256": digest})
    authored = {row["name"]: row for row in read_json(plan["reference_workload"])["cases"]}
    references = read_json(plan["reference_receipt"])["cases"]
    candidates = read_json(plan["candidate_receipt"])["cases"]
    for name, entry in plan["cases"].items():
        source, candidate = references[name], candidates[name]
        if (entry["source_case"] != authored[name] or
                entry["source_geometry_hash"] != source["geometry_hash"] or
                entry["original_cameras"] != source["reference_cameras"] or
                entry["candidate_geometry_hash"] != candidate["geometry_hash"]):
            raise ValueError("authored source/original camera declaration differs")
        if arrays(entry["source_npz"]).content_hash != entry["source_geometry_hash"]:
            raise ValueError("source indexed geometry differs")
        if (arrays(entry["candidate_npz"]).content_hash != entry["candidate_geometry_hash"] or
                entry["candidate_npz"]["sha256"] != candidate["npz_sha256"]):
            raise ValueError("candidate indexed geometry differs")
        candidate_folder = Path(candidate["artifact_directory"]).resolve()
        source_folder = Path(plan["reference_receipt"]["path"]).parent / name
        if (Path(entry["source_npz"]["path"]) != source_folder / "evaluated-exact.npz" or
                Path(entry["candidate_npz"]["path"]) != candidate_folder / "evaluated-exact.npz" or
                Path(entry["candidate_camera_records"]["path"]) != candidate_folder / "camera-snapshots.json"):
            raise ValueError("declared source/candidate artifact locations differ")
        cameras = read_json(entry["candidate_camera_records"])
        if set(cameras) != set(CANONICAL_VIEWS):
            raise ValueError("candidate camera declarations are incomplete")
        for view in CANONICAL_VIEWS:
            camera_frame_sha256(entry["original_cameras"][view])
            mask = entry["original_masks"][view]
            if (mask["sha256"] != source["reference_cameras"][view]["png_sha256"] or
                    mask != candidate["observed_inputs"][view]):
                raise ValueError("original mask binding differs")
            read_bound(mask)
            artifact = entry["candidate_neutral_artifacts"][view]
            if (Path(mask["path"]) != source_folder / (view + "-mask.png") or
                    Path(artifact["path"]) != candidate_folder / (view + "-neutral.png")):
                raise ValueError("original mask/candidate preview artifact locations differ")
            require_retained_producer(cameras[view], entry["candidate_geometry_hash"], artifact,
                candidate["inspection_artifacts"]["views"][view])


def source_declaration(entry, folder, regenerated_hash, cameras, equivalence):
    """Emit only the report's existing source-only seam after all fresh bindings pass."""
    if (equivalence.get("equivalent") is not True or
            equivalence.get("indexed_reference_hash") != entry["source_geometry_hash"] or
            equivalence.get("indexed_regenerated_hash") != regenerated_hash or
            set(cameras) != set(CANONICAL_VIEWS)):
        raise ValueError("fresh source oriented equivalence/camera coverage unavailable")
    original = arrays(entry["source_npz"])
    regenerated_binding = bind(Path(folder) / "evaluated-exact.npz")
    regenerated = arrays(regenerated_binding)
    if (regenerated.content_hash != regenerated_hash or
            reference_equivalence(original, regenerated) != equivalence or
            read_json(bind(Path(folder) / "camera-snapshots.json")) != cameras):
        raise ValueError("fresh source archive/camera bytes differ from producer")
    candidate_cameras = read_json(entry["candidate_camera_records"])
    neutral = {}
    for view, camera in cameras.items():
        artifact = bind(Path(folder) / (view + "-neutral.png"))
        require_candidate_neutral(camera, regenerated_hash, artifact)
        require_same_frame(camera, candidate_cameras[view])
        actual = camera["pass_artifacts"]["neutral"].get("actual_camera")
        if not isinstance(actual, dict):
            raise ValueError("fresh per-pass actual camera unavailable")
        require_same_frame(camera, actual)
        neutral[view] = artifact
    return {"authored_parameters": deepcopy(entry["source_case"]["parameters"]),
        "reference_npz": deepcopy(entry["source_npz"]), "reference_geometry_hash": entry["source_geometry_hash"],
        "regenerated_npz": regenerated_binding,
        "regenerated_geometry_hash": regenerated_hash,
        "oriented_surface_sha256": equivalence["reference_oriented_surface"],
        "camera_records": bind(Path(folder) / "camera-snapshots.json"), "neutral_artifacts": neutral}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    plan = read_json({"path": str(args.plan.resolve()), "sha256": args.plan_sha256})
    validate_plan(plan)
    output = args.output.absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / "temp/tasks"):
        raise ValueError("fresh output must stay under the isolated worktree temp/tasks")
    import bpy
    from evaluation.canonical_artifacts import camera_record, canonical_artifact_inventory
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import evaluated_arrays
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_quality_coverage_check import save_mesh
    from run_reference_neutral_inspection import shading_state
    from run_selected_canonical_inspection import _settings
    from run_surface_quality_check import _replay_orthographic_camera, _write
    started = time.monotonic()
    deadline = started + plan["work_seconds"]
    receipt = {"protocol": PROTOCOL, "status": "running", "cases": {}, "completed_native_frames": 0,
        "candidate_renders": 0, "candidate_replays": 0, "fits": 0, "raw_comparisons": 0, "qualifier_children": 0,
        "lifecycle_scope": "fresh workload artifacts; complete-tree join belongs to the separate bounded supervisor",
        "environment": {"blender": bpy.app.version_string, "python": sys.version}}
    owner = OwnedRun(output, producer="source_camera_coverage", max_generated_bytes=134217728,
                     shared_inputs=plan["input_sha256"])
    with owner:
        def publish(relative, value):
            path = owner.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, value)
            owner.register_file(relative, "diagnostic")
        publish("frozen-workload.json", plan)
        publish("results.json", receipt)
        declarations = {}
        for name in FAMILIES:
            before_objects = set(bpy.data.objects)
            folder = owner.root / name
            folder.mkdir()
            row = {"status": "blocked", "artifact_directory": str(folder)}
            receipt["cases"][name] = row
            try:
                if time.monotonic() > deadline:
                    raise TimeoutError("source camera capture deadline exceeded")
                entry = plan["cases"][name]
                original = arrays(entry["source_npz"])
                source = build_quality_reference(entry["source_case"])
                rebuilt, archive_hashes = save_mesh(source.object, folder)
                equivalence = reference_equivalence(original, rebuilt)
                publish(Path(name) / "reference-equivalence.json", equivalence)
                publish(Path(name) / "source-recipe.json", entry["source_case"])
                row.update(reference_equivalence=equivalence, **archive_hashes)
                if not equivalence["equivalent"]:
                    raise ValueError("authored source exact oriented geometry cannot be reproduced")
                style = shading_state(source.object)
                publish(Path(name) / "shading-style-before.json", style)
                candidate_cameras = read_json(entry["candidate_camera_records"])
                cameras = {}
                with silhouette_session(target_objects=[source.object], resolution=(512, 512),
                        color_mode="BW", transparent_bg=False, engine="BLENDER_WORKBENCH",
                        force_material=False, ensure_light_obj=False) as session:
                    if session.target_objects != [source.object]:
                        raise ValueError("source preview mesh target differs")
                    shading = session.scene.display.shading
                    shading.light, shading.color_type = "STUDIO", "SINGLE"
                    shading.single_color = (.65, .65, .65)
                    shading.show_shadows, shading.show_cavity = True, False
                    shading.background_type = "WORLD"
                    for view in CANONICAL_VIEWS:
                        if time.monotonic() > deadline:
                            raise TimeoutError("source camera capture deadline exceeded")
                        replay = {**entry["original_cameras"][view], **{key: candidate_cameras[view][key]
                            for key in ("clip_start", "clip_end")}}
                        _replay_orthographic_camera(session.camera, replay)
                        bpy.context.view_layer.update()
                        capture = lambda: camera_record(session.camera, resolution=(512, 512),
                            pixel_aspect=(session.scene.render.pixel_aspect_x, session.scene.render.pixel_aspect_y))
                        actual = capture()
                        frame = require_same_frame(actual, candidate_cameras[view])
                        if evaluated_arrays(source.object).content_hash != rebuilt.content_hash:
                            raise ValueError("source geometry changed before neutral frame")
                        owner.reserve_bytes(8388608)
                        path = folder / (view + "-neutral.png")
                        settings = _settings(session.scene, "neutral")
                        render_silhouette_frame(session, path)
                        receipt["completed_native_frames"] += 1
                        after = capture()
                        require_same_frame(actual, after)
                        if (_settings(session.scene, "neutral") != settings or
                                evaluated_arrays(source.object).content_hash != rebuilt.content_hash):
                            raise ValueError("neutral frame changed source geometry/settings")
                        artifact = bind(path)
                        actual.update(geometry_hash=rebuilt.content_hash, render_passes={"neutral": settings},
                            executed_passes=["neutral"], geometry_unchanged_after_passes=False,
                            pass_artifacts={"neutral": {"path": path.name, "sha256": artifact["sha256"],
                                "geometry_hash": rebuilt.content_hash, "camera_sha256": frame, "actual_camera": after}})
                        cameras[view] = actual
                        owner.register_file(path.relative_to(owner.root), "final_output")
                        publish(Path(name) / "camera-snapshots.json", cameras)
                        publish("results.json", receipt)
                style_after = shading_state(source.object)
                publish(Path(name) / "shading-style-after.json", style_after)
                if style != style_after or evaluated_arrays(source.object).content_hash != rebuilt.content_hash:
                    raise ValueError("source inspection changed authored geometry/shading style")
                for camera in cameras.values():
                    camera["geometry_unchanged_after_passes"] = True
                publish(Path(name) / "camera-snapshots.json", cameras)
                inventory = canonical_artifact_inventory(folder, geometry_hash=rebuilt.content_hash,
                    camera_records=cameras, reference_camera_records=candidate_cameras,
                    pass_states={"mask": "unrun", "neutral": "completed", "normals": "unrun"})
                publish(Path(name) / "canonical-inspection.json", inventory)
                declaration = source_declaration(entry, folder, rebuilt.content_hash, cameras, equivalence)
                declarations[name] = declaration
                row.update(status="source_five_view_binding_complete", source_declaration=declaration,
                    camera_records=cameras, actual_frames_and_clips_match=True, shading_state_restored=True,
                    inspection_artifacts=inventory, complete_three_pass_inventory=False,
                    legacy_scope="original masks/owners unchanged; no source clip/geometry history backfilled")
            except Exception as exc:
                row.update(status="blocked", reason=repr(exc))
            finally:
                for path in folder.rglob("*"):
                    if path.is_file():
                        owner.register_file(path.relative_to(owner.root), "diagnostic")
                for obj in reversed(list(bpy.data.objects)):
                    if obj not in before_objects:
                        mesh = obj.data if obj.type == "MESH" else None
                        bpy.data.objects.remove(obj, do_unlink=True)
                        if mesh is not None and mesh.users == 0:
                            bpy.data.meshes.remove(mesh)
            publish("results.json", receipt)
        publish("source-declarations.json", {"protocol": "source-camera-declarations-v1", "families": declarations,
            "scope": "fresh source-only report additions; report policy and historical owners remain unchanged"})
        receipt.update(status="completed" if len(declarations) == len(FAMILIES) else "blocked",
            elapsed_seconds=time.monotonic() - started, run_root=str(owner.root))
        publish("results.json", receipt)
        print(json.dumps({"run_root": str(owner.root), "status": receipt["status"],
                          "completed_native_frames": receipt["completed_native_frames"]}), flush=True)
        if receipt["status"] != "completed":
            raise RuntimeError("one or more authored source camera captures blocked; retained receipt is authoritative")


if __name__ == "__main__":
    main()