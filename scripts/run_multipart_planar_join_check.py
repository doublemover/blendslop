#!/usr/bin/env python3
"""One explicit shared-plane multipart candidate; no fitting or history rewrites."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT / "scripts"), str(ROOT)]
import test_runner
from evaluation.canonical_artifacts import CANONICAL_VIEWS, camera_record
from run_selected_canonical_inspection import load_exact, read_json, reference_equivalence, require_same_frame, sha

FAMILY = "asymmetric_multipart_solid"
BASELINE = "b2c937a949368471cb472b2d1cd3aa12f2d0a39871365d257a15154ff4edecb1"
SOURCE = "2142108d7cff97c2bbd65c222a799cbc308e913a4b5c4893abe6d23c839945ca"
PROTOCOL = "multipart_explicit_shared_plane_check_v1"


def prepare_plan():
    """Freeze current source bytes and exact retained inputs before native work."""
    from reconstruction.multipart_planar_join import shared_far_plane_program
    retained_plan = ROOT / "docs/quality-selected-canonical-continuation-20261009/native-plan.json"
    selected = read_json(retained_plan)
    entry = selected["cases"][FAMILY]
    if entry["candidate_geometry_hash"] != BASELINE or entry["source_geometry_hash"] != SOURCE:
        raise ValueError("selected multipart identities changed")
    load_exact(entry["candidate_npz"], BASELINE)
    load_exact(entry["source_npz"], SOURCE)
    wire = read_json(entry["candidate_program"])
    if wire != entry["candidate_recipe"]:
        raise ValueError("retained multipart recipe changed")
    proposal = shared_far_plane_program(wire).to_dict()
    case = read_json(entry["candidate_receipt"])["cases"][FAMILY]
    if case["geometry_hash"] != BASELINE or case["program"] != wire or not case["source_edit"]["passed"]:
        raise ValueError("retained recipe/edit does not bind selected body")
    saved = Path(entry["candidate_receipt"]).parent
    old_frozen = read_json(saved / "frozen-workload.json")
    packet = ROOT / "temp/tasks/quality-continuation-20261009/selected-canonical-01/owned-lnxl2keh" / FAMILY
    source_cameras = read_json(packet / "source/camera-snapshots.json")
    engineering = ROOT / "docs/quality-family-surface-contracts-continuation-20261009/eleven-family-evidence.json"
    limits = read_json(engineering)["selected_multipart_repair"]["engineering_limits"]
    if limits != {"normal_angle_p95_degrees_max": 1., "symmetric_mean_distance_world_max": .0018125506404794065}:
        raise ValueError("independently frozen multipart engineering limits changed")
    files = set(selected["input_sha256"])
    files.update(str(ROOT / p) for p in (
        "scripts/run_multipart_planar_join_check.py", "blender_blocking/reconstruction/multipart_planar_join.py",
        "blender_blocking/test_multipart_planar_join.py", "blender_blocking/evaluation/inspection_camera.py",
        "blender_blocking/test_inspection_camera.py", "blender_blocking/reconstruction/native_qualification.py",
        "scripts/qualify_native_solid.py", "blender_blocking/evaluation/triangle_contacts.py",
        "scripts/run_bounded_owned_command.py", "blender_blocking/utils/owned_process_supervisor.py",
        "blender_blocking/utils/run_ownership.py", "blender_blocking/utils/owned_process.py",
        "blender_blocking/utils/primary_process_cleanup.py", "blender_blocking/test_runner.py"))
    files.update([str(retained_plan), str(engineering), str(saved / "frozen-workload.json"), str(packet / "source/camera-snapshots.json"), old_frozen["python"]])
    for key in ("candidate_npz", "source_npz", "candidate_program", "candidate_receipt"):
        files.add(entry[key])
    for view in CANONICAL_VIEWS:
        original = entry["original_masks"][view]
        if sha(original["path"]) != original["sha256"]:
            raise ValueError("original silhouette input changed")
        files.add(original["path"])
        neutral = source_cameras[view]["pass_artifacts"]["neutral"]
        path = packet / "source" / neutral["path"]
        if sha(path) != neutral["sha256"] or neutral["geometry_hash"] != SOURCE:
            raise ValueError("retained source neutral pass changed")
        files.add(str(path))
    return {"protocol": PROTOCOL, "family": FAMILY, "baseline_hash": BASELINE, "source_hash": SOURCE,
            "entry": entry, "proposal": proposal, "source_cameras": source_cameras,
            "source_neutral_root": str(packet / "source"), "qualifier_python": old_frozen["python"],
            "engineering_limits": limits, "input_sha256": {str(Path(p).resolve()): sha(p) for p in sorted(files)},
            "render_frames": 12, "candidate_frames": 11, "source_frames": 1,
            "candidate_original_masks": 5, "candidate_original_neutrals": 5,
            "display_only_unclipped_neutrals": {"candidate": 1, "source": 1},
            "reused_source_gate_masks": 5, "reused_source_neutral_frames": 5,
            "reused_source_lifecycle_scope": "retained byte/frame binding only; prior failed supervisor is not released or rewritten",
            "measurement": "original display-calibrated mask gates unchanged; no controlled-alpha relabeling",
            "fits": 0, "raw_pairs": 1, "samples_per_direction": 4096, "seed": 61007,
            "qualifier_children": 1, "helper_timeout_seconds": 15., "threads": 2,
            "work_seconds": 85., "join_seconds": 5., "rss_bytes": 8 * 1024 ** 3,
            "committed_bytes": 8 * 1024 ** 3, "resolution": [512, 512],
            "artist_limits": None, "aggregate_accepted": False,
            "scope": "explicit new geometry model; original b2c, source, failed receipts and selected UI unchanged"}


def validate_plan(plan):
    """Refuse changed family/model/resource scope before native allocation."""
    from reconstruction.multipart_planar_join import shared_far_plane_program
    expected = {"protocol": PROTOCOL, "family": FAMILY, "baseline_hash": BASELINE, "source_hash": SOURCE,
                "render_frames": 12, "candidate_frames": 11, "source_frames": 1,
                "candidate_original_masks": 5, "candidate_original_neutrals": 5,
                "display_only_unclipped_neutrals": {"candidate": 1, "source": 1},
                "reused_source_gate_masks": 5, "reused_source_neutral_frames": 5,
                "fits": 0, "raw_pairs": 1, "samples_per_direction": 4096, "seed": 61007,
                "qualifier_children": 1, "helper_timeout_seconds": 15., "threads": 2,
                "work_seconds": 85., "join_seconds": 5., "rss_bytes": 8 * 1024 ** 3,
                "committed_bytes": 8 * 1024 ** 3, "resolution": [512, 512],
                "artist_limits": None, "aggregate_accepted": False,
                "engineering_limits": {"normal_angle_p95_degrees_max": 1.,
                                       "symmetric_mean_distance_world_max": .0018125506404794065}}
    for key, value in expected.items():
        if plan.get(key) != value or type(plan.get(key)) is not type(value):
            raise ValueError("unexpected frozen planar-join scope: " + key)
    entry = plan["entry"]
    if (entry["candidate_geometry_hash"] != BASELINE or entry["source_geometry_hash"] != SOURCE
            or set(entry["original_cameras"]) != set(CANONICAL_VIEWS)
            or set(entry["original_masks"]) != set(CANONICAL_VIEWS)
            or set(plan["source_cameras"]) != set(CANONICAL_VIEWS)
            or shared_far_plane_program(entry["candidate_recipe"]).to_dict() != plan["proposal"]):
        raise ValueError("frozen family/recipe/camera identity differs from the declared shared-plane proposal")
    inputs = plan.get("input_sha256", {})
    if not isinstance(inputs, dict) or not 1 <= len(inputs) <= 80:
        raise ValueError("frozen input inventory must be bounded to eighty files")
    return plan


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from mathutils import Vector
    from evaluation.inspection_camera import inspection_frame_bounds, unclipped_inspection_camera
    from evaluation.canonical_artifacts import raw_surface_observation
    from evaluation.silhouette_eval import SilhouetteGateConfig, evaluate_silhouette_pair
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.multipart_planar_join import compile_shared_plane_multipart, shared_depth_coordinates
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.native_qualification import qualify_geometry
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_frozen_multipart_reconstruction import compile_live_multipart
    from run_quality_coverage_check import render_masks, save_mesh
    from run_reference_neutral_inspection import shading_state, style_comparison
    from run_surface_quality_check import _replay_orthographic_camera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    plan = validate_plan(read_json(args.plan))
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("frozen input changed: " + path)
    if not args.output.resolve().is_relative_to(ROOT / "temp/tasks"):
        raise ValueError("new output must stay inside isolated temp/tasks")
    entry = plan["entry"]
    source_data = load_exact(entry["source_npz"], SOURCE)
    baseline = load_exact(entry["candidate_npz"], BASELINE)
    owner = OwnedRun(args.output.resolve(), producer="multipart_explicit_shared_plane_check",
                     max_generated_bytes=134217728, shared_inputs={"plan": str(args.plan), "plan_sha256": sha(args.plan)})
    started = time.monotonic()
    receipt = {"protocol": PROTOCOL, "status": "running", "completed_native_frames": 0,
               "fits": 0, "artist_surface_limits": None, "aggregate_accepted": False}
    with owner:
        def publish(relative, value):
            path = owner.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            owner.register_file(relative, "final_output")
        def checkpoint():
            if time.monotonic() - started > 65:
                raise TimeoutError("insufficient reserved budget before fixed15s helper and final joins")
        def save(obj, folder):
            data, hashes = save_mesh(obj, folder)
            for name in ("evaluated-exact.npz", "evaluated.obj"):
                owner.register_file((folder / name).relative_to(owner.root), "final_output")
            return data, hashes
        publish("frozen-workload.json", plan)
        publish("results.json", receipt)
        try:
            old, _ = compile_live_multipart(retained_multipart_program(entry["candidate_recipe"]))
            replayed = evaluated_arrays(old)
            receipt["baseline_replay"] = {"expected_hash": BASELINE, "actual_hash": replayed.content_hash,
                                           "exact": replayed.content_hash == baseline.content_hash}
            if not receipt["baseline_replay"]["exact"]:
                raise ValueError("b2c baseline exact replay failed before renders")
            obj, parts, construction = compile_shared_plane_multipart(retained_multipart_program(plan["proposal"]))
            data, hashes = save(obj, owner.root / "candidate")
            receipt.update(candidate_geometry_hash=data.content_hash, candidate_artifacts=hashes, construction=construction)
            publish("candidate/program.json", plan["proposal"])
            # Direct source world vertices establish the actual shared plane.
            by_id = {p.get("blendslop_shape_node_id"): p for p in parts}
            base, arm = by_id[construction["base_node_id"]], by_id[construction["arm_node_id"]]
            def world(part):
                co = np.asarray([tuple(v.co) for v in part.data.vertices], float)
                matrix = np.asarray(part.matrix_world, float)
                return co @ matrix[:3, :3].T + matrix[:3, 3]
            far_base, far_arm = float(world(base)[:, 1].max()), float(world(arm)[:, 1].max())
            receipt["exact_native_far_plane"] = {"base": far_base, "arm": far_arm, "equal": far_base == far_arm}
            if far_base != far_arm:
                raise ValueError("native source far planes are not exactly shared")
            vertices = np.empty(len(arm.data.vertices) * 3, np.float32)
            arm.data.vertices.foreach_get("co", vertices)
            before = vertices.reshape(-1, 3).copy()
            def source_controls():
                return {"root_pointer": obj.as_pointer(), "sources": [
                    {"node": part.get("blendslop_shape_node_id"), "pointer": part.as_pointer(),
                     "mesh_pointer": part.data.as_pointer(), "matrix_world": [list(row) for row in part.matrix_world]}
                    for part in parts], "modifiers": [
                        {"name": m.name, "type": m.type, "show_render": bool(m.show_render),
                         "show_viewport": bool(m.show_viewport), "operation": getattr(m, "operation", None),
                         "solver": getattr(m, "solver", None),
                         "operand_pointer": m.object.as_pointer() if m.type == "BOOLEAN" and m.object else None}
                        for m in obj.modifiers]}
            original_controls = source_controls()
            if (len(original_controls["modifiers"]) != 2 or any(
                    m["type"] != "BOOLEAN" or m["operation"] != "UNION" or m["solver"] != "EXACT"
                    or not m["show_render"] or not m["show_viewport"] or m["operand_pointer"] is None
                    for m in original_controls["modifiers"])):
                raise ValueError("new model did not retain two live Exact union controls")
            depth = float(np.ptp(before[:, 1].astype(float)) * float(arm.scale.y))
            try:
                changed = shared_depth_coordinates(before, depth_world=depth * 1.05, frame_depth_world=float(arm.scale.y))
                arm.data.vertices.foreach_set("co", changed.ravel()); arm.data.update(); bpy.context.view_layer.update()
                edited = evaluated_arrays(obj)
                ratio = float(np.ptp(world(arm)[:, 1]) / depth)
                fixed_far = float(world(arm)[:, 1].max()) == far_base
            finally:
                arm.data.vertices.foreach_set("co", before.ravel()); arm.data.update(); bpy.context.view_layer.update()
            restored = evaluated_arrays(obj)
            edit = {"control": "tall-arm near-side depth x1.05; shared far plane fixed", "depth_ratio": ratio,
                    "fixed_far_plane": fixed_far, "union_geometry_changed": edited.content_hash != data.content_hash,
                    "same_source_and_mesh_pointers": source_controls() == original_controls,
                    "original_controls": original_controls, "restored_controls": source_controls(),
                    "baseline_hash": data.content_hash, "edited_hash": edited.content_hash, "restored_hash": restored.content_hash,
                    "exact_indexed_restoration": restored.content_hash == data.content_hash}
            edit["passed"] = bool(abs(ratio - 1.05) <= 1e-5 and fixed_far and edit["union_geometry_changed"]
                                  and edit["same_source_and_mesh_pointers"] and edit["exact_indexed_restoration"])
            receipt["source_edit"] = edit
            if not edit["passed"]:
                raise ValueError("specific aligned-depth source edit failed")
            checkpoint(); owner.reserve_bytes(33554432)
            camera_inputs = {v: {"clip_start": .1, "clip_end": 1000., **entry["original_cameras"][v]} for v in CANONICAL_VIEWS}
            cameras, views = {}, {}
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            for view in CANONICAL_VIEWS:
                checkpoint()
                masks, actual = render_masks(obj, owner.root / "candidate", (Vector(data.vertices.min(axis=0)), Vector(data.vertices.max(axis=0))),
                    (view,), camera_records=camera_inputs, inspection_passes=("mask", "neutral"))
                cameras.update(actual)
                original = np.asarray(Image.open(entry["original_masks"][view]["path"]).convert("L")) < 128
                views[view] = evaluate_silhouette_pair(original, masks[view], view=view, config=gates)
                for style in ("mask", "neutral"):
                    owner.register_file(Path("candidate") / (view + "-" + style + ".png"), "final_output")
                    receipt["completed_native_frames"] += 1
                require_same_frame(plan["source_cameras"][view], cameras[view])
                publish("candidate/camera-snapshots.json", cameras)
                publish("results.json", receipt)
            receipt["silhouette"] = {"all_five_passed": all(v["passed"] for v in views.values()), "views": views,
                                     "scope": plan["measurement"], "original_legacy_clip_scope": "unavailable; fresh actual neutral clips matched separately"}
            publish("candidate/camera-snapshots.json", cameras)
            source = build_quality_reference(entry["source_case"])
            rebuilt, _ = save(source.object, owner.root / "source")
            receipt["source_equivalence"] = reference_equivalence(source_data, rebuilt)
            if not receipt["source_equivalence"]["equivalent"]:
                raise ValueError("source exact oriented equivalence failed")
            receipt["styles"] = {"source": shading_state(source.object), "candidate": shading_state(obj)}
            receipt["styles"]["comparison"] = style_comparison(receipt["styles"]["source"],receipt["styles"]["candidate"])
            display_camera = unclipped_inspection_camera(camera_inputs["oblique_35_28"], np.vstack((source_data.vertices, data.vertices)))
            publish("display-only-camera.json", display_camera)
            display = {}
            for role, target in (("source", source.object), ("candidate", obj)):
                checkpoint(); folder = owner.root / role
                with silhouette_session(target_objects=[target], resolution=(512,512), engine="BLENDER_WORKBENCH", color_mode="BW",
                        transparent_bg=False, force_material=False, ensure_light_obj=False) as session:
                    _replay_orthographic_camera(session.camera, display_camera)
                    sh = session.scene.display.shading
                    sh.light, sh.color_type, sh.single_color = "STUDIO", "SINGLE", (.65,.65,.65)
                    sh.show_shadows, sh.show_cavity, sh.background_type = True, False, "WORLD"
                    path = folder / "unclipped-oblique_35_28-neutral.png"
                    render_silhouette_frame(session,path)
                    actual = camera_record(session.camera)
                    bounds = inspection_frame_bounds(actual, evaluated_arrays(target).vertices)
                    if not bounds["projection_enclosed"] or not bounds["clipping_enclosed"]:
                        raise ValueError("actual display camera still clips geometry")
                    display[role] = {"actual_camera": actual, "projected_bounds": bounds, "png_sha256": sha(path),
                        "geometry_hash": evaluated_arrays(target).content_hash,
                        "render_settings": {"engine": session.scene.render.engine,
                            "color_mode": session.scene.render.image_settings.color_mode,
                            "resolution": [session.scene.render.resolution_x, session.scene.render.resolution_y],
                            "resolution_percentage": session.scene.render.resolution_percentage,
                            "view_transform": session.scene.view_settings.view_transform,
                            "exposure": session.scene.view_settings.exposure, "gamma": session.scene.view_settings.gamma,
                            "light": sh.light, "color_type": sh.color_type, "single_color": list(sh.single_color),
                            "shadows": sh.show_shadows, "cavity": sh.show_cavity,
                            "studio_light": sh.studio_light, "background_type": sh.background_type},
                        "scope": "fresh human inspection only; excluded from original gates and raw surface"}
                    owner.register_file(path.relative_to(owner.root), "final_output")
                    receipt["completed_native_frames"] += 1
            require_same_frame(display["source"]["actual_camera"], display["candidate"]["actual_camera"])
            receipt["display_only"] = display
            if evaluated_arrays(obj).content_hash != data.content_hash:
                raise ValueError("renders changed new candidate geometry")
            styles_after = {"source": shading_state(source.object), "candidate": shading_state(obj)}
            if any(styles_after[role] != receipt["styles"][role] for role in ("source", "candidate")):
                raise ValueError("display passes changed authored or candidate shading state")
            receipt["shading_state_restored"] = True
            checkpoint()
            raw = raw_surface_observation(source_data, data, count=4096, seed=61007)
            receipt["raw_surface"] = raw
            limits = plan["engineering_limits"]
            receipt["surface_engineering"] = {"limits": limits, "passed": bool(
                raw["symmetric_mean_distance_world"] <= limits["symmetric_mean_distance_world_max"]
                and raw["normal_angle_p95_degrees"] <= limits["normal_angle_p95_degrees_max"]), "artist_limits": None}
            checkpoint()
            boundary = qualify_geometry(data, python=plan["qualifier_python"], timeout_s=15., ownership_root=owner.root / "qualification-children")
            receipt["boundary"] = boundary
            receipt["independent_checks_passed"] = bool(receipt["silhouette"]["all_five_passed"] and edit["passed"]
                and receipt["surface_engineering"]["passed"] and boundary.get("single_solid_qualified") is True
                and boundary.get("geometry_content_hash") == data.content_hash and boundary.get("cache_hit") is False)
            receipt.update(status="measured", elapsed_seconds=time.monotonic()-started)
            publish("results.json", receipt)
            return 0 if receipt["independent_checks_passed"] else 1
        except BaseException as exc:
            receipt.update(status="failed", error=type(exc).__name__ + ": " + str(exc), elapsed_seconds=time.monotonic()-started)
            try:
                # Retain completed partial PNG bytes even if a later pass/camera guard failed.
                # This inventory makes no camera, geometry, lifecycle or gate admission.
                partial = []
                expected_paths = [Path("candidate") / (view + "-" + style + ".png")
                                  for view in CANONICAL_VIEWS for style in ("mask", "neutral")]
                expected_paths += [Path(role) / "unclipped-oblique_35_28-neutral.png"
                                   for role in ("source", "candidate")]
                for relative in expected_paths:
                    path = owner.root / relative
                    if path.is_file():
                        owner.register_file(relative, "diagnostic")
                        partial.append({"path": str(relative), "sha256": sha(path), "bytes": path.stat().st_size})
                receipt["retained_partial_png_bytes"] = {"files": partial,
                    "scope": "bytes only; any absent camera/geometry/pass binding remains unavailable"}
                owner.mark_failed(receipt["error"]); publish("results.json",receipt)
            except BaseException as secondary:
                exc.add_note("secondary receipt failure: " + repr(secondary))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
