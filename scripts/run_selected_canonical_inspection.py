#!/usr/bin/env python3
"""Fresh exact source/candidate canonical packet for two selected final meshes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT / "scripts"), str(ROOT)]
import test_runner
from evaluation.canonical_artifacts import CANONICAL_VIEWS, INSPECTION_PASSES, camera_frame_sha256
from evaluation.reference_noise import oriented_surface_identity
from reconstruction.native_geometry import GeometryArrays

SELECTED = {
    "rounded_box": "9f6c8d4db73fdccedf0ac346682c33b626830259989f0f0069f60306938f9b9a",
    "asymmetric_multipart_solid": "b2c937a949368471cb472b2d1cd3aa12f2d0a39871365d257a15154ff4edecb1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_exact(path, expected_hash):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 67108864:
            raise ValueError("exact archive exceeds bounded decompression allowance")
    with np.load(path, allow_pickle=False) as archive:
        data = GeometryArrays.capture(archive["vertices"], archive["faces"])
    if data.content_hash != expected_hash:
        raise ValueError("retained indexed geometry identity differs")
    return data


def reference_equivalence(original, rebuilt):
    a, b = oriented_surface_identity(original), oriented_surface_identity(rebuilt)
    return {"equivalent": a == b, "indexed_reference_hash": original.content_hash,
            "indexed_regenerated_hash": rebuilt.content_hash,
            "reference_oriented_surface": a, "regenerated_oriented_surface": b,
            "scope": "exact binary64 vertex and oriented triangle-coordinate multisets; no topology identity qualification"}


def require_same_frame(a, b):
    import math
    if camera_frame_sha256(a) != camera_frame_sha256(b):
        raise ValueError("actual projection/frame mismatch")
    for record in (a, b):
        clips = [float(record[key]) for key in ("clip_start", "clip_end")]
        if not all(math.isfinite(v) for v in clips) or not 0 < clips[0] < clips[1]:
            raise ValueError("actual clipping unavailable or invalid")
    if any(a[key] != b[key] for key in ("clip_start", "clip_end")):
        raise ValueError("actual clipping mismatch")
    return camera_frame_sha256(a)


def matched_packet(source, candidate, source_cameras, candidate_cameras):
    """Require every actual pass binding; inventory's optional normals also count."""
    pairs = {}
    for label, inventory in (("source", source), ("candidate", candidate)):
        if inventory["geometry"]["identity_status"] != "verified":
            raise ValueError(label + " exact geometry unavailable")
    for view in CANONICAL_VIEWS:
        source_record, candidate_record = source_cameras[view], candidate_cameras[view]
        frame = require_same_frame(source_record, candidate_record)
        for label, inventory, record in (("source", source, source_record), ("candidate", candidate, candidate_record)):
            if (record.get("geometry_hash") != inventory["geometry"]["geometry_hash"] or
                    record.get("geometry_unchanged_after_passes") is not True):
                raise ValueError(label + " camera geometry binding unavailable")
        pairs[view] = {"actual_frame_match": True, "actual_clipping_match": True,
                       "camera_sha256": frame, "passes": {}}
        for name in INSPECTION_PASSES:
            rows = {}
            for label, inventory, record in (("source", source, source_record),
                                             ("candidate", candidate, candidate_record)):
                row = inventory["views"][view]["artifacts"][name]
                binding = record.get("pass_artifacts", {}).get(name, {})
                if (row["status"] != "available" or row["producer_status"] != "completed" or
                        row["geometry_binding"] != "verified_by_producer" or
                        row["geometry_hash"] != inventory["geometry"]["geometry_hash"] or
                        row["camera_sha256"] != frame or
                        binding.get("geometry_hash") != inventory["geometry"]["geometry_hash"] or
                        binding.get("sha256") != row["sha256"] or binding.get("camera_sha256") != frame):
                    raise ValueError(label + " pass binding unavailable: " + view + "/" + name)
                require_same_frame(record, binding["actual_camera"])
                rows[label] = {"path": row["path"], "sha256": row["sha256"],
                               "geometry_hash": row["geometry_hash"], "render_settings": row["render_settings"]}
            require_same_frame(source_record["pass_artifacts"][name]["actual_camera"],
                               candidate_record["pass_artifacts"][name]["actual_camera"])
            if rows["source"]["render_settings"] != rows["candidate"]["render_settings"]:
                raise ValueError("source/candidate pass settings mismatch: " + view + "/" + name)
            pairs[view]["passes"][name] = rows
    return {"status": "complete", "actual_matched_views": pairs,
            "scope": "current recapture only; historical missing bindings remain unchanged",
            "acceptance": "independent; no fitting, silhouette metrics, surface comparisons or qualifiers"}


def validate_plan(plan):
    if (plan.get("protocol") != "selected_canonical_recapture_v1" or
            set(plan.get("cases", {})) != set(SELECTED) or
            plan.get("views") != list(CANONICAL_VIEWS) or plan.get("passes") != list(INSPECTION_PASSES) or
            plan.get("resolution") != [512, 512] or plan.get("native_frames") != 60 or
            plan.get("fits") != 0 or plan.get("raw_comparisons") != 0 or plan.get("qualifier_children") != 0 or
            plan.get("threads") != 2 or plan.get("deadline_seconds") != 120 or
            plan.get("rss_limit_bytes") != 8 * 1024 ** 3):
        raise ValueError("selected canonical frozen scope differs")
    workload = read_json(plan["reference_workload"])
    references = read_json(plan["reference_receipt"])["cases"]
    authored = {row["name"]: row for row in workload["cases"]}
    required_files = {plan["reference_workload"], plan["reference_receipt"], str(Path(__file__).resolve())}
    for name, entry in plan["cases"].items():
        if (entry["source_case"] != authored[name] or
                entry["source_geometry_hash"] != references[name]["geometry_hash"] or
                entry["original_cameras"] != references[name]["reference_cameras"]):
            raise ValueError("authored reference/camera declaration changed")
        required_files.update(entry[key] for key in ("source_npz", "candidate_npz", "candidate_program", "candidate_receipt"))
        required_files.update(row["path"] for row in entry["original_masks"].values())
        if entry["candidate_geometry_hash"] != SELECTED[name] or entry["source_case"]["name"] != name:
            raise ValueError("selected final recipe/geometry declaration differs")
        if set(entry["original_cameras"]) != set(CANONICAL_VIEWS):
            raise ValueError("all five original camera declarations required")
        for view, record in entry["original_cameras"].items():
            camera_frame_sha256(record)
            mask = entry["original_masks"][view]
            if mask["sha256"] != record["png_sha256"] or sha(mask["path"]) != mask["sha256"]:
                raise ValueError("original retained mask bytes differ")
    if not required_files.issubset(plan["input_sha256"]):
        raise ValueError("frozen source/recipe/camera inputs are not all hash-bound")
    for filename, digest in plan["input_sha256"].items():
        if sha(filename) != digest:
            raise ValueError("frozen input/source changed: " + filename)


def _normal_material():
    import bpy
    material = bpy.data.materials.new("SelectedCanonicalWorldShadingNormal")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    normal = nodes.new("ShaderNodeNewGeometry")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = .5
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "ADD"
    offset.inputs[1].default_value = (.5, .5, .5)
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(normal.outputs["Normal"], scale.inputs[0])
    links.new(scale.outputs["Vector"], offset.inputs[0])
    links.new(offset.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _settings(scene, name):
    view = scene.view_settings
    record = {"engine": scene.render.engine, "format": scene.render.image_settings.file_format,
              "color_mode": scene.render.image_settings.color_mode,
              "color_depth": scene.render.image_settings.color_depth,
              "resolution_percentage": scene.render.resolution_percentage,
              "display": {"view_transform": view.view_transform, "look": view.look,
                          "exposure": view.exposure, "gamma": view.gamma},
              "purpose": "human inspection only; no geometric-normal metric"}
    if name == "neutral":
        shading = scene.display.shading
        record.update(light=shading.light, color_type=shading.color_type,
                      single_color=list(shading.single_color), shadows=shading.show_shadows,
                      cavity=shading.show_cavity, background_type=shading.background_type)
    else:
        record.update(material="world shading normal * .5 + .5 emission", samples=scene.eevee.taa_render_samples)
    return record


def render_role(obj, folder, cameras, owner, deadline, receipt, publish):
    import bpy
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import camera_record
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import evaluated_arrays
    from run_surface_quality_check import _replay_orthographic_camera

    before = evaluated_arrays(obj).content_hash
    actual = {}
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    defaults = {"shift_x": 0., "shift_y": 0., "clip_start": .1, "clip_end": 1000.}

    def frame(view):
        if time.monotonic() > deadline:
            raise TimeoutError("selected canonical deadline exceeded")
        owner.reserve_bytes(8388608)
        _replay_orthographic_camera(camera, {**defaults, **cameras[view]})
        bpy.context.view_layer.update()

    def bind(view, name, path, settings):
        record = camera_record(camera, resolution=(512, 512),
            pixel_aspect=(bpy.context.scene.render.pixel_aspect_x, bpy.context.scene.render.pixel_aspect_y))
        if evaluated_arrays(obj).content_hash != before:
            raise ValueError("inspection pass changed exact evaluated geometry")
        if view not in actual:
            actual[view] = {**record, "geometry_hash": before, "render_passes": {},
                            "pass_artifacts": {}, "executed_passes": []}
        require_same_frame(actual[view], record)
        actual[view]["render_passes"][name] = settings
        actual[view]["pass_artifacts"][name] = {"path": path.name, "sha256": sha(path),
            "geometry_hash": before, "camera_sha256": camera_frame_sha256(record), "actual_camera": record}
        actual[view]["executed_passes"].append(name)
        if name == "mask":
            actual[view]["png_sha256"] = sha(path)
        owner.register_file(path.relative_to(owner.root), "final_output")
        receipt["completed_native_frames"] += 1
        publish(folder.relative_to(owner.root) / "camera-snapshots.json", actual)
        publish("results.json", receipt)

    try:
        with controlled_measurement_session(target_objects=[obj], camera=camera, resolution=(512, 512),
                                            samples=64, filter_size=1.5) as session:
            if session.target_objects != [obj]:
                raise ValueError("unexpected measurement mesh descendants")
            for view in CANONICAL_VIEWS:
                frame(view)
                path = folder / (view + "-mask.exr")
                measured = render_controlled_measurement(session, path)
                owner.register_file(path.relative_to(owner.root), "final_output")
                coverage_path = folder / (view + "-coverage.npy")
                np.save(coverage_path, measured.pop("coverage"), allow_pickle=False)
                owner.register_file(coverage_path.relative_to(owner.root), "final_output")
                mask = measured.pop("mask")
                png = folder / (view + "-mask.png")
                Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(png)
                measured.update(coverage_sha256=sha(coverage_path), hard_mask_sha256=sha(png),
                                hard_mask_encoding="black occupied / white empty; actual linear alpha >= .5")
                publish(folder.relative_to(owner.root) / (view + "-measurement.json"), measured)
                bind(view, "mask", png, measured["contract"])
        normal_material = _normal_material()
        originals = list(obj.data.materials)
        try:
            with silhouette_session(target_objects=[obj], camera=camera, resolution=(512, 512),
                    color_mode="BW", transparent_bg=False, engine="BLENDER_WORKBENCH", force_material=False,
                    ensure_light_obj=False) as session:
                if session.target_objects != [obj]:
                    raise ValueError("unexpected preview mesh descendants")
                shading = session.scene.display.shading
                shading.light, shading.color_type = "STUDIO", "SINGLE"
                shading.single_color = (.65, .65, .65)
                shading.show_shadows, shading.show_cavity = True, False
                shading.background_type = "WORLD"
                for name in ("neutral", "normals"):
                    if name == "normals":
                        session.scene.render.engine = "BLENDER_EEVEE"
                        session.scene.render.image_settings.color_mode = "RGB"
                        session.scene.eevee.taa_render_samples = 64
                        obj.data.materials.clear()
                        obj.data.materials.append(normal_material)
                    for view in CANONICAL_VIEWS:
                        frame(view)
                        path = folder / (view + "-" + name + ".png")
                        render_silhouette_frame(session, path)
                        bind(view, name, path, _settings(session.scene, name))
        finally:
            obj.data.materials.clear()
            for material in originals:
                obj.data.materials.append(material)
            bpy.data.materials.remove(normal_material)
    finally:
        bpy.data.objects.remove(camera, do_unlink=True)
    if evaluated_arrays(obj).content_hash != before:
        raise ValueError("role inspection changed geometry")
    for record in actual.values():
        record["geometry_unchanged_after_passes"] = True
    publish(folder.relative_to(owner.root) / "camera-snapshots.json", actual)
    return actual


def main():
    import bpy
    from evaluation.canonical_artifacts import canonical_artifact_inventory
    from primitives.shape_program_compiler import compile_shape_program
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.frozen_family import retained_family_program
    from reconstruction.output_targets import output_mesh_targets
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_frozen_multipart_reconstruction import compile_live_multipart
    from run_quality_coverage_check import save_mesh
    from run_reference_neutral_inspection import shading_state, style_comparison
    from run_surface_quality_check import _write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    plan = read_json(args.plan)
    validate_plan(plan)
    if not args.output.resolve().is_relative_to(ROOT / "temp/tasks"):
        raise ValueError("selected packet output must stay under isolated temp/tasks")
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(), producer="selected_canonical_recapture", max_generated_bytes=268435456,
                     shared_inputs={"plan": str(args.plan.resolve()), "plan_sha256": sha(args.plan)})
    receipt = {"protocol": plan["protocol"], "status": "running", "cases": {}, "completed_native_frames": 0,
               "aggregate_accepted": False, "fits": 0, "raw_comparisons": 0, "qualifier_children": 0}
    with owner:
        def publish(relative, value):
            path = owner.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, value)
            owner.register_file(relative, "final_output")
        publish("frozen-workload.json", plan)
        publish("results.json", receipt)
        for name in SELECTED:
            folder = owner.root / name
            folder.mkdir()
            before_objects = set(bpy.data.objects)
            entry = plan["cases"][name]
            row = {"status": "blocked", "source_geometry_hash": entry["source_geometry_hash"],
                   "candidate_geometry_hash": entry["candidate_geometry_hash"], "artifact_directory": str(folder)}
            receipt["cases"][name] = row
            try:
                source_data = load_exact(entry["source_npz"], entry["source_geometry_hash"])
                candidate_data = load_exact(entry["candidate_npz"], entry["candidate_geometry_hash"])
                source = build_quality_reference(entry["source_case"])
                source_folder, candidate_folder = folder / "source", folder / "candidate"
                rebuilt, _ = save_mesh(source.object, source_folder)
                equivalence = reference_equivalence(source_data, rebuilt)
                row["source_equivalence"] = equivalence
                publish(Path(name) / "reference-equivalence.json", equivalence)
                publish(Path(name) / "source-recipe.json", entry["source_case"])
                if not equivalence["equivalent"]:
                    raise ValueError("authored source exact oriented triangles cannot be reproduced")
                wire = read_json(entry["candidate_program"])
                if wire != entry["candidate_recipe"]:
                    raise ValueError("selected candidate recipe changed")
                if name == "rounded_box":
                    compiled = compile_shape_program(retained_family_program(wire), lathe_segments=96, weighted_normals=False)
                    targets = output_mesh_targets([compiled.root_object])
                    if len(targets) != 1:
                        raise ValueError("selected box needs exactly one mesh descendant")
                    candidate = targets[0]
                    captured = evaluated_arrays(compiled.root_object)
                else:
                    candidate, parts = compile_live_multipart(retained_multipart_program(wire))
                    captured = evaluated_arrays(candidate)
                row["candidate_replay"] = {"expected_indexed_hash": candidate_data.content_hash,
                    "actual_indexed_hash": captured.content_hash, "exact": captured.content_hash == candidate_data.content_hash}
                # Retain a failed replay before refusing its renders.
                save_mesh(candidate, candidate_folder)
                publish(Path(name) / "candidate-program.json", wire)
                if not row["candidate_replay"]["exact"]:
                    raise ValueError("selected candidate indexed recipe replay differs")
                styles = {"source": shading_state(source.object), "candidate": shading_state(candidate)}
                styles["comparison"] = style_comparison(styles["source"], styles["candidate"])
                publish(Path(name) / "shading-style.json", styles)
                source_cameras = render_role(source.object, source_folder, entry["original_cameras"], owner,
                    started + 120, receipt, publish)
                candidate_cameras = render_role(candidate, candidate_folder, entry["original_cameras"], owner,
                    started + 120, receipt, publish)
                if shading_state(source.object) != styles["source"] or shading_state(candidate) != styles["candidate"]:
                    raise ValueError("inspection changed authored or candidate shading state")
                inventories = {}
                for role, data, cameras, peer in (("source", rebuilt, source_cameras, candidate_cameras),
                                                  ("candidate", captured, candidate_cameras, source_cameras)):
                    inventory = canonical_artifact_inventory(folder / role, geometry_hash=data.content_hash,
                        camera_records=cameras, reference_camera_records=peer,
                        pass_states={pass_name: "completed" for pass_name in INSPECTION_PASSES})
                    inventories[role] = inventory
                    publish(Path(name) / role / "canonical-inspection.json", inventory)
                packet = matched_packet(inventories["source"], inventories["candidate"], source_cameras, candidate_cameras)
                packet["legacy_original_declaration"] = {view: {
                    "source_actual_frame_matches_declared": camera_frame_sha256(source_cameras[view]) == camera_frame_sha256(entry["original_cameras"][view]),
                    "original_clipping_status": "available" if all(k in entry["original_cameras"][view] for k in ("clip_start", "clip_end")) else "unavailable",
                    "original_geometry_binding": "legacy unavailable; this fresh packet does not backfill it"} for view in CANONICAL_VIEWS}
                publish(Path(name) / "matched-packet.json", packet)
                row.update(status="complete", packet=packet, styles=styles,
                           source_actual_geometry_hash=rebuilt.content_hash,
                           candidate_actual_geometry_hash=captured.content_hash)
            except Exception as exc:
                row.update(status="blocked", reason=repr(exc))
            finally:
                for path in folder.rglob("*"):
                    if path.is_file():
                        owner.register_file(path.relative_to(owner.root), "final_output")
                # Only this fresh native job's objects are removed; no input file mutation.
                for obj in list(set(bpy.data.objects) - before_objects):
                    bpy.data.objects.remove(obj, do_unlink=True)
                publish("results.json", receipt)
        validate_plan(plan)
        receipt["status"] = "complete" if all(row["status"] == "complete" for row in receipt["cases"].values()) else "blocked"
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["environment"] = {"blender": bpy.app.version_string, "python": sys.version}
        publish("results.json", receipt)
        if receipt["status"] != "complete":
            owner.mark_failed("selected canonical packet has explicitly blocked rows")
    print(json.dumps({"receipt": str(owner.root / "results.json"), "status": receipt["status"],
                      "frames": receipt["completed_native_frames"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
