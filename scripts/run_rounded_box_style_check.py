#!/usr/bin/env python3
"""Five exact-frame rounded-box neutral previews after explicit source style."""
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
from run_family_source_edit_check import _compile_exact, _cleanup, _publish_receipt
from run_reference_neutral_inspection import shading_state, style_comparison
from run_quality_coverage_check import save_mesh


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import bpy
    import numpy as np
    from evaluation.canonical_artifacts import CANONICAL_VIEWS, camera_record, camera_frame_sha256, canonical_artifact_inventory
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from utils.run_ownership import OwnedRun
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--source-neutral-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-sharp", choices=("true", "false"), default="true")
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("fresh output must stay under isolated repository temp/tasks")
    prior = json.loads(args.candidate_receipt.read_text())["cases"]["rounded_box"]
    reference = json.loads(args.source_neutral_receipt.read_text())["cases"]["rounded_box"]
    source_folder, candidate_folder = Path(reference["artifact_directory"]), Path(prior["artifact_directory"])
    source_workload = json.loads((source_folder.parent/"frozen-workload.json").read_text())
    originals = source_workload["inputs"]["rounded_box"]["original_cameras"]
    source_cameras = json.loads((source_folder/"camera-snapshots.json").read_text())
    with np.load(candidate_folder/"evaluated-exact.npz", allow_pickle=False) as archive:
        captured = GeometryArrays.capture(archive["vertices"], archive["faces"])
    if captured.content_hash != prior["geometry_hash"] or _sha(candidate_folder/"evaluated-exact.npz") != prior["npz_sha256"]:
        raise ValueError("retained candidate geometry changed")
    original = json.loads((candidate_folder/"program.json").read_text())
    if original != prior["program"]:
        raise ValueError("retained candidate recipe changed")
    effective = deepcopy(original)
    effective["root_nodes"][0]["parameters"]["bevel_segments"] = 8
    styled = deepcopy(effective)
    styled["root_nodes"][0]["parameters"]["weighted_normals"] = True
    styled["root_nodes"][0]["parameters"]["weighted_normals_keep_sharp"] = args.keep_sharp == "true"
    for view in CANONICAL_VIEWS:
        pair = reference["paired_neutral_views"][view]["source_neutral"]
        if _sha(pair["path"]) != pair["sha256"]:
            raise ValueError("retained source neutral changed")
    settings = source_workload["render_settings"]
    frozen = {"protocol": "rounded_box_authored_style_v1", "original_recipe": original,
        "effective_recipe": effective, "styled_recipe": styled,
        "candidate_receipt": {"path": str(args.candidate_receipt.resolve()), "sha256": _sha(args.candidate_receipt)},
        "source_neutral_receipt": {"path": str(args.source_neutral_receipt.resolve()), "sha256": _sha(args.source_neutral_receipt)},
        "required_indexed_geometry_hash": captured.content_hash, "source_indexed_geometry_hash": reference["geometry_hash"],
        "original_cameras": originals, "source_actual_cameras": source_cameras,
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in
            (Path(__file__), ROOT/"blender_blocking/primitives/shape_program_compiler.py",
             ROOT/"scripts/run_family_source_edit_check.py", ROOT/"scripts/run_reference_neutral_inspection.py",
             ROOT/"scripts/run_surface_quality_check.py", ROOT/"blender_blocking/integration/blender_ops/silhouette_render.py")},
        "render_settings": settings, "neutral_frames": 5, "mask_frames": 0, "normal_frames": 0,
        "fits": 0, "raw_comparisons": 0, "qualification_children": 0,
        "deadline_seconds": 60, "memory_limit_bytes": 8*1024**3, "threads": 2,
        "scope": "optional authored weighted-normal style; unchanged candidate geometry and source frames",
        "environment": {"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__}}
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(), producer="rounded_box_authored_style", max_generated_bytes=16777216,
                     shared_inputs={"candidate_receipt": str(args.candidate_receipt.resolve()), "source_neutral_receipt": str(args.source_neutral_receipt.resolve())})
    with owner:
        _write(owner.root/"frozen-workload.json", frozen)
        owner.register_file("frozen-workload.json", "diagnostic")
        receipt = {"protocol": frozen["protocol"], "status": "running", "aggregate_accepted": False}
        _publish_receipt(owner, receipt)
        compiled = None
        try:
            compiled, source, baseline = _compile_exact(effective, captured)
            baseline_style = shading_state(source)
            _cleanup(compiled)
            compiled = None
            compiled, source, changed = _compile_exact(styled, captured)
            changed_style = shading_state(source)
            if baseline.content_hash != changed.content_hash or not changed_style["evaluated_mesh"]["has_custom_normals"]:
                raise ValueError("weighted-normal style changed geometry or lacks native custom normals")
            _, hashes = save_mesh(source, owner.root)
            _write(owner.root/"program.json", styled)
            cameras, pairs = {}, {}
            with silhouette_session(target_objects=[source], resolution=(512,512), color_mode="BW",
                    transparent_bg=False, engine="BLENDER_EEVEE", background_color=(1,1,1,1), silhouette_color=(0,0,0,1)) as session:
                session.scene.render.engine = "BLENDER_WORKBENCH"
                shading = session.scene.display.shading
                shading.light, shading.color_type = "STUDIO", "SINGLE"
                shading.single_color = (.65,.65,.65)
                shading.show_shadows, shading.show_cavity = True, False
                shading.background_type = "WORLD"
                for view in CANONICAL_VIEWS:
                    if time.monotonic()-started > 60:
                        raise TimeoutError("bounded style deadline exceeded")
                    _replay_orthographic_camera(session.camera, originals[view])
                    bpy.context.view_layer.update()
                    record = camera_record(session.camera, resolution=(512,512),
                        pixel_aspect=(session.scene.render.pixel_aspect_x, session.scene.render.pixel_aspect_y))
                    digest = camera_frame_sha256(record)
                    if digest != camera_frame_sha256(source_cameras[view]):
                        raise ValueError("styled candidate actual frame differs from source: "+view)
                    if any(record[key] != source_cameras[view][key] for key in ("clip_start","clip_end")):
                        raise ValueError("styled candidate clipping differs from source")
                    path = owner.root/(view+"-neutral.png")
                    render_silhouette_frame(session,path)
                    after_record = camera_record(session.camera, resolution=(512,512),
                        pixel_aspect=(session.scene.render.pixel_aspect_x, session.scene.render.pixel_aspect_y))
                    if camera_frame_sha256(after_record) != digest:
                        raise ValueError("style render changed camera frame")
                    record.update(geometry_hash=changed.content_hash, render_passes={"neutral": settings},
                        executed_passes=["neutral"], pass_artifacts={"neutral": {"path": path.name, "sha256": _sha(path),
                        "geometry_hash": changed.content_hash, "camera_sha256": digest}},
                        display_settings={"view_transform": session.scene.view_settings.view_transform,
                                          "look": session.scene.view_settings.look, "exposure": session.scene.view_settings.exposure,
                                          "gamma": session.scene.view_settings.gamma, "preview_only": True})
                    cameras[view] = record
                    pairs[view] = {"source_neutral": reference["paired_neutral_views"][view]["source_neutral"],
                        "previous_candidate_neutral": reference["paired_neutral_views"][view]["candidate_neutral"],
                        "styled_candidate_neutral": {"path": str(path), "sha256": _sha(path), "geometry_hash": changed.content_hash},
                        "actual_frame_match": True, "actual_clipping_match": True, "camera_sha256": digest}
            if evaluated_arrays(source).content_hash != captured.content_hash:
                raise ValueError("styled previews changed saved candidate geometry")
            for record in cameras.values():
                record["geometry_unchanged_after_passes"] = True
            _write(owner.root/"camera-snapshots.json", cameras)
            inventory = canonical_artifact_inventory(owner.root, geometry_hash=changed.content_hash,
                camera_records=cameras, reference_camera_records=originals,
                pass_states={"mask": "unrun", "neutral": "completed", "normals": "unrun"})
            _write(owner.root/"canonical-inspection.json", inventory)
            receipt.update(status="styled_neutrals_retained", **hashes,
                baseline_geometry_hash=baseline.content_hash, styled_geometry_hash=changed.content_hash,
                geometry_unchanged=True, baseline_style=baseline_style, styled_style=changed_style,
                source_style=reference["shading_style"]["authored_source"],
                style_comparison=style_comparison(reference["shading_style"]["authored_source"],changed_style),
                paired_neutral_views=pairs, inspection_artifacts=inventory,
                canonical_passes_complete=False, surface_acceptance="unqualified; no metric recomputation or borrowed limits")
        finally:
            if compiled is not None:
                _cleanup(compiled)
        receipt.update(elapsed_seconds=time.monotonic()-started, run_root=str(owner.root))
        _publish_receipt(owner,receipt)
        for path in owner.root.iterdir():
            if path.is_file() and path.name not in {"run-ownership.json","run-lease.json","results.json","frozen-workload.json"}:
                owner.register_file(path.name,"diagnostic" if path.suffix==".png" else "final_output")
        print("ROUNDED_STYLE_RESULT="+str(owner.root/"results.json"),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
