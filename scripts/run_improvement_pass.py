"""Finite, isolated baseline/candidate matrix; no installations or network use."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def export_object(obj, path):
    import bpy
    from blender_blocking.reconstruction.mesh_io import write_obj
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = [tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        faces = [tuple(t.vertices) for t in mesh.loop_triangles]
        write_obj(path, {"vertices": vertices, "faces": faces})
    finally:
        evaluated.to_mesh_clear()


def novel_render(obj, output, center, scale):
    from blender_blocking.integration.blender_ops.silhouette_render import silhouette_session, set_camera_orbit, render_silhouette_frame
    import math
    path = output / "orbit45.png"
    output.mkdir(parents=True, exist_ok=True)
    with silhouette_session(target_objects=[obj],resolution=(512,512),color_mode="RGBA",transparent_bg=True,
                            force_material=True,engine="BLENDER_EEVEE") as session:
        set_camera_orbit(session.camera,center,scale*2.,math.radians(-45.),scale)
        render_silhouette_frame(session,path)
    return str(path)


def apply_config_overlay(config, overlay):
    """Reject unknown typed settings instead of silently ignoring an experiment."""
    from dataclasses import fields
    from blender_blocking.config_models.backends import CandidateConfig
    for section, values in overlay.items():
        target = getattr(config, section, None)
        if target is None or not isinstance(values, dict):
            raise ValueError('unknown config section: '+section)
        allowed = {f.name for f in fields(target)}
        unknown = set(values)-allowed
        if unknown:
            raise ValueError(f'unknown {section} settings: {sorted(unknown)}')
        for key, value in values.items():
            if section == 'ensemble' and key == 'candidates':
                value = tuple(CandidateConfig(**row) for row in value)
            setattr(target, key, value)
    config.validate()
    return config


def measure_saved_views(payload, reference_manifest):
    """Measure available observed views while retaining primary failure evidence."""
    from PIL import Image
    import numpy as np
    from blender_blocking.evaluation.silhouette_eval import evaluate_silhouette_pair, SilhouetteGateConfig
    references = reference_manifest.get("views", {})
    required = tuple(reference_manifest.get("required_views", ("front", "side", "top")))
    renders = payload.get("rendered_paths") or {}
    measured, unavailable = {}, {}
    for view in dict.fromkeys((*required, *references)):
        reference, rendered = references.get(view), renders.get(view)
        if not reference or not Path(reference).is_file():
            unavailable[view] = {"required": view in required, "reason": "missing_reference"}
            continue
        if not rendered or not Path(rendered).is_file():
            unavailable[view] = {"required": view in required, "reason": "missing_render"}
            continue
        try:
            ref = np.asarray(Image.open(reference).convert("RGBA"))
            pred = np.asarray(Image.open(rendered).convert("RGBA"))
            measured[view] = evaluate_silhouette_pair(ref[:, :, 3] > 127, pred[:, :, 3] > 127,
                view=view, config=SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.1,
                                                       max_signed_distance_loss=.1))
        except (OSError, ValueError) as exc:
            unavailable[view] = {"required": view in required, "reason": "invalid_view_artifact", "error": str(exc)}
    complete = all(view in measured for view in required) and bool(required)
    values = [measured[view]["area_iou"] for view in required if view in measured]
    return {"views": measured, "unavailable_views": unavailable,
            "average_iou": float(np.mean(values)) if complete else None,
            "min_view_iou": float(min(values)) if complete else None,
            "passed": complete and all(measured[v]["passed"] for v in required),
            "metrics_available": complete}


def worker(args):
    import bpy
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    from blender_blocking.config import BlockingConfig
    from blender_blocking.synthetic.registry import get_definition
    from blender_blocking.synthetic.blender_builders import render_views, build_mesh_object
    root = args.output.resolve()
    ref = root / "references" / args.case
    if args.worker == "reference":
        spec = get_definition(args.case).create(args.seed)
        from blender_blocking.integration.blender_ops.camera_framing import compute_bounds_world
        from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views_detailed
        from blender_blocking.config import RenderConfig
        obj = build_mesh_object(spec)
        bounds_min, bounds_max = compute_bounds_world([obj])
        center = (bounds_min+bounds_max)*.5
        scale = max(bounds_max-bounds_min)*1.2
        calibration = {}
        for view, axes in {"front":(0,2),"side":(1,2),"top":(0,1)}.items():
            # Held-out crop/scale changes are known camera metadata, never hidden solver guesses.
            factor = {"front":1., "side":1.12, "top":1.08}[view] if args.seed == 77 else 1.
            offset = {"front":(.035,-.02),"side":(-.03,.025),"top":(.02,.035)}[view] if args.seed == 77 else (0.,0.)
            u, v = center[axes[0]]+offset[0]*scale, center[axes[1]]+offset[1]*scale
            half = scale*factor*.5
            calibration[view] = {"projection":"orthographic", "world_bounds":[u-half,u+half,v-half,v+half],
                "axes":list(axes),"world_units":"metres", "orientation":"canonical_positive_axes", "source":"recorded_capture_camera"}
        rc = RenderConfig(resolution=(512,512),view_calibration=calibration,force_material=True)
        rendered = render_orthogonal_views_detailed(str(ref),target_objects=[obj],render_config=rc)
        views = rendered.paths
        export_object(obj, ref / "ground_truth.obj")
        novel = novel_render(obj, ref, center, scale)
        dump(ref / "manifest.json", {"spec": spec.to_dict(), "blender": bpy.app.version_string,
             "views": views, "calibration":calibration, "render_metadata":rendered.to_dict(),
             "novel":novel, "novel_center":list(center),"novel_scale":scale,
             "python": sys.version, "build_hash": bpy.app.build_hash.decode(),
             "hashes": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in views.items()},
             "ground_truth_sha256": hashlib.sha256((ref / "ground_truth.obj").read_bytes()).hexdigest()})
        return
    from blender_blocking.e2e.validator import test_with_custom_images
    from blender_blocking.e2e.ground_truth import _mesh_path_from_payload
    from blender_blocking.evaluation.comparable_geometry import compare_meshes
    cfg = BlockingConfig()
    cfg.reconstruction.reconstruction_mode = args.mode
    # Fixed fidelity and budgets for both arms; differences below are one declared round.
    cfg.render_silhouette.resolution = (512, 512)
    cfg.primitive_fit.max_runtime_s = 8.
    cfg.differentiable_render.max_runtime_s = 20.
    cfg.visual_hull.resolution = 96
    cfg.profile_sampling.num_samples = 160
    cfg.mesh_from_profile.radial_segments = 48
    cfg.gaussian_ellipsoid.primitive_count = 32
    cfg.gaussian_ellipsoid.initialization = "kmeans"
    reference_manifest = json.loads((ref / "manifest.json").read_text())
    # Copied campaign inputs are authoritative; stale absolute producer paths
    # must not redirect rendering or novel-view evaluation to mutable originals.
    reference_manifest['views'] = {view: str(ref/(view+'.png')) for view in ('front','side','top')}
    reference_manifest['novel'] = str(ref/'orbit45.png')
    cfg.reconstruction.view_calibration = reference_manifest["calibration"]
    if args.arm != "camera_control":
        cfg.silhouette_extract_ref.fill_holes = False
        cfg.silhouette_extract_render.fill_holes = False
        cfg.silhouette_extract_ref.largest_component_only = False
        cfg.silhouette_extract_render.largest_component_only = False
    if args.arm in {"improved", "final"}:
        if args.arm == "improved":
            cfg.gaussian_ellipsoid.cluster_sigma = 2.0
            cfg.gaussian_ellipsoid.negative_space_seed_guard = True
            cfg.gaussian_ellipsoid.kmeans_seed = "farthest"
        cfg.primitive_fit.kmeans_seed = "farthest"
        cfg.primitive_fit.silhouette_objective = "mesh_union"
        cfg.ensemble.evidence_routing = True
        cfg.ensemble.total_timeout_s = 45.
    if args.arm in {"poisson", "poisson_closed"}:
        cfg.visual_hull.postprocess = "screened_poisson"
        cfg.visual_hull.postprocess_required = True
        cfg.visual_hull.external_open3d_python = str(ROOT / ".venv312" / "Scripts" / "python.exe")
        cfg.visual_hull.poisson_depth = 6
        cfg.visual_hull.poisson_density_quantile = 0.01 if args.arm == "poisson" else None
        cfg.visual_hull.poisson_crop_to_input_bounds = args.arm == "poisson"
        cfg.visual_hull.poisson_timeout_s = 60.
    overlay_path = getattr(args, 'config_overlay', None)
    overlay = json.loads(Path(overlay_path).read_text(encoding='utf-8-sig')) if overlay_path else {}
    apply_config_overlay(cfg, overlay)
    experiment = getattr(args, 'experiment', None) or args.arm
    if not experiment.replace('_', '').replace('-', '').isalnum():
        raise ValueError('experiment ID must be a simple name')
    out = root / experiment / args.case / args.mode
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "config.json", cfg.to_dict())
    from blender_blocking.main_integration import BlockingWorkflow
    configured_workflow = BlockingWorkflow(config=cfg)
    expanded = {name: configured_workflow._config_for_backend(name) for name in
                ('profile_loft', 'visual_hull_voxel', 'gaussian_ellipsoid_proxy', 'primitive_fit_refine',
                 'differentiable_refine', 'hybrid_loft_hull', 'shape_program')}
    dump(out / 'dispatched-options.json', {'experiment': experiment, 'typed_overlay': overlay,
                                          'backend_configs': expanded})
    start = time.perf_counter()
    passed = test_with_custom_images(str(ref / "front.png"), str(ref / "side.png"), str(ref / "top.png"),
             workflow_config=cfg, render_config=cfg.render_silhouette, validation_mode="render-iou",
             iou_threshold=.7, boundary_iou_threshold=.1, signed_distance_loss_threshold=.1,
             num_slices=cfg.reconstruction.num_slices, result_json=out / "result.json",
             render_output_dir=out / "renders", artifact_root=out / "artifacts",
             run_id=f"bounded-{args.arm}-{args.case}-{args.mode}", progress=False)
    elapsed = time.perf_counter() - start
    payload = json.loads((out / "result.json").read_text())
    mesh = _mesh_path_from_payload(payload)
    if mesh and not Path(mesh).is_file():
        mesh = None
    novel_metrics = None
    novel_start = time.perf_counter()
    if mesh:
        from blender_blocking.e2e.rendering import _import_obj_for_render
        from PIL import Image
        import numpy as np
        from blender_blocking.evaluation.silhouette_eval import evaluate_silhouette_pair, SilhouetteGateConfig
        from mathutils import Vector
        obj = _import_obj_for_render(Path(mesh))
        novel_path = novel_render(obj, out / "novel", Vector(reference_manifest["novel_center"]),reference_manifest["novel_scale"])
        def mask(path):
            pixels = np.asarray(Image.open(path).convert("RGBA"))
            return pixels[:,:,3] > 127
        novel_metrics = evaluate_silhouette_pair(mask(reference_manifest["novel"]),mask(novel_path),view="orbit_45", config=SilhouetteGateConfig(min_area_iou=.7))
    novel_evaluation_s = time.perf_counter() - novel_start
    from blender_blocking.evaluation.solid_validity import solid_validity_report
    from blender_blocking.evaluation.comparable_geometry import read_obj
    solid = solid_validity_report(*read_obj(mesh)) if mesh else None
    geometry_start = time.perf_counter()
    geometry = compare_meshes(ref / "ground_truth.obj", mesh, seed=args.seed) if mesh else {"reason": "missing_mesh"}
    if mesh:
        from blender_blocking.e2e.evidence import attach_evaluation_evidence
        payload = attach_evaluation_evidence(payload, geometry_payload={"geometry_true": geometry})
        dump(out / "result.json", payload)
    geometry_evaluation_s = time.perf_counter() - geometry_start
    saved_views_start = time.perf_counter()
    measured = measure_saved_views(payload, reference_manifest)
    saved_view_evaluation_s = time.perf_counter() - saved_views_start
    measured_views = measured["views"]
    reported = {"average_iou": payload.get("average_iou"), "views": payload.get("views"), "passed": passed}
    mean_iou, min_iou = measured["average_iou"], measured["min_view_iou"]
    passed = bool(mesh) and measured["passed"]
    backend = payload.get("backend_result", {})
    selected = backend.get("selected") or backend.get("selected_result") or backend
    if isinstance(selected, str):
        selected = {"candidate_id": selected}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    row = {"input_hashes": reference_manifest["hashes"],
           "ground_truth_sha256": reference_manifest["ground_truth_sha256"],
           "seed": args.seed, "calibration": reference_manifest["calibration"],
           "arm": args.arm, "campaign_variant": experiment, "case": args.case, "requested_mode": args.mode,
           "selected_backend": selected.get("backend_name", args.mode), "selected_candidate_id": selected.get("candidate_id"), "passed": passed,
           "average_iou": mean_iou, "min_view_iou": min_iou, "workflow_reported_metrics":reported,
           "views": measured_views, "unavailable_views": measured["unavailable_views"],
           "candidate_status": "evaluated" if mesh and measured["metrics_available"] else "failed",
           "failure_code": payload.get("failure_code") or (None if mesh and measured["metrics_available"] else
                           "missing_renderable_mesh" if not mesh else "missing_required_render"),
           "primary_failure": {**{key: payload.get(key) for key in ("failure_code", "error", "errors")},
                "backend_errors": backend.get("errors", []),
                "candidate_failures": [{key: c.get(key) for key in ("candidate_id", "status", "errors", "warnings")}
                    for c in backend.get("candidates", []) if c.get("status") != "success"]},
           "geometry": geometry, "novel_metrics":novel_metrics, "solid_structure":solid,
           "reconstruction_and_validation_s": elapsed,
           "novel_view_evaluation_s": novel_evaluation_s,
           "geometry_evaluation_s": geometry_evaluation_s, "saved_view_evaluation_s": saved_view_evaluation_s,
           "backend_metrics": selected.get("metric_result", {}), "cost_report": payload.get("cost_report", {}),
           "result_path": str(out / "result.json"), "blender": bpy.app.version_string,
           "python": sys.version, "build_hash": bpy.app.build_hash.decode(),
           "render_engine": payload.get("render_engine"),
           "repo_revision": revision, "config_sha256": hashlib.sha256((out / "config.json").read_bytes()).hexdigest(),
           "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "mesh_sha256": hashlib.sha256(Path(mesh).read_bytes()).hexdigest() if mesh else None,
           "evaluation_status": payload.get("evaluation_bundle", {}).get("status")}
    dump(out / "comparison.json", row)


def run_child(command, log, timeout):
    """Only the child started here is terminated; no global Blender process cleanup."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with log.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), cwd=ROOT)
        try:
            return child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            else:
                child.kill()
            child.wait()
            return "timeout_or_interrupted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--output", type=Path, default=ROOT / "temp" / "improvement-phase-20261006")
    parser.add_argument("--arm", choices=("camera_control", "mask_control", "improved", "final", "poisson", "poisson_closed"), default="camera_control")
    parser.add_argument("--cases", default="box,vase,torus")
    parser.add_argument("--modes", default="profile_loft,visual_hull_voxel,gaussian_ellipsoid_proxy,primitive_fit_refine,differentiable_refine,hybrid_loft_hull,shape_program,ensemble")
    parser.add_argument("--case")
    parser.add_argument("--experiment")
    parser.add_argument("--config-overlay", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=100)
    parser.add_argument("--worker", choices=("reference", "candidate"))
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(raw)
    if args.worker:
        worker(args)
        return
    rows = []
    source_files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "blender_blocking", "scripts"],
        cwd=ROOT, text=True).splitlines()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in source_files if (ROOT / name).is_file() and name.endswith(".py")}
    dump(args.output / f"{args.arm}-software.json", {"repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
         "source_sha256": source_hashes, "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    cases, modes = args.cases.split(","), args.modes.split(",")
    args.output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        base = [args.blender, "--background", "--factory-startup", "--disable-autoexec", "--threads", "4", "--python-exit-code", "2",
                "--python", str(Path(__file__).resolve()), "--", "--output", str(args.output.resolve()),
                "--case", case, "--seed", str(args.seed)]
        if not (args.output / "references" / case / "manifest.json").exists():
            code = run_child(base + ["--worker", "reference"], args.output / "logs" / f"reference-{case}.log", args.timeout)
            if code != 0:
                raise RuntimeError(f"reference {case} failed: {code}; inspect log")
        if args.experiment:
            base += ['--experiment', args.experiment]
        if args.config_overlay:
            base += ['--config-overlay', str(args.config_overlay.resolve())]
        for mode in modes:
            print(f"{args.arm} {case} {mode}", flush=True)
            code = run_child(base + ["--worker", "candidate", "--arm", args.arm, "--mode", mode],
                             args.output / "logs" / f"{args.arm}-{case}-{mode}.log", args.timeout)
            path = args.output / (args.experiment or args.arm) / case / mode / "comparison.json"
            rows.append(json.loads(path.read_text()) if path.exists() and code == 0 else
                        {"arm": args.arm, "case": case, "requested_mode": mode, "process_status": code})
            dump(args.output / f"{args.experiment or args.arm}.json", {"schema": "calibrated_improvement_pass_v1", "seed": args.seed,
                "views": ["front", "side", "top"], "render_resolution": [512, 512],
                "geometry_protocol": "bbox_center_longest_extent_v1; no rotation/anisotropic fitting",
                "command": sys.argv, "rows": rows})
    print(f"Completed {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
