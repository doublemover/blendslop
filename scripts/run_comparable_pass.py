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
        views = render_views(spec, ref, resolution=(512, 512))
        obj = build_mesh_object(spec)
        export_object(obj, ref / "ground_truth.obj")
        dump(ref / "manifest.json", {"spec": spec.to_dict(), "blender": bpy.app.version_string,
             "views": views, "python": sys.version, "build_hash": bpy.app.build_hash.decode(),
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
    if args.arm == "candidate":
        cfg.visual_hull.resolution = 96
        cfg.profile_sampling.num_samples = 160
        cfg.mesh_from_profile.radial_segments = 48
        cfg.gaussian_ellipsoid.primitive_count = 32
        cfg.gaussian_ellipsoid.initialization = "kmeans"
    cfg.validate()
    out = root / args.arm / args.case / args.mode
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "config.json", cfg.to_dict())
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
    geometry_start = time.perf_counter()
    geometry = compare_meshes(ref / "ground_truth.obj", mesh, seed=args.seed) if mesh else {"reason": "missing_mesh"}
    if mesh:
        from blender_blocking.e2e.evidence import attach_evaluation_evidence
        payload = attach_evaluation_evidence(payload, geometry_payload={"geometry_true": geometry})
        dump(out / "result.json", payload)
    backend = payload.get("backend_result", {})
    selected = backend.get("selected") or backend.get("selected_result") or backend
    if isinstance(selected, str):
        selected = {"candidate_id": selected}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    row = {"arm": args.arm, "case": args.case, "requested_mode": args.mode,
           "selected_backend": selected.get("backend_name", args.mode), "passed": passed,
           "average_iou": payload.get("average_iou"), "min_view_iou": payload.get("min_view_iou"),
           "views": payload.get("views", {}), "geometry": geometry,
           "reconstruction_and_validation_s": elapsed,
           "geometry_evaluation_s": time.perf_counter() - geometry_start,
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
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            return "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--output", type=Path, default=ROOT / "temp" / "bounded-pass-20261006")
    parser.add_argument("--arm", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--cases", default="box,vase,torus")
    parser.add_argument("--modes", default="profile_loft,visual_hull_voxel,gaussian_ellipsoid_proxy,primitive_fit_refine,differentiable_refine,hybrid_loft_hull,shape_program,ensemble")
    parser.add_argument("--case")
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
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in subprocess.check_output(["git", "ls-files", "blender_blocking", "scripts/run_comparable_pass.py"], cwd=ROOT, text=True).splitlines()
                     if (ROOT / name).is_file() and name.endswith(".py")}
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
        for mode in modes:
            print(f"{args.arm} {case} {mode}", flush=True)
            code = run_child(base + ["--worker", "candidate", "--arm", args.arm, "--mode", mode],
                             args.output / "logs" / f"{args.arm}-{case}-{mode}.log", args.timeout)
            path = args.output / args.arm / case / mode / "comparison.json"
            rows.append(json.loads(path.read_text()) if path.exists() and code == 0 else
                        {"arm": args.arm, "case": case, "requested_mode": mode, "process_status": code})
            dump(args.output / f"{args.arm}.json", {"schema": "comparable_pass_v1", "seed": args.seed,
                "views": ["front", "side", "top"], "render_resolution": [512, 512],
                "geometry_protocol": "bbox_center_longest_extent_v1; no rotation/anisotropic fitting",
                "command": sys.argv, "rows": rows})
    print(f"Completed {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
