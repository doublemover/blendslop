"""Backfill exact-mesh measurements; never rerun optimization during measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]
from scripts.run_comparable_pass import run_child, worker, dump

BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
OUT = ROOT / "temp/bounded-pass-20261006"

def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", default=BLENDER)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--worker", choices=("instrument", "measure", "contract"))
    parser.add_argument("--case")
    parser.add_argument("--mode")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    OUT = args.output.resolve()
    if args.worker:
        from blender_blocking.verify_setup import configure_dependency_paths
        configure_dependency_paths()
        if args.worker == "instrument":
            args.arm = "baseline-instrumented"
            args.output = OUT
            args.seed = 1234
            worker(args)
            return
        from blender_blocking.evaluation.comparable_geometry import compare_meshes, occupancy
        from blender_blocking.e2e.evidence import attach_evaluation_evidence
        from blender_blocking.e2e.ground_truth import _mesh_path_from_payload
        if args.worker == "contract":
            from blender_blocking.synthetic.registry import get_definition
            from blender_blocking.synthetic.blender_builders import build_mesh_object, render_views
            from blender_blocking.evaluation.comparable_geometry import export_evaluated_object
            from blender_blocking.e2e.ground_truth import _synthetic_ground_truth_row
            obj = build_mesh_object(get_definition("box").create(1234))
            path = export_evaluated_object(obj, OUT / "contract" / "same.obj")
            result = compare_meshes(path, path)
            assert result["chamfer_l1"] == 0 and result["fscore_tau"] == 1 and result["volumetric_iou"] == 1
            # Exercise the regular synthetic reference/E2E geometry integration.
            views = render_views(get_definition("box").create(1234), OUT / "contract" / "references", resolution=(128,128))
            row = _synthetic_ground_truth_row(get_definition("box").create(1234), {"mesh_path": str(path)}, reference_mesh_path=views["ground_truth_obj"])
            assert row["available"] and row["geometry_payload"]["geometry_true"]["fscore_tau"] == 1
            import bpy
            dump(OUT / "contract" / "result.json", {"status": "pass", "blender": bpy.app.version_string, "same_mesh": result, "matrix_ground_truth": row})
            print("Exact evaluated mesh, identity surface/volume, synthetic reference integration: PASS")
            return
        measured = []
        for arm in ("baseline", "candidate"):
            rows = json.loads((OUT / f"{arm}.json").read_text())["rows"]
            for row in rows:
                case, mode = row["case"], row["requested_mode"]
                result_path = OUT / arm / case / mode / "result.json"
                if not result_path.exists() or row.get("average_iou") is None:
                    measured.append(dict(row, geometry_validity="unavailable"))
                    continue
                payload = json.loads(result_path.read_text())
                geometry_source = result_path
                if arm == "baseline" and mode in ("profile_loft", "hybrid_loft_hull"):
                    geometry_source = OUT / "baseline-instrumented" / case / mode / "result.json"
                    instrumentation = json.loads(geometry_source.read_text())
                    assert abs(instrumentation["average_iou"] - payload["average_iou"]) < 1e-10, (case,mode,"instrumentation altered silhouette")
                    assert abs(instrumentation["min_view_iou"] - payload["min_view_iou"]) < 1e-10
                    assert all(abs(instrumentation["views"][view]["area_iou"] - metrics["area_iou"]) < 1e-10
                               for view, metrics in payload["views"].items())
                    payload = instrumentation
                mesh = _mesh_path_from_payload(payload)
                if mesh:
                    geometry = compare_meshes(OUT / "references" / case / "ground_truth.obj", mesh)
                    refreshed = attach_evaluation_evidence(payload, geometry_payload={"geometry_true": geometry})
                    # Preserve raw run outputs; refreshed evidence has its own artifact.
                    fresh_path = OUT / "measured" / arm / case / mode / "result.json"
                    dump(fresh_path, refreshed)
                    row = dict(row, geometry=geometry, geometry_validity="measured_exact_render_mesh" if payload.get("mesh_path") else "backend_native_mesh_same_import_axes",
                               geometry_mesh_path=str(mesh), geometry_source_result=str(geometry_source), measured_result_path=str(fresh_path),
                               mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest())
                    if arm == "baseline" and mode in ("profile_loft", "hybrid_loft_hull"):
                        row["geometry_note"] = "Original settings; instrumentation rerun had exactly equal mean silhouette IoU"
                measured.append(row)
                print(f"measure {arm} {case} {mode}", flush=True)
        dump(OUT / "measured.json", {"schema": "comparable_pass_measured_v1", "rows": measured,
             "protocol": "bbox_center_longest_extent_v1; 8192 area-weighted samples; seed1234; Fscore tolerance.02;24^3 occupancy only one closed component",
             "input_manifests": [str(OUT / "references" / case / "manifest.json") for case in ("box","vase","torus")]})
        return
    base = [args.blender, "--background", "--factory-startup", "--threads", "4", "--python-exit-code", "2", "--python", str(Path(__file__).resolve()), "--", "--output", str(OUT)]
    for case in ("box", "vase", "torus"):
        for mode in ("profile_loft", "hybrid_loft_hull"):
            code = run_child(base + ["--worker", "instrument", "--case", case, "--mode", mode], OUT / "logs" / f"instrument-{case}-{mode}.log", 100)
            if code != 0:
                raise RuntimeError(f"instrumentation failed {case}/{mode}: {code}")
            print(f"instrument {case} {mode}: PASS", flush=True)
    for action in ("contract", "measure"):
        code = run_child(base + ["--worker", action], OUT / "logs" / f"{action}.log", 150)
        if code != 0:
            raise RuntimeError(f"{action} failed: {code}")
        print(f"{action}: PASS", flush=True)

if __name__ == "__main__":
    main()
