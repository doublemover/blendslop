"""Summarize retained Blender-version runs without rerunning reconstruction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_rows = [r for r in json.loads((args.old / "measured.json").read_text())["rows"] if r["arm"] == "candidate"]
    new_rows = json.loads((args.new / "candidate.json").read_text())["rows"]
    assert len(old_rows) == len(new_rows) == 24, "Expected a complete finite 3-shape x 8-mode run"
    old = {(r["case"], r["requested_mode"]): r for r in old_rows}
    references = []
    from PIL import Image
    import numpy as np
    for case in ("box", "vase", "torus"):
        a, b = args.old / "references" / case, args.new / "references" / case
        reference = {"case": case, "ground_truth_file_equal": digest(a / "ground_truth.obj") == digest(b / "ground_truth.obj"), "views": {}}
        for view in ("front", "side", "top"):
            left, right = a / f"{view}.png", b / f"{view}.png"
            pa, pb = [np.asarray(Image.open(path).convert("RGBA")) for path in (left, right)]
            reference["views"][view] = {"file_sha256_old": digest(left), "file_sha256_new": digest(right),
                "pixels_equal": bool(np.array_equal(pa, pb)), "pixel_sha256_old": hashlib.sha256(pa.tobytes()).hexdigest(),
                "pixel_sha256_new": hashlib.sha256(pb.tobytes()).hexdigest(), "changed_pixels": int(np.any(pa != pb, axis=2).sum())}
        references.append(reference)
    rows = []
    for new in new_rows:
        prior = old[(new["case"], new["requested_mode"])]
        result_path = Path(new["result_path"]) if new.get("result_path") else None
        payload = json.loads(result_path.read_text()) if result_path and result_path.exists() else {}
        backend = payload.get("backend_result", {})
        selected = backend.get("selected") or backend.get("selected_result") or backend
        selected = selected if isinstance(selected, dict) else {}
        row = {"case": new["case"], "requested_mode": new["requested_mode"],
               "selected_backend_52": new.get("selected_backend"), "process_status": new.get("process_status", 0),
               "backend_status_52": selected.get("status"), "backend_errors_52": selected.get("errors", []),
               "backend_warnings_52": selected.get("warnings", []),
               "render_gate_pass_52": new.get("passed"), "config_equal": prior.get("config_sha256") == new.get("config_sha256"),
               "raw_result_52": new.get("result_path"), "raw_result_50": prior.get("result_path"),
               "geometry_result_50": prior.get("measured_result_path")}
        for key in ("average_iou", "min_view_iou", "reconstruction_and_validation_s", "geometry_evaluation_s"):
            row[f"{key}_50"], row[f"{key}_52"] = prior.get(key), new.get(key)
            row[f"{key}_delta"] = new[key] - prior[key] if new.get(key) is not None and prior.get(key) is not None else None
        for key in ("chamfer_l1", "chamfer_l2", "fscore_tau", "volumetric_iou"):
            row[f"{key}_50"], row[f"{key}_52"] = prior.get("geometry", {}).get(key), new.get("geometry", {}).get(key)
            row[f"{key}_delta"] = row[f"{key}_52"] - row[f"{key}_50"] if row[f"{key}_52"] is not None and row[f"{key}_50"] is not None else None
        row["geometry_warnings_52"] = new.get("geometry", {}).get("warnings", [])
        rows.append(row)
    summaries = []
    for mode in dict.fromkeys(row["requested_mode"] for row in rows):
        group = [r for r in rows if r["requested_mode"] == mode]
        summary = {"requested_mode": mode, "rows": len(group), "process_ok_52": sum(r["process_status"] == 0 for r in group),
                   "render_gate_passes_52": sum(r["render_gate_pass_52"] is True for r in group)}
        for key in ("average_iou_50", "average_iou_52", "min_view_iou_52", "chamfer_l1_52", "fscore_tau_52", "volumetric_iou_52", "reconstruction_and_validation_s_50", "reconstruction_and_validation_s_52"):
            values = [r[key] for r in group if r.get(key) is not None]
            summary[f"mean_{key}"] = statistics.mean(values) if values else None
            summary[f"available_{key}"] = len(values)
        summaries.append(summary)
    quick = (args.new / "quick.log").read_text(encoding="utf-8", errors="replace")
    quick_counts = re.search(r"Results: (\d+) passed, (\d+) failed, (\d+) skipped", quick)
    assert quick_counts, "Quick suite has no completed summary"
    preflight = json.loads((args.new / "preflight.json").read_text())
    exports = json.loads((args.new / "exports/result.json").read_text())
    export_before = json.loads((args.new / "exports-before-fix/result.json").read_text())
    export_additional = {name: json.loads((args.new / name / "result.json").read_text())
                         for name in ("exports-unit1", "exports-blender50")}
    contract = json.loads((args.new / "contract/result.json").read_text())
    source = json.loads((args.new / "candidate-software.json").read_text())
    prior_source = json.loads((args.old / "final-source.json").read_text())
    same_keys = source["source_sha256"].keys() & prior_source["source_sha256"].keys()
    changed = sorted(k for k in same_keys if source["source_sha256"][k] != prior_source["source_sha256"][k])
    metrics = ("average_iou", "min_view_iou", "chamfer_l1", "fscore_tau")
    exact = sum(all(r.get(f"{m}_delta") is not None and abs(r[f"{m}_delta"]) < 1e-12 for m in metrics) for r in rows)
    report = {"schema": "blender_version_compatibility_v1", "date": "2026-10-06", "scope": "One candidate repeat; not another optimization pass or a new within-5.2 baseline comparison",
              "preflight": preflight, "quick_suite": dict(zip(("passed", "failed", "skipped"), map(int, quick_counts.groups()))),
              "contract": contract, "exports": exports, "exports_before_fix": export_before,
              "export_additional_checks": export_additional, "references": references, "rows": rows, "techniques": summaries,
              "source": {"runtime_run_revision": source["repo_revision"], "working_tree_at_run": source["working_tree"],
                         "prior_final_revision": prior_source["source_commit"], "prior_run_provenance_notes": prior_source.get("notes", []),
                         "changed_common_source_files": changed,
                         "after_matrix_repair": "GLB QA adapter now applies modifiers and encodes scene units in metres with source-transform restoration. The matrix did not enable export QA; its retained metrics precede this delivery-only repair.",
                         "protocol_sha256": source["protocol_sha256"], "exact_metric_rows": exact},
              "limits": ["Python and native dependency versions differ; wall-budget methods can take different iteration counts.",
                         "Configured optimizer budgets do not bound setup or total process time; each child has a separate 100-second cap.",
                         "Single timings exclude process startup and geometry evaluation; shared Windows host, no throughput claim.",
                         "8192 area-weighted samples, seed1234, bbox-center/uniform-longest-extent normalization, F-score tolerance .02.",
                         "Volume IoU is 24^3 parity only for one closed component; overlapping parts and self-intersection not certified.",
                         "Open3D absent: optional Poisson path remains untested. No full procedural, GPU/OpenVDB or real dataset certification."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blender52-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, values in (("blender-version-comparison.csv", rows), ("blender52-techniques.csv", summaries)):
        with (args.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps({"rows": len(rows), "exact_metric_rows": exact, "process_ok": sum(r["process_status"] == 0 for r in rows), "render_gate_passes": sum(r["render_gate_pass_52"] is True for r in rows), "techniques": summaries}, indent=2))


if __name__ == "__main__":
    main()
