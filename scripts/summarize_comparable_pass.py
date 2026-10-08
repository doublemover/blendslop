"""Small shareable tables; detailed renders/meshes remain under ignored temp."""
import argparse
import csv
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[1]

def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "temp/bounded-pass-20261006")
    args = parser.parse_args()
    output = args.output
    raw = json.loads((output / "measured.json").read_text())["rows"]
    table = []
    for row in raw:
        geometry = row.get("geometry", {})
        views = row.get("views", {})
        table.append({"arm": row["arm"], "case": row["case"], "mode": row["requested_mode"],
            "selected_backend": row.get("selected_backend"), "render_gate_pass": row.get("passed"),
            "mean_render_iou": row.get("average_iou"), "min_render_iou": row.get("min_view_iou"),
            "min_boundary_iou": min((v["boundary_iou"] for v in views.values()), default=None),
            "mean_render_precision": average([v.get("precision") for v in views.values()]),
            "mean_render_recall": average([v.get("recall") for v in views.values()]),
            "normalized_chamfer_l1": geometry.get("chamfer_l1"), "normalized_chamfer_l2": geometry.get("chamfer_l2"),
            "fscore_tau_0_02": geometry.get("fscore_tau"), "volume_iou_24": geometry.get("volumetric_iou"),
            "normal_consistency": geometry.get("normal_consistency"),
            "geometry_validity": row.get("geometry_validity"),
            "watertight": geometry.get("candidate_topology", {}).get("watertight"),
            "components": geometry.get("candidate_topology", {}).get("connected_components"),
            "workflow_render_validation_s": row.get("reconstruction_and_validation_s"),
            "backend_s": row.get("backend_metrics", {}).get("elapsed_s"),
            "process_status": row.get("process_status", "completed"),
            "result_path": row.get("result_path"), "measured_result_path": row.get("measured_result_path"),
            "mesh_sha256": row.get("mesh_sha256"),
            "geometry_warnings": "; ".join(geometry.get("warnings", []))})
    aggregates = []
    for mode in dict.fromkeys(r["mode"] for r in table):
        base = [r for r in table if r["mode"] == mode and r["arm"] == "baseline"]
        cand = [r for r in table if r["mode"] == mode and r["arm"] == "candidate"]
        def stats(rows, arm):
            return {"mode": mode, "arm": arm, "rows": len(rows), "available_render_rows": sum(r["mean_render_iou"] is not None for r in rows),
                "render_gate_passes": sum(bool(r["render_gate_pass"]) for r in rows),
                "mean_render_iou": average([r["mean_render_iou"] for r in rows]),
                "mean_min_render_iou": average([r["min_render_iou"] for r in rows]),
                "worst_min_render_iou": min((r["min_render_iou"] for r in rows if r["min_render_iou"] is not None), default=None),
                "mean_chamfer_l1": average([r["normalized_chamfer_l1"] for r in rows]),
                "mean_fscore_tau_0_02": average([r["fscore_tau_0_02"] for r in rows]),
                "mean_volume_iou_24": average([r["volume_iou_24"] for r in rows]),
                "volume_available_rows": sum(r["volume_iou_24"] is not None for r in rows),
                "mean_workflow_render_validation_s": average([r["workflow_render_validation_s"] for r in rows])}
        aggregates.extend((stats(base, "baseline"), stats(cand, "candidate")))
    pairs = []
    for baseline in [r for r in table if r["arm"] == "baseline"]:
        candidate = next(r for r in table if r["arm"] == "candidate" and r["case"] == baseline["case"] and r["mode"] == baseline["mode"])
        pair = {"case": baseline["case"], "mode": baseline["mode"],
                "baseline_render_gate": baseline["render_gate_pass"], "candidate_render_gate": candidate["render_gate_pass"]}
        for field in ("mean_render_iou", "min_render_iou", "normalized_chamfer_l1", "fscore_tau_0_02", "workflow_render_validation_s"):
            a, b = baseline[field], candidate[field]
            pair[f"baseline_{field}"] = a
            pair[f"candidate_{field}"] = b
            pair[f"delta_{field}"] = b-a if a is not None and b is not None else None
        pair["decision"] = "unpromoted; multiple factors and single seed"
        if not candidate["render_gate_pass"]:
            pair["decision"] = "reject tested config: render gate failed"
        elif pair["delta_min_render_iou"] is not None and pair["delta_min_render_iou"] < -1e-4:
            pair["decision"] = "reject tested config: weakest view regressed"
        pairs.append(pair)
    for name, data in (("comparison", table), ("technique-summary", aggregates), ("paired-deltas", pairs)):
        (output / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    lines = ["# Matched bounded comparison", "", "Three shapes (box, vase, torus), seed1234; same512px front/side/top references, Blender5.0.1. Passes are silhouette gates only. No configuration defaults were promoted.", "",
             "| Technique | Mean rendered IoU baseline → trial | Mean weakest-view IoU baseline → trial | Gate passes baseline → trial | Mean workflow seconds baseline → trial |", "|---|---:|---:|---:|---:|"]
    fmt = lambda value: "missing" if value is None else f"{value:.4f}"
    for a, b in zip(aggregates[::2], aggregates[1::2]):
        lines.append(f"| {a['mode']} | {fmt(a['mean_render_iou'])} → {fmt(b['mean_render_iou'])} | {fmt(a['mean_min_render_iou'])} → {fmt(b['mean_min_render_iou'])} | {a['render_gate_passes']}/3 → {b['render_gate_passes']}/3 | {fmt(a['mean_workflow_render_validation_s'])} → {fmt(b['mean_workflow_render_validation_s'])} |")
    lines += ["", "3D metrics and each shape's regressions are in comparison.csv and paired-deltas.csv. Chamfer uses the SUM of two directional mean distances; L2 is the sum of squared distances. F-score tolerance is .02 of the longest bbox extent. Uniform centering/scaling preserves rotation and aspect errors. Volume is coarse24³ parity and is absent for open/multiple-component meshes; no self-intersection certification is claimed.", "", "Wall time is a single noisy measurement on a shared host, includes workflow/render/validation, excludes Blender startup and separate geometry evaluation. The pure suite overlapped early candidate cases. Each child had a100s cap; primitive fitting8s and differentiable20s caps. There are no held-out RGB datasets, human studies, or reproduced external SOTA runs in this experiment."]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines[3:14]).replace("→", "->"))

if __name__ == "__main__":
    main()
