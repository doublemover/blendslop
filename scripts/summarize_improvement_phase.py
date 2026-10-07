"""Summarize every matched final cell, including regressions and failed gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    summaries = []
    for group in ("paired", "heldout"):
        root = args.phase / group
        control = json.loads((root / "mask_control.json").read_text())["rows"]
        final = json.loads((root / "final.json").read_text())["rows"]
        assert len(control) == len(final) == 24
        by_key = {(r["case"], r["requested_mode"]): r for r in control}
        for row in final:
            before = by_key[(row["case"], row["requested_mode"])]
            if "process_status" in row:
                pairs.append({"group": group, "case": row["case"], "mode": row["requested_mode"], "process_status": row["process_status"]})
                continue
            record = {"group": group, "case": row["case"], "mode": row["requested_mode"],
                      "selected_backend": row["selected_backend"], "control_result": before["result_path"],
                      "final_result": row["result_path"], "control_passed": before["passed"], "final_passed": row["passed"]}
            for label, accessor in {
                "iou": lambda r: r["average_iou"], "min_iou": lambda r: r["min_view_iou"],
                "fscore": lambda r: r["geometry"]["fscore_tau"], "chamfer_l1": lambda r: r["geometry"]["chamfer_l1"],
                "normal_consistency": lambda r: r["geometry"]["normal_consistency"],
                "novel_iou": lambda r: r["novel_metrics"]["area_iou"],
                "workflow_s": lambda r: r["reconstruction_and_validation_s"]}.items():
                record["control_" + label] = accessor(before)
                record["final_" + label] = accessor(row)
                record["delta_" + label] = accessor(row) - accessor(before)
            record["mesh_identical"] = before.get("mesh_sha256") == row.get("mesh_sha256")
            record["geometry_regression"] = record["delta_fscore"] < -.002 or record["delta_chamfer_l1"] > .002
            record["novel_regression"] = record["delta_novel_iou"] < -.002
            record["silhouette_decision"] = "regression" if record["delta_min_iou"] < -.002 or record["delta_iou"] < -.002 else "improved" if record["delta_iou"] > .002 else "quality_unchanged"
            pairs.append(record)
        for mode in dict.fromkeys(r["requested_mode"] for r in control):
            cells = [r for r in pairs if r["group"] == group and r["mode"] == mode]
            summary = {"group": group, "mode": mode, "cells": len(cells),
                       "control_gate_passes": sum(r.get("control_passed", False) for r in cells),
                       "final_gate_passes": sum(r.get("final_passed", False) for r in cells),
                       "regressions": [r["case"] for r in cells if r.get("silhouette_decision") == "regression"]}
            for metric in ("iou", "min_iou", "fscore", "chamfer_l1", "novel_iou", "workflow_s"):
                for arm in ("control", "final"):
                    values = [r[arm + "_" + metric] for r in cells if arm + "_" + metric in r]
                    summary[arm + "_" + metric] = mean(values) if len(values) == len(cells) else None
            summaries.append(summary)
    native = json.loads((args.phase / "native-solids.json").read_text())
    native_final = [r for r in native["rows"] if "/final/" in r["case"]]
    poisson = json.loads((args.phase / "paired/poisson_closed.json").read_text())["rows"]
    validation = json.loads((args.phase / "validation.json").read_text())
    result = {"validation": validation, "paired_cells": pairs, "techniques": summaries,
              "poisson_closed": poisson, "native_solid_final": native_final,
              "scope_limit": "six synthetic cases, one run each; native fixture editability is not artist judgment"}
    (args.output / "final-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, rows in (("paired-cells", pairs), ("techniques", summaries)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Frozen improvement candidate: matched results", "",
             "Control rows reuse the verified saved camera-control source. All final cells were rerun against one frozen source tree.", "",
             "| Set | Technique | Silhouette IoU before → after | F-score before → after | Workflow seconds before → after | Gates |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        def pair(metric, percent=False):
            a, b = row["control_" + metric], row["final_" + metric]
            if a is None or b is None:
                return "incomplete"
            return f"{100*a:.1f}% → {100*b:.1f}%" if percent else f"{a:.1f} → {b:.1f}"
        lines.append(f'| {row["group"]} | {row["mode"]} | {pair("iou", True)} | {pair("fscore", True)} | {pair("workflow_s")} | {row["control_gate_passes"]}/3 → {row["final_gate_passes"]}/3 |')
    eligible = sum(r.get("validity", {}).get("single_solid_eligible", False) for r in native_final)
    lines += ["", f"Native intersection checks: {eligible}/{len(native_final)} final meshes meet the stated single-solid structural screen.",
              "Assemblies can remain useful editable parts while failing solid acceptance. Per-case failures and all regressions are in paired-cells.csv.", "",
              "Geometry uses 8192 area-weighted samples, uniform bbox center/longest-extent normalization, no rotation or anisotropic fitting, and F-score tolerance .02. Cameras are fixed recorded 512×512 orthographic views.",
              "Workflow times include reconstruction and observed-view validation, exclude startup, novel-view and 3D evaluation. Single-run timings are noisy; no statistical speed claim.", "",
              "Orbit-45 views and reference geometry are evaluation only. Selection uses observed masks and recorded cameras. The ensemble admission budget is soft; each worker has a separate 100-second hard cap."]
    (args.output / "final-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Summarized all 48 matched cells and the bounded Poisson follow-up")


if __name__ == "__main__":
    main()
