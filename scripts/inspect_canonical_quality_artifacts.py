#!/usr/bin/env python3
"""Read retained quality artifacts into a new inventory; never render or qualify.

Example: --receipt .../results.json --row cases/smooth_vase --directory
.../smooth_vase --pass-state mask=completed --pass-state neutral=unrun
--pass-state normals=unrun --output temp/tasks/inspection/smooth-vase.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT)]
import test_runner  # expose the already installed bundled-Python dependencies
from evaluation.canonical_artifacts import canonical_artifact_inventory, INSPECTION_PASSES


def receipt_row(receipt, pointer):
    value = receipt
    for key in pointer.split("/") if pointer else ():
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError("receipt row must be an object")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--row", default="", help="Slash-separated receipt keys, e.g. cases/smooth_vase")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pass-state", action="append", default=[], help="Explicit pass=completed/unrun/unavailable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if (output != output.resolve() or not output.is_relative_to(ROOT / "temp" / "tasks") or
            output.exists()):
        raise ValueError("inventory output must be a new contained repository temp/tasks file")
    states = {name: "unavailable" for name in INSPECTION_PASSES}
    for declaration in args.pass_state:
        name, state = declaration.split("=", 1)
        if name not in states:
            raise ValueError("unknown inspection pass: " + name)
        states[name] = state
    receipt_path = args.receipt.resolve()
    source = receipt_path.read_bytes()
    row = receipt_row(json.loads(source), args.row)
    inventory = canonical_artifact_inventory(args.directory, geometry_hash=row["geometry_hash"],
        camera_records=row.get("reference_cameras", row.get("cameras")), pass_states=states)
    inventory["source_receipt"] = {"path": str(receipt_path), "sha256": hashlib.sha256(source).hexdigest(),
                                   "row": args.row}
    inventory["inspection_scope"] = "existing retained files only; no missing passes generated or quality gates inferred"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": inventory["status"], "output": str(output),
                      "retained_passes": len(inventory["retained_paths"]), "gaps": len(inventory["gaps"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
