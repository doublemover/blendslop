#!/usr/bin/env python3
"""Candidate-blind reference discretization check; no renders or acceptance gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT)]
import test_runner
from evaluation.reference_noise import (reference_noise_workload, validate_reference_workload,
    build_reference_tessellation, reference_noise_observation, canonical_digest, oriented_surface_identity)
from reconstruction.native_geometry import GeometryArrays
from utils.run_ownership import OwnedRun


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    import bpy
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Authored quality-coverage reference output only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=("sphere", "rounded_triangle_dot"),
                        help="An explicitly bounded failed-only subset of the independent contract")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    reference_root, parent = args.reference.absolute(), args.output.absolute()
    if reference_root != reference_root.resolve() or parent != parent.resolve() or not parent.is_relative_to(ROOT / "temp" / "tasks"):
        raise ValueError("references and output must not be redirected; output stays under repository temp/tasks")
    frozen = reference_noise_workload()
    if args.families:
        frozen["families"] = {name: frozen["families"][name] for name in dict.fromkeys(args.families)}
    source_receipt = reference_root / "results.json"
    source_workload = reference_root / "frozen-workload.json"
    retained = json.loads(source_receipt.read_text(encoding="utf-8"))
    validate_reference_workload(json.loads(source_workload.read_text(encoding="utf-8")), frozen)
    identities = {}
    for family in frozen["families"]:
        path = reference_root / family / "evaluated-exact.npz"
        if path.is_symlink() or path.resolve() != path.absolute():
            raise ValueError("redirected reference geometry is refused")
        identities[family] = {"path": str(path), "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "geometry_hash": retained["cases"][family]["geometry_hash"]}
        if identities[family]["npz_sha256"] != retained["cases"][family]["npz_sha256"]:
            raise ValueError("retained authored reference archive hash changed")
    frozen.update(reference_inputs=identities, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_receipts={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_receipt, source_workload)},
        environment={"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__},
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_files_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__).resolve(), ROOT / "blender_blocking/evaluation/reference_noise.py",
            ROOT / "blender_blocking/evaluation/canonical_artifacts.py", ROOT / "blender_blocking/evaluation/surface_quality.py",
            ROOT / "blender_blocking/evaluation/comparable_geometry.py", ROOT / "blender_blocking/primitives/rounded_triangle.py",
            ROOT / "blender_blocking/synthetic/quality_contracts.py", ROOT / "blender_blocking/synthetic/blender_builders.py")})
    frozen_hash = canonical_digest(frozen)
    owner = OwnedRun(parent, producer="candidate_blind_reference_noise",
        max_generated_bytes=frozen["limits"]["max_generated_bytes"], shared_inputs={"references": identities})
    with owner:
        output = owner.root
        _write(output / "frozen-workload.json", frozen)
        owner.register_file("frozen-workload.json", "final_output")
        started = time.monotonic()
        receipt = {"status": "running", "frozen_workload_sha256": frozen_hash, "families": {},
                   "candidate_accessed": False, "family_acceptance_contract": None,
                   "aggregate_accepted": False, "acceptance_blocker": frozen["family_acceptance_status"]}
        _write(output / "results.json", receipt)
        owner.register_file("results.json", "final_output")
        for family, identity in identities.items():
            if canonical_digest(frozen) != frozen_hash:
                raise ValueError("frozen workload changed before reference measurement")
            path = Path(identity["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != identity["npz_sha256"]:
                raise ValueError("reference archive changed after freezing")
            with np.load(path, allow_pickle=False) as stored:
                baseline = GeometryArrays.capture(stored["vertices"], stored["faces"])
            if baseline.content_hash != identity["geometry_hash"]:
                raise ValueError("reference geometry differs from retained authored receipt")
            regenerated = build_reference_tessellation(family, "baseline", frozen)
            baseline_identity = {"retained_geometry_hash": baseline.content_hash,
                "regenerated_geometry_hash": regenerated.content_hash,
                "indexed_hash_matched": regenerated.content_hash == baseline.content_hash,
                "retained_oriented_surface_hash": oriented_surface_identity(baseline),
                "regenerated_oriented_surface_hash": oriented_surface_identity(regenerated),
                "equivalence_scope": "reference-surface geometric equivalence only: exact vertex multiset and oriented triangles; no numeric tolerance; topology identity remains independent"}
            receipt.setdefault("baseline_identity", {})[family] = baseline_identity
            _write(output / "results.json", receipt)
            owner.register_file("results.json", "final_output")
            if baseline_identity["regenerated_oriented_surface_hash"] != baseline_identity["retained_oriented_surface_hash"]:
                name = family + "-failed-regenerated-baseline-exact.npz"
                np.savez_compressed(output / name, vertices=regenerated.vertices, faces=regenerated.faces)
                owner.register_file(name, "diagnostic")
                receipt.update(status="baseline_identity_failed", failed_family=family)
                _write(output / "results.json", receipt)
                owner.register_file("results.json", "final_output")
                raise ValueError("regenerated oriented baseline differs from retained authored reference: " + family)
            levels = {"baseline": baseline}
            for level in ("dense", "finer"):
                if time.monotonic() - started > frozen["limits"]["wall_seconds"]:
                    raise TimeoutError("frozen reference-noise time bound exceeded")
                levels[level] = build_reference_tessellation(family, level, frozen)
                name = family + "-" + level + "-exact.npz"
                np.savez_compressed(output / name, vertices=levels[level].vertices, faces=levels[level].faces)
                owner.register_file(name, "final_output")
            print("reference-noise " + family, flush=True)
            observation = reference_noise_observation(family, levels, frozen)
            observation["baseline_exact_oriented_surface_reproduction"] = True
            observation["baseline_identity"] = baseline_identity
            observation["baseline_npz_sha256"] = identity["npz_sha256"]
            receipt["families"][family] = observation
            _write(output / "results.json", receipt)
            owner.register_file("results.json", "final_output")
        if canonical_digest(frozen) != frozen_hash:
            raise ValueError("frozen workload changed during reference measurement")
        receipt.update(status="reference_noise_measured", elapsed_seconds=time.monotonic() - started)
        _write(output / "results.json", receipt)
        owner.register_file("results.json", "final_output")
        print("REFERENCE_NOISE_RESULT=" + str(output / "results.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
