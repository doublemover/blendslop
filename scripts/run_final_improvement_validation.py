"""One serial, bounded validation pass against an immutable source candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "blender_blocking", "scripts"], cwd=ROOT, text=True).splitlines()
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names if name.endswith(".py") and (ROOT / name).is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, default=ROOT / "temp/improvement-phase-20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = source_hashes()
    manifest = {"source_sha256": frozen, "repo_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "protocol": "512px orthographic; 8192 surface samples; F-score .02; uniform bbox normalization",
        "candidate": "final: legacy Gaussian; spread primitive mesh-union; fitted refinement; measured ensemble",
        "reference_sources": {}, "control_manifests_verified": {}, "jobs": []}
    def save():
        (out / "validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for group in ("paired", "heldout"):
        source = args.phase / group
        software = json.loads((source / "mask_control-software.json").read_text())
        snapshot = args.phase / "camera-control-source"
        missing = [name for name, digest in software["source_sha256"].items()
                   if not (snapshot / name).is_file() or hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest]
        if missing:
            raise RuntimeError(f"preserved control source is unverifiable: {missing[:5]}")
        manifest["control_manifests_verified"][group] = len(software["source_sha256"])
        destination = out / group
        destination.mkdir()
        shutil.copytree(source / "references", destination / "references")
        shutil.copy2(source / "mask_control.json", destination / "mask_control.json")
        manifest["reference_sources"][group] = str(source / "references")
        for file in (destination / "references").glob("*/manifest.json"):
            reference = json.loads(file.read_text())
            for view, digest in reference["hashes"].items():
                if hashlib.sha256(Path(reference["views"][view]).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"reference image changed: {file} {view}")
            if hashlib.sha256((file.parent / "ground_truth.obj").read_bytes()).hexdigest() != reference["ground_truth_sha256"]:
                raise RuntimeError(f"reference geometry changed: {file}")
    save()
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for key in ("BLENDER_USER_CONFIG", "BLENDER_USER_SCRIPTS", "BLENDER_USER_DATAFILES"):
        env[key] = str(out / key.lower())
    blender = [args.blender, "--background", "--factory-startup", "--disable-autoexec",
               "--threads", "4", "--python-exit-code", "2", "--python"]
    def run(name, command, timeout):
        if source_hashes() != frozen:
            raise RuntimeError("candidate source changed after freeze")
        log = out / (name + ".log")
        print("START " + name, flush=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                status = child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                else:
                    child.kill()
                child.wait()
                status = "timeout_or_interrupted"
        manifest["jobs"].append({"name": name, "status": status,
            "wall_s": time.perf_counter() - start, "command": command, "log": str(log)})
        manifest["source_unchanged"] = source_hashes() == frozen
        save()
        print(f"FINISH {name}: {status}", flush=True)
    for group, cases, seed in [("paired", "box,vase,torus", 1234), ("heldout", "cylinder,bottle,chair", 77)]:
        run("final-" + group, [sys.executable, str(ROOT / "scripts/run_improvement_pass.py"),
            "--blender", args.blender, "--output", str(out / group), "--arm", "final",
            "--cases", cases, "--seed", str(seed), "--timeout", "100"], 2700)
    run("poisson-closed", [sys.executable, str(ROOT / "scripts/run_improvement_pass.py"),
        "--blender", args.blender, "--output", str(out / "paired"), "--arm", "poisson_closed",
        "--modes", "visual_hull_voxel", "--timeout", "100"], 330)
    run("native-full", blender + [str(ROOT / "blender_blocking/test_runner.py"), "--"], 180)
    for name, script, extras, cap in [
        ("solid-usability", "verify_solid_usability.py", ["--output", str(out / "solids")], 100),
        ("exports-centimetres", "verify_blender_exports.py", ["--output", str(out / "exports-centimetres"), "--scale-length", ".01"], 100),
        ("exports-metres", "verify_blender_exports.py", ["--output", str(out / "exports-metres"), "--scale-length", "1"], 100),
        ("candidate-exports", "verify_candidate_exports.py", ["--phase", str(out), "--output", str(out / "candidate-exports")], 180)]:
        run(name, blender + [str(ROOT / "scripts" / script), "--"] + extras, cap)
    run("native-solid-analysis", [sys.executable, str(ROOT / "scripts/analyze_solids.py"),
        "--phase", str(out), "--output", str(out / "native-solids.json")], 300)
    manifest["complete"] = True
    manifest["source_unchanged"] = source_hashes() == frozen
    save()
    if not manifest["source_unchanged"] or any(job["status"] != 0 for job in manifest["jobs"]):
        raise SystemExit("validation has blockers; inspect preserved logs")


if __name__ == "__main__":
    main()
