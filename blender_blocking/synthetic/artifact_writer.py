"""Artifact writing and manifest validation for synthetic shape outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .analytic_sdf import require_numpy
from .degradations import save_png_or_pgm
from .specs import GENERATOR_VERSION, SyntheticArtifactSet, SyntheticShapeSpec, json_safe


def write_artifact_set(
    spec: SyntheticShapeSpec,
    artifacts: Mapping[str, Any],
    output_root: Path,
    source_code_version: str | None = None,
    blender_version: str | None = None,
) -> SyntheticArtifactSet:
    shape_dir = output_root / spec.shape_id
    shape_dir.mkdir(parents=True, exist_ok=True)

    spec_path = shape_dir / "spec.json"
    _write_json(spec_path, spec.to_dict())

    mesh_paths: dict[str, Path] = {
        str(name): Path(path) for name, path in artifacts.get("mesh_paths", {}).items()
    }

    volume_paths: dict[str, Path] = {}
    for name, volume in artifacts.get("volumes", {}).items():
        path = shape_dir / "volume" / f"{name}.npz"
        _write_npz(path, {"occupancy": volume})
        volume_paths[name] = path

    if artifacts.get("sdf_samples"):
        path = shape_dir / "volume" / "sdf-samples.npz"
        _write_npz(path, artifacts["sdf_samples"])
        volume_paths["sdf-samples"] = path

    mask_paths: dict[str, Path] = {}
    for name, image in artifacts.get("masks", {}).items():
        path = shape_dir / "masks" / f"{name}.png"
        result = save_png_or_pgm(path, image)
        actual_path = path if result == "png" else path.with_suffix(".pgm")
        mask_paths[name] = actual_path

    metric_paths: dict[str, Path] = {}
    if artifacts.get("profile_json"):
        profile_path = shape_dir / "profile" / "profile.json"
        _write_json(profile_path, artifacts["profile_json"])
        metric_paths["profile"] = profile_path

    metrics_path = shape_dir / "metrics" / "expected.json"
    _write_json(metrics_path, artifacts.get("quality_targets", {}))
    metric_paths["expected"] = metrics_path

    metadata_path = shape_dir / "metrics" / "ground_truth.json"
    _write_json(metadata_path, artifacts.get("metadata", {}))
    metric_paths["ground_truth"] = metadata_path

    manifest_path = shape_dir / "manifest.json"
    artifact_paths: dict[str, Path] = {
        "spec": spec_path,
        **{f"mesh/{key}": value for key, value in mesh_paths.items()},
        **{f"volume/{key}": value for key, value in volume_paths.items()},
        **{f"mask/{key}": value for key, value in mask_paths.items()},
        **{f"metric/{key}": value for key, value in metric_paths.items()},
    }
    manifest = build_manifest(
        spec,
        shape_dir,
        artifact_paths,
        source_code_version=source_code_version,
        blender_version=blender_version,
        metadata=artifacts.get("metadata", {}),
    )
    _write_json(manifest_path, manifest)

    return SyntheticArtifactSet(
        shape_spec_path=spec_path,
        mesh_paths=mesh_paths,
        volume_paths=volume_paths,
        image_paths={},
        mask_paths=mask_paths,
        metric_paths=metric_paths,
        manifest_path=manifest_path,
    )


def build_manifest(
    spec: SyntheticShapeSpec,
    shape_dir: Path,
    artifact_paths: Mapping[str, Path],
    source_code_version: str | None = None,
    blender_version: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    hashes = {
        key: {
            "path": path.relative_to(shape_dir).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in sorted(artifact_paths.items())
    }
    return json_safe(
        {
            "manifest_version": 1,
            "generator_version": GENERATOR_VERSION,
            "seed": spec.seed,
            "shape_id": spec.shape_id,
            "shape_family": spec.family,
            "parameters": spec.parameters,
            "transforms": spec.transforms,
            "materials": spec.materials,
            "artifact_hashes": hashes,
            "camera_matrices": {},
            "expected_metrics": "metrics/expected.json",
            "known_ambiguity_notes": list(spec.expected_failure_modes),
            "degradation_parameters": dict(metadata or {}).get("degradation_parameters", {}),
            "source_code_version": source_code_version or _git_revision(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "blender_version": blender_version,
        }
    )


def validate_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = manifest_path.parent
    missing: list[str] = []
    mismatched: list[str] = []
    for key, item in manifest.get("artifact_hashes", {}).items():
        artifact_path = root / item["path"]
        if not artifact_path.exists():
            missing.append(key)
            continue
        digest = sha256_file(artifact_path)
        if digest != item["sha256"]:
            mismatched.append(key)
    return {
        "manifest": manifest_path.as_posix(),
        "ok": not missing and not mismatched,
        "missing": missing,
        "mismatched": mismatched,
        "artifact_count": len(manifest.get("artifact_hashes", {})),
    }


def validate_manifest_tree(root: Path) -> dict[str, object]:
    manifest_paths = list(root.rglob("manifest.json")) if root.is_dir() else [root]
    results = [validate_manifest(path) for path in manifest_paths]
    return {
        "root": root.as_posix(),
        "ok": all(result["ok"] for result in results),
        "manifest_count": len(results),
        "results": results,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(data), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    np = require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()
