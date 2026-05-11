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
try:
    from blender_blocking.utils.json_io import write_json as _write_json
except ImportError:  # pragma: no cover - script-style imports
    from utils.json_io import write_json as _write_json

_ARTIFACT_POLICY_VERSION = 1
GENERATED_ARTIFACT_CATEGORIES = frozenset({"mesh", "volume", "mask"})


def write_artifact_set(
    spec: SyntheticShapeSpec,
    artifacts: Mapping[str, Any],
    output_root: Path,
    generation_policy: Mapping[str, object] | None = None,
    source_code_version: str | None = None,
    blender_version: str | None = None,
) -> SyntheticArtifactSet:
    generation_policy = _normalize_generation_policy(generation_policy)
    shape_dir = output_root / spec.shape_id
    shape_dir.mkdir(parents=True, exist_ok=True)

    spec_path = shape_dir / "spec.json"
    _write_json(spec_path, spec.to_dict())

    mesh_paths: dict[str, Path] = {
        str(name): Path(path) for name, path in artifacts.get("mesh_paths", {}).items()
    }

    volume_paths: dict[str, Path] = {}
    for name in artifacts.get("volumes", {}):
        path = shape_dir / "volume" / f"{name}.npz"
        key = _artifact_key("volume", name)
        if _artifact_should_retain(key, generation_policy):
            _write_npz(path, {"occupancy": artifacts["volumes"][name]})
        volume_paths[name] = path

    if artifacts.get("sdf_samples"):
        sdf_path = shape_dir / "volume" / "sdf-samples.npz"
        key = _artifact_key("volume", "sdf-samples")
        if _artifact_should_retain(key, generation_policy):
            _write_npz(sdf_path, artifacts["sdf_samples"])
        volume_paths["sdf-samples"] = sdf_path

    mask_paths: dict[str, Path] = {}
    for name, image in artifacts.get("masks", {}).items():
        preferred_path = shape_dir / "masks" / f"{name}.png"
        key = _artifact_key("mask", name)
        if _artifact_should_retain(key, generation_policy):
            result = save_png_or_pgm(preferred_path, image)
            mask_paths[name] = preferred_path if result == "png" else preferred_path.with_suffix(".pgm")
        else:
            mask_paths[name] = preferred_path

    metric_paths: dict[str, Path] = {
        "expected": shape_dir / "metrics" / "expected.json",
        "ground_truth": shape_dir / "metrics" / "ground_truth.json",
    }
    metric_expected_key = _artifact_key("metric", "expected")
    if _artifact_should_retain(metric_expected_key, generation_policy):
        _write_json(metric_paths["expected"], artifacts.get("quality_targets", {}))
    metric_ground_truth_key = _artifact_key("metric", "ground_truth")
    if _artifact_should_retain(metric_ground_truth_key, generation_policy):
        _write_json(metric_paths["ground_truth"], artifacts.get("metadata", {}))

    if artifacts.get("profile_json"):
        metric_paths["profile"] = shape_dir / "profile" / "profile.json"
        profile_key = _artifact_key("metric", "profile")
        if _artifact_should_retain(profile_key, generation_policy):
            _write_json(metric_paths["profile"], artifacts["profile_json"])

    all_artifact_paths: dict[str, Path] = {
        "spec": spec_path,
        **{_artifact_key("mesh", name): path for name, path in mesh_paths.items()},
        **{_artifact_key("volume", name): path for name, path in volume_paths.items()},
        **{_artifact_key("mask", name): path for name, path in mask_paths.items()},
        **{_artifact_key("metric", name): path for name, path in metric_paths.items()},
    }

    artifact_policies = {
        key: _artifact_policy_entry(key, generation_policy) for key in all_artifact_paths
    }
    retained_artifact_paths = {
        key: path
        for key, path in all_artifact_paths.items()
        if artifact_policies[key]["retained"]
    }

    manifest_path = shape_dir / "manifest.json"
    manifest = build_manifest(
        spec,
        shape_dir,
        retained_artifact_paths,
        artifact_policies=artifact_policies,
        source_code_version=source_code_version,
        blender_version=blender_version,
        metadata=artifacts.get("metadata", {}),
        generation_policy=generation_policy,
    )
    _write_json(manifest_path, manifest)

    return SyntheticArtifactSet(
        shape_spec_path=spec_path,
        mesh_paths=_retained_named_paths("mesh", mesh_paths, generation_policy),
        volume_paths=_retained_named_paths("volume", volume_paths, generation_policy),
        image_paths={},
        mask_paths=_retained_named_paths("mask", mask_paths, generation_policy),
        metric_paths=metric_paths,
        manifest_path=manifest_path,
    )


def build_manifest(
    spec: SyntheticShapeSpec,
    shape_dir: Path,
    artifact_paths: Mapping[str, Path],
    artifact_policies: Mapping[str, Mapping[str, object]] | None = None,
    source_code_version: str | None = None,
    blender_version: str | None = None,
    metadata: Mapping[str, object] | None = None,
    generation_policy: Mapping[str, object] | None = None,
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
            "artifact_policy_version": _ARTIFACT_POLICY_VERSION,
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
            "generation_policy": _normalize_generation_policy(generation_policy),
            "artifact_policies": {
                key: json_safe(value)
                for key, value in sorted((artifact_policies or {}).items())
            },
        }
    )


def validate_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = manifest_path.parent
    missing: list[str] = []
    mismatched: list[str] = []
    ignored: list[str] = []

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        artifact_hashes = {}
    policies = manifest.get("artifact_policies")
    if not isinstance(policies, Mapping):
        policies = {}

    for key, item in artifact_hashes.items():
        if not isinstance(item, Mapping):
            mismatched.append(key)
            continue
        path_value = item.get("path")
        expected_sha = item.get("sha256")
        expected_bytes = item.get("bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(expected_sha, str)
            or not isinstance(expected_bytes, int)
        ):
            mismatched.append(key)
            continue
        artifact_path = root / path_value
        policy = policies.get(key)
        if not artifact_path.exists():
            if isinstance(policy, Mapping) and not bool(policy.get("retained", True)):
                ignored.append(key)
                continue
            missing.append(key)
            continue
        digest = sha256_file(artifact_path)
        if digest != expected_sha or artifact_path.stat().st_size != expected_bytes:
            mismatched.append(key)

    ignored.extend(
        key
        for key, policy in policies.items()
        if isinstance(policy, Mapping)
        and not bool(policy.get("retained", True))
        and key not in artifact_hashes
    )

    return {
        "manifest": manifest_path.as_posix(),
        "ok": not missing and not mismatched,
        "missing": missing,
        "mismatched": mismatched,
        "ignored": sorted(set(ignored)),
        "artifact_count": len(artifact_hashes),
        "generation_policy": manifest.get("generation_policy", {}),
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


def normalize_generation_policy(policy: Mapping[str, object] | None) -> dict[str, bool]:
    return _normalize_generation_policy(policy)


def should_keep_generated_artifacts(policy: Mapping[str, object] | None) -> bool:
    generation_policy = _normalize_generation_policy(policy)
    return not generation_policy["commit_small_fixtures_only"] or generation_policy["keep_heavy_artifacts"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_generation_policy(policy: Mapping[str, object] | None) -> dict[str, bool]:
    if policy is None:
        return {
            "commit_small_fixtures_only": False,
            "keep_heavy_artifacts": True,
        }
    return {
        "commit_small_fixtures_only": bool(policy.get("commit_small_fixtures_only", False)),
        "keep_heavy_artifacts": bool(policy.get("keep_heavy_artifacts", True)),
    }


def _artifact_key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}"


def _artifact_category(key: str) -> str:
    if key == "spec":
        return "source"
    if key.startswith("mesh/"):
        return "mesh"
    if key.startswith("volume/"):
        return "volume"
    if key.startswith("mask/"):
        return "mask"
    if key.startswith("metric/"):
        return "metric"
    return "other"


def _artifact_should_retain(key: str, generation_policy: Mapping[str, bool]) -> bool:
    category = _artifact_category(key)
    if category in {"source", "metric"}:
        return True
    if generation_policy.get("commit_small_fixtures_only"):
        return bool(generation_policy.get("keep_heavy_artifacts"))
    return True


def _artifact_policy_entry(key: str, generation_policy: Mapping[str, bool]) -> dict[str, object]:
    category = _artifact_category(key)
    retained = _artifact_should_retain(key, generation_policy)
    return {
        "category": category,
        "retained": retained,
        "generated": category in GENERATED_ARTIFACT_CATEGORIES,
        "retention_reason": None
        if retained
        else "commit_small_fixtures_only",
    }


def _retained_named_paths(
    category: str,
    paths: Mapping[str, Path],
    generation_policy: Mapping[str, bool],
) -> dict[str, Path]:
    return {
        name: path
        for name, path in paths.items()
        if _artifact_should_retain(_artifact_key(category, name), generation_policy)
    }


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
