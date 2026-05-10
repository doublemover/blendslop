# ruff: noqa: E402,F401,F403
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from blender_blocking.config import BlockingConfig
from blender_blocking.config import CandidateConfig as ConfigCandidateConfig
from blender_blocking.config import RenderConfig
from blender_blocking.evaluation.cost_model import CostRecorder
from blender_blocking.evaluation.silhouette_eval import (
    SilhouetteGateConfig,
    evaluate_silhouette_pair,
    missing_silhouette_view,
    summarize_silhouette_views,
)
from blender_blocking.integration.blender_ops.render_utils import (
    parse_orbit_view_degrees,
    render_orthogonal_views,
)
from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.utils.generation_context import GenerationContext
from blender_blocking.utils.progress import progress_bar
from blender_blocking.validation.silhouette_iou import canonicalize_mask, mask_from_image_array
from blender_blocking.e2e.constants import *
from blender_blocking.e2e.console import _print_kv_table, _print_rule, _print_section
from blender_blocking.e2e.cost import _matrix_cost_summary
from blender_blocking.e2e.ground_truth import _synthetic_ground_truth_row
from blender_blocking.e2e.payloads import _evaluation_outputs_from_payload, _json_dump
from blender_blocking.e2e.validator import test_with_custom_images


def run_synthetic_suite_matrix(
    *,
    suite: str,
    modes: Sequence[str] = DEFAULT_SYNTHETIC_MATRIX_MODES,
    seed: int = 1234,
    count: Optional[int] = None,
    output_root: Path = TEMP_OUTPUT_ROOT / "e2e_synthetic",
    base_config: Optional[BlockingConfig] = None,
    iou_threshold: float = 0.7,
    view_thresholds: Optional[Dict[str, float]] = None,
    boundary_iou_threshold: Optional[float] = None,
    signed_distance_loss_threshold: Optional[float] = None,
    validation_mode: str = "auto",
    config_label: str = "synthetic",
    result_json: Optional[Path] = None,
    run_id: Optional[str] = None,
    novel_view_names: Sequence[str] = (),
    novel_compute_ssim: bool = True,
    novel_compute_lpips: bool = False,
    novel_psnr_threshold: Optional[float] = 20.0,
    novel_ssim_threshold: Optional[float] = 0.65,
    novel_lpips_threshold: Optional[float] = None,
    cost_report_json: Optional[Path] = None,
    cost_track_memory: bool = False,
    cost_fail_max_wall_ms: Optional[float] = None,
    cost_fail_max_backend_wall_ms: Optional[float] = None,
    progress: bool = False,
    strict_skips: bool = False,
) -> bool:
    """Render Blender-backed synthetic fixtures and validate each mode."""
    if not BLENDER_AVAILABLE:
        print("ERROR: synthetic suite matrix requires Blender.")
        return False

    from blender_blocking.synthetic.blender_builders import render_views
    from blender_blocking.synthetic.registry import get_definition, specs_for_suite

    output_root = Path(output_root).resolve()
    if result_json is not None:
        result_json = Path(result_json).resolve()
    if cost_report_json is not None:
        cost_report_json = Path(cost_report_json).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix = []
    run_label = run_id or _utc_run_id(f"{suite}_matrix")
    base = base_config or BlockingConfig()
    specs = specs_for_suite(suite, seed=seed, count=count)

    _print_rule("SYNTHETIC E2E MATRIX", width=72)
    _print_kv_table(
        (
            ("suite", suite),
            ("seed", seed),
            ("count", len(specs)),
            ("modes", ",".join(modes)),
            ("output", output_root),
            ("run_id", run_label),
        )
    )

    for spec_index, spec in enumerate(specs):
        definition_name = _definition_name_from_spec(spec)
        definition = get_definition(definition_name)
        case = f"{suite}:{definition_name}"
        if not definition.blender_supported:
            row = {
                "artifact": "e2e",
                "case": case,
                "suite": suite,
                "shape_id": spec.shape_id,
                "definition": definition_name,
                "mode": "",
                "name": f"{spec.shape_id}/skipped",
                "status": "skip",
                "passed": True,
                "metrics": {"passed": 1.0},
                "message": "synthetic definition has no Blender mesh",
            }
            matrix.append(row)
            print(f"SKIP: {spec.shape_id}: no Blender mesh builder")
            continue

        reference_dir = output_root / "references" / spec.shape_id
        include_orbit = bool(
            validation_mode == "novel-view"
            and any(parse_orbit_view_degrees(view) is not None for view in novel_view_names)
        )
        rendered = render_views(
            spec,
            reference_dir,
            resolution=tuple(base.render_silhouette.resolution),
            include_orbit=include_orbit,
        )
        reference_paths = {
            key: str(rendered[key])
            for key in ("front", "side", "top")
            if key in rendered
        }
        novel_reference_paths = {
            key: str(rendered[key])
            for key in novel_view_names
            if key in rendered
        }
        if set(reference_paths) != {"front", "side", "top"}:
            row = {
                "artifact": "e2e",
                "case": case,
                "suite": suite,
                "shape_id": spec.shape_id,
                "definition": definition_name,
                "mode": "",
                "name": f"{spec.shape_id}/incomplete-references",
                "status": "skip",
                "passed": not strict_skips,
                "metrics": {"passed": 0.0 if strict_skips else 1.0},
                "message": "front/side/top synthetic references were not all generated",
            }
            matrix.append(row)
            continue

        for mode in modes:
            cfg = copy.deepcopy(base)
            cfg.reconstruction.reconstruction_mode = mode
            cfg.reconstruction.num_slices = base.reconstruction.num_slices
            mode_label = f"{config_label}-{mode}"
            case_run_id = f"{run_label}_{spec_index:03d}_{mode}"
            case_dir = output_root / "results" / mode / spec.shape_id
            case_json = case_dir / "result.json"
            print(f"\nCase: {spec.shape_id} mode={mode}")
            try:
                passed = test_with_custom_images(
                    reference_paths["front"],
                    reference_paths["side"],
                    reference_paths["top"],
                    num_slices=cfg.reconstruction.num_slices,
                    iou_threshold=iou_threshold,
                    view_thresholds=view_thresholds,
                    boundary_iou_threshold=boundary_iou_threshold,
                    signed_distance_loss_threshold=signed_distance_loss_threshold,
                    render_config=cfg.render_silhouette,
                    workflow_config=cfg,
                    config_label=mode_label,
                    validation_mode=validation_mode,
                    render_output_dir=case_dir / "renders",
                    artifact_root=case_dir / "artifacts",
                    result_json=case_json,
                    run_id=case_run_id,
                    novel_view_reference_paths=novel_reference_paths,
                    novel_view_names=novel_view_names,
                    novel_compute_ssim=novel_compute_ssim,
                    novel_compute_lpips=novel_compute_lpips,
                    novel_psnr_threshold=novel_psnr_threshold,
                    novel_ssim_threshold=novel_ssim_threshold,
                    novel_lpips_threshold=novel_lpips_threshold,
                    cost_track_memory=cost_track_memory,
                    cost_fail_max_wall_ms=cost_fail_max_wall_ms,
                    cost_fail_max_backend_wall_ms=cost_fail_max_backend_wall_ms,
                    progress=progress,
                )
                result_payload = _load_optional_json(case_json)
                metrics = _matrix_metrics(result_payload, passed)
                ground_truth = _synthetic_ground_truth_row(
                    spec,
                    result_payload,
                    reference_paths=reference_paths,
                    config=cfg,
                )
                metrics.update(ground_truth.get("metrics", {}))
                status = "pass" if passed else "fail"
                message = ""
            except Exception as exc:
                passed = False
                result_payload = {}
                metrics = {"passed": 0.0}
                ground_truth = _synthetic_ground_truth_row(
                    spec,
                    result_payload,
                    reference_paths=reference_paths,
                    config=cfg,
                )
                status = "error"
                message = str(exc)
                if progress:
                    import traceback

                    traceback.print_exc()

            row = {
                "artifact": "e2e",
                "case": case,
                "suite": suite,
                "shape_id": spec.shape_id,
                "definition": definition_name,
                "mode": mode,
                "name": f"{spec.shape_id}/{mode}",
                "status": status,
                "passed": passed,
                "metrics": metrics,
                "ground_truth": ground_truth,
                "result_json": case_json.as_posix(),
                "message": message,
            }
            cost_report = result_payload.get("cost_report")
            if isinstance(cost_report, Mapping):
                row["cost_report"] = cost_report
            cost_gate = result_payload.get("cost_gate")
            if isinstance(cost_gate, Mapping):
                row["cost_gate"] = cost_gate
            row.update(_evaluation_outputs_from_payload(result_payload))
            matrix.append(row)

    skipped_failures = [
        row for row in matrix if row["status"] == "skip" and not row["passed"]
    ]
    failed = [row for row in matrix if row["status"] in {"fail", "error"}]
    cost_summary = _matrix_cost_summary(matrix)
    summary = {
        "schema_version": "e2e_synthetic_matrix_v1",
        "generated_at": _utc_now(),
        "suite": suite,
        "seed": seed,
        "run_id": run_label,
        "modes": list(modes),
        "output_root": output_root.as_posix(),
        "passed": not failed and not skipped_failures,
        "counts": {
            "total": len(matrix),
            "passed": sum(1 for row in matrix if row["status"] == "pass"),
            "failed": len(failed),
            "skipped": sum(1 for row in matrix if row["status"] == "skip"),
        },
        "cost_report": cost_summary,
        "matrix": matrix,
        "bundles": _evaluation_bundles_from_matrix(matrix),
    }
    if cost_report_json:
        if result_json is not None and cost_report_json == result_json:
            print("\nCost report included in synthetic matrix JSON")
        else:
            _json_dump(cost_report_json, cost_summary)
            print(f"\nSaved synthetic matrix cost report JSON: {cost_report_json}")
    if result_json:
        _json_dump(result_json, summary)
        print(f"\nSaved synthetic matrix JSON: {result_json}")
    _print_section("Synthetic Matrix Summary")
    _print_kv_table(
        (
            ("passed", summary["passed"]),
            ("total", summary["counts"]["total"]),
            ("failed", summary["counts"]["failed"]),
            ("skipped", summary["counts"]["skipped"]),
        )
    )
    return bool(summary["passed"])

def _definition_name_from_spec(spec: object) -> str:
    parameters = getattr(spec, "parameters")
    for key in (
        "primitive",
        "profile_kind",
        "blockout_kind",
        "mask_kind",
        "degradation",
    ):
        if key in parameters:
            return str(parameters[key])
    raise ValueError(
        f"Cannot infer registry definition for {getattr(spec, 'shape_id', '<unknown>')}"
    )

def _matrix_metrics(payload: Mapping[str, Any], passed: bool) -> Dict[str, float]:
    metrics: Dict[str, float] = {"passed": 1.0 if passed else 0.0}
    for key in (
        "average_iou",
        "min_view_iou",
        "failed_required_view_count",
        "missing_required_metric_count",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    summary = payload.get("silhouette_summary")
    if isinstance(summary, Mapping):
        for key in (
            "mean_boundary_iou",
            "min_boundary_iou",
            "mean_signed_distance_loss",
            "required_view_count",
            "missing_required_view_count",
        ):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                metrics[f"silhouette_{key}"] = float(value)
    novel = payload.get("novel_view")
    if isinstance(novel, Mapping):
        for key in ("psnr", "ssim", "lpips", "mse", "image_count"):
            value = novel.get(key)
            if isinstance(value, (int, float)):
                metrics[f"novel_view_{key}"] = float(value)
    novel_summary = payload.get("novel_view_summary")
    if isinstance(novel_summary, Mapping):
        value = novel_summary.get("passed")
        if isinstance(value, bool):
            metrics["novel_view_passed"] = 1.0 if value else 0.0
    cost_report = payload.get("cost_report")
    if isinstance(cost_report, Mapping):
        value = cost_report.get("combined_total_wall_ms")
        if isinstance(value, (int, float)):
            metrics["cost_combined_total_wall_ms"] = float(value)
        validation = cost_report.get("validation")
        if isinstance(validation, Mapping):
            value = validation.get("total_wall_ms")
            if isinstance(value, (int, float)):
                metrics["cost_validation_total_wall_ms"] = float(value)
        backend_cost = cost_report.get("backend")
        if isinstance(backend_cost, Mapping):
            value = backend_cost.get("total_wall_ms")
            if isinstance(value, (int, float)):
                metrics["cost_backend_total_wall_ms"] = float(value)
    cost_gate = payload.get("cost_gate")
    if isinstance(cost_gate, Mapping):
        value = cost_gate.get("passed")
        if isinstance(value, bool):
            metrics["cost_gate_passed"] = 1.0 if value else 0.0
    views = payload.get("views", {})
    if isinstance(views, Mapping):
        for view, view_payload in views.items():
            if isinstance(view_payload, Mapping) and isinstance(
                view_payload.get("iou"), (int, float)
            ):
                metrics[f"{view}_iou"] = float(view_payload["iou"])
    backend = payload.get("backend_result")
    backend_payload = backend if isinstance(backend, Mapping) else payload
    status = backend_payload.get("status")
    if isinstance(status, str):
        metrics["backend_status_ok"] = 1.0 if _backend_status_ok(status) else 0.0
    selected = backend_payload.get("selected")
    if isinstance(selected, Mapping):
        status = selected.get("status")
        metrics["backend_status_ok"] = 1.0 if _backend_status_ok(status) else 0.0
        metric_result = selected.get("metric_result", {})
        if isinstance(metric_result, Mapping):
            for key in (
                "area_iou_mean",
                "area_iou_min",
                "boundary_iou_mean",
                "topology_score",
                "editability_score",
                "complexity_penalty",
                "elapsed_s",
            ):
                value = metric_result.get(key)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
    elif isinstance(backend_payload.get("metric_result"), Mapping):
        metric_result = backend_payload["metric_result"]
        for key in (
            "area_iou_mean",
            "area_iou_min",
            "boundary_iou_mean",
            "topology_score",
            "editability_score",
            "complexity_penalty",
            "elapsed_s",
        ):
            value = metric_result.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
    for bundle in _evaluation_bundle_sequence(payload):
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if not isinstance(metric, Mapping):
                    continue
                name = str(metric.get("name", "")).replace(".", "_")
                value = metric.get("value")
                if name and isinstance(value, (int, float, bool)):
                    metrics[name] = float(value)
    return metrics

def _evaluation_bundle_sequence(
    payload: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    outputs = _evaluation_outputs_from_payload(payload)
    bundles = []
    value = outputs.get("evaluation_bundles")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        bundles.extend(item for item in value if isinstance(item, Mapping))
    single = outputs.get("evaluation_bundle")
    if isinstance(single, Mapping):
        bundles.append(single)
    return tuple(bundles)

def _evaluation_bundles_from_matrix(
    matrix: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    bundles: list[Mapping[str, Any]] = []
    for row in matrix:
        bundles.extend(_evaluation_bundle_sequence(row))
    return bundles

def _backend_status_ok(status: object) -> bool:
    return str(status) in BACKEND_STATUS_OK

def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _utc_run_id(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{safe}"
