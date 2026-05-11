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
from blender_blocking.metrics.namespaces import (
    namespace_metric_key,
    set_metric_path,
    set_render_aggregate_metrics,
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
from blender_blocking.e2e.backend_status import (
    _backend_status_ok as _status_ok_from_backend,
)
from blender_blocking.e2e.backend_status import _candidate_status_payload, _print_backend_summary
from blender_blocking.e2e.console import _print_kv_table, _print_rule, _print_section
from blender_blocking.e2e.cost import _matrix_cost_summary
from blender_blocking.e2e.ground_truth import _synthetic_ground_truth_row
from blender_blocking.e2e.payloads import _evaluation_outputs_from_payload, _json_dump
from blender_blocking.e2e.validator import test_with_custom_images
from blender_blocking.utils.path_safety import compact_path_segment


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
    allow_failed_rows: bool = False,
) -> bool:
    """Run a synthetic suite across reconstruction modes.

    Blender-backed fixtures render front/side/top references from generated
    geometry. Pure 2D mask fixtures build backend targets directly from their
    generated masks so adversarial/capture suites become scored rows instead of
    allowed skips.
    """

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
    has_blender_cases = any(
        get_definition(_definition_name_from_spec(spec)).blender_supported
        for spec in specs
    )
    if has_blender_cases and not BLENDER_AVAILABLE:
        print("ERROR: synthetic suite matrix contains Blender-backed fixtures.")
        print("Use a pure-mask suite or run inside Blender.")
        return False

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
            matrix.extend(
                _run_pure_mask_matrix_rows(
                    spec=spec,
                    suite=suite,
                    case=case,
                    definition_name=definition_name,
                    modes=modes,
                    base_config=base,
                    output_root=output_root,
                    run_label=run_label,
                    spec_index=spec_index,
                    config_label=config_label,
                    validation_mode=validation_mode,
                    strict_skips=strict_skips,
                    progress=progress,
                )
            )
            continue

        reference_dir = (
            output_root
            / "ref"
            / compact_path_segment(spec.shape_id, max_length=36, fallback="shape")
        )
        orbit_angles = tuple(
            angle
            for angle in (
                parse_orbit_view_degrees(view) for view in novel_view_names
            )
            if angle is not None
        )
        include_orbit = bool(validation_mode == "novel-view" and orbit_angles)
        rendered = render_views(
            spec,
            reference_dir,
            resolution=tuple(base.render_silhouette.resolution),
            include_orbit=include_orbit,
            orbit_angles=orbit_angles if orbit_angles else None,
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
            case_dir = (
                output_root
                / "r"
                / compact_path_segment(mode, max_length=28, fallback="mode")
                / compact_path_segment(spec.shape_id, max_length=36, fallback="shape")
            )
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
                    render_output_dir=case_dir / "r",
                    artifact_root=case_dir / "a",
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
    failed = [row for row in matrix if row["status"] == "fail"]
    errors = [row for row in matrix if row["status"] == "error"]
    cost_summary = _matrix_cost_summary(matrix)
    contract_passed = not errors and not skipped_failures
    summary = {
        "schema_version": "e2e_synthetic_matrix_v1",
        "generated_at": _utc_now(),
        "suite": suite,
        "seed": seed,
        "run_id": run_label,
        "modes": list(modes),
        "output_root": output_root.as_posix(),
        "passed": not failed and contract_passed,
        "contract_passed": contract_passed,
        "allow_failed_rows": allow_failed_rows,
        "counts": {
            "total": len(matrix),
            "passed": sum(1 for row in matrix if row["status"] == "pass"),
            "failed": len(failed) + len(errors),
            "errors": len(errors),
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
            ("contract_passed", summary["contract_passed"]),
            ("total", summary["counts"]["total"]),
            ("failed", summary["counts"]["failed"]),
            ("errors", summary["counts"]["errors"]),
            ("skipped", summary["counts"]["skipped"]),
        )
    )
    return bool(summary["contract_passed"] if allow_failed_rows else summary["passed"])

def _run_pure_mask_matrix_rows(
    *,
    spec: object,
    suite: str,
    case: str,
    definition_name: str,
    modes: Sequence[str],
    base_config: BlockingConfig,
    output_root: Path,
    run_label: str,
    spec_index: int,
    config_label: str,
    validation_mode: str,
    strict_skips: bool,
    progress: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if validation_mode not in {"auto", "backend-status"}:
        passed = not strict_skips
        rows.append(
            {
                "artifact": "e2e",
                "case": case,
                "suite": suite,
                "shape_id": getattr(spec, "shape_id", ""),
                "definition": definition_name,
                "mode": "",
                "name": f"{getattr(spec, 'shape_id', '')}/pure-mask-unsupported",
                "status": "skip",
                "passed": passed,
                "metrics": {"passed": 1.0 if passed else 0.0},
                "message": (
                    "pure 2D mask fixtures support backend-status synthetic "
                    f"matrix validation, not {validation_mode!r}"
                ),
            }
        )
        print(f"SKIP: {getattr(spec, 'shape_id', '')}: pure masks require backend-status")
        return rows

    try:
        pure_case = _build_pure_mask_case(spec=spec, output_root=output_root)
    except Exception as exc:
        rows.append(
            {
                "artifact": "e2e",
                "case": case,
                "suite": suite,
                "shape_id": getattr(spec, "shape_id", ""),
                "definition": definition_name,
                "mode": "",
                "name": f"{getattr(spec, 'shape_id', '')}/pure-mask-error",
                "status": "error",
                "passed": False,
                "metrics": {"passed": 0.0},
                "message": str(exc),
            }
        )
        return rows

    for mode in modes:
        cfg = copy.deepcopy(base_config)
        cfg.reconstruction.reconstruction_mode = mode
        mode_label = f"{config_label}-{mode}"
        case_run_id = f"{run_label}_{spec_index:03d}_{mode}"
        case_dir = (
            output_root
            / "r"
            / compact_path_segment(mode, max_length=28, fallback="mode")
            / compact_path_segment(
                getattr(spec, "shape_id", "pure_mask"),
                max_length=36,
                fallback="shape",
            )
        )
        case_json = case_dir / "result.json"
        print(f"\nCase: {getattr(spec, 'shape_id', '')} mode={mode} [pure-mask]")
        try:
            result_payload = _run_pure_mask_backend_status(
                views=pure_case["views"],
                mode=mode,
                config=cfg,
                config_label=mode_label,
                run_id=case_run_id,
                artifact_root=case_dir / "a",
                result_json=case_json,
                pure_case=pure_case,
            )
            status = str(result_payload.get("status", "unstructured"))
            passed = _status_ok_from_backend(status)
            metrics = _matrix_metrics(result_payload, passed)
            ground_truth = _synthetic_ground_truth_row(
                spec,
                result_payload,
                reference_paths=pure_case.get("reference_paths", {}),
                config=cfg,
            )
            metrics.update(ground_truth.get("metrics", {}))
            row_status = "pass" if passed else "fail"
            message = ""
        except Exception as exc:
            result_payload = {}
            metrics = {"passed": 0.0}
            ground_truth = _synthetic_ground_truth_row(
                spec,
                result_payload,
                reference_paths=pure_case.get("reference_paths", {}),
                config=cfg,
            )
            passed = False
            row_status = "error"
            message = str(exc)
            if progress:
                import traceback

                traceback.print_exc()

        row = {
            "artifact": "e2e",
            "case": case,
            "suite": suite,
            "shape_id": getattr(spec, "shape_id", ""),
            "definition": definition_name,
            "mode": mode,
            "name": f"{getattr(spec, 'shape_id', '')}/{mode}",
            "status": row_status,
            "passed": passed,
            "metrics": metrics,
            "ground_truth": ground_truth,
            "reference_paths": dict(pure_case.get("reference_paths", {})),
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
        rows.append(row)
    return rows

def _build_pure_mask_case(
    *,
    spec: object,
    output_root: Path,
) -> dict[str, Any]:
    from blender_blocking.synthetic.degradations import save_png_or_pgm
    from blender_blocking.synthetic.ground_truth import build_pure_artifacts

    artifacts = build_pure_artifacts(spec)
    masks = artifacts.get("masks", {})
    if not isinstance(masks, Mapping) or not masks:
        raise ValueError("pure-mask synthetic fixture did not emit masks")

    source_images = _select_pure_mask_source_images(masks, spec)
    if not source_images:
        raise ValueError("pure-mask synthetic fixture had no usable mask images")

    reference_dir = (
        output_root
        / "ref"
        / compact_path_segment(
            getattr(spec, "shape_id", "pure_mask"),
            max_length=36,
            fallback="shape",
        )
    )
    reference_paths: dict[str, str] = {}
    views: dict[str, np.ndarray] = {}
    for view, image in source_images.items():
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"pure mask view {view!r} must be a 2D image")
        path = reference_dir / f"{view}.png"
        save_png_or_pgm(path, arr)
        actual_path = path if path.exists() else path.with_suffix(".pgm")
        reference_paths[view] = actual_path.as_posix()
        views[view] = arr

    return {
        "views": views,
        "reference_paths": reference_paths,
        "metadata": dict(artifacts.get("metadata", {}) or {}),
        "quality_targets": artifacts.get("quality_targets", {}),
    }

def _select_pure_mask_source_images(
    masks: Mapping[str, Any],
    spec: object,
) -> dict[str, np.ndarray]:
    parameters = getattr(spec, "parameters", {}) or {}
    degradation = str(parameters.get("degradation", ""))
    preferred_prefixes = ("noisy", "clean") if degradation else ("clean", "noisy")
    views: dict[str, np.ndarray] = {}

    for view in ("front", "side", "top"):
        image = _mask_by_logical_view(masks, view, preferred_prefixes)
        if image is not None:
            views[view] = np.asarray(image, dtype=np.uint8)

    if views:
        kind = str(parameters.get("mask_kind", parameters.get("degradation", "")))
        base = views["front"] if "front" in views else next(iter(views.values()))
        views.setdefault("front", np.asarray(base, dtype=np.uint8))
        views.setdefault("side", _pure_mask_variant(base, kind=kind, view="side"))
        views.setdefault("top", _pure_mask_variant(base, kind=kind, view="top"))
        if degradation == "missing_top_view":
            views.pop("top", None)
        return views

    first = np.asarray(next(iter(masks.values())), dtype=np.uint8)
    kind = str(parameters.get("mask_kind", parameters.get("degradation", "")))
    expanded = {
        "front": first,
        "side": _pure_mask_variant(first, kind=kind, view="side"),
        "top": _pure_mask_variant(first, kind=kind, view="top"),
    }
    if degradation == "missing_top_view":
        expanded.pop("top", None)
    return expanded

def _mask_by_logical_view(
    masks: Mapping[str, Any],
    view: str,
    prefixes: Sequence[str],
) -> np.ndarray | None:
    for prefix in prefixes:
        key = f"{prefix}/{view}"
        if key in masks:
            return np.asarray(masks[key], dtype=np.uint8)
    for key, image in masks.items():
        if str(key).endswith(f"/{view}"):
            return np.asarray(image, dtype=np.uint8)
    return None

def _pure_mask_variant(image: np.ndarray, *, kind: str, view: str) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if kind == "inconsistent_front_side" and view == "side":
        shifted = np.roll(arr, max(1, arr.shape[1] // 8), axis=1)
        band = np.full_like(shifted, 255)
        band[:, arr.shape[1] // 3 : (arr.shape[1] * 2) // 3] = shifted[
            :, arr.shape[1] // 3 : (arr.shape[1] * 2) // 3
        ]
        return band
    if view == "side":
        return np.fliplr(arr)
    if view == "top":
        return np.flipud(arr)
    return arr

def _run_pure_mask_backend_status(
    *,
    views: Mapping[str, np.ndarray],
    mode: str,
    config: BlockingConfig,
    config_label: str,
    run_id: str,
    artifact_root: Path,
    result_json: Path,
    pure_case: Mapping[str, Any],
) -> dict[str, Any]:
    from blender_blocking.main_integration import BlockingWorkflow

    context = GenerationContext(run_id=run_id)
    context.artifact_root = str(artifact_root)
    workflow = BlockingWorkflow(config=config, context=context)
    workflow.views = {view: np.asarray(image) for view, image in views.items()}
    workflow.run_backend_reconstruction(mode=mode)
    status, payload = _candidate_status_payload(workflow.reconstruction_result)
    payload = dict(payload)
    payload.setdefault("status", status)
    payload["mode"] = mode
    payload["validation_mode"] = "backend-status"
    payload["config_label"] = config_label
    payload["pure_mask_case"] = {
        "views": sorted(views),
        "reference_paths": dict(pure_case.get("reference_paths", {})),
        "metadata": dict(pure_case.get("metadata", {}) or {}),
        "quality_targets": pure_case.get("quality_targets", {}),
    }
    payload.update(_evaluation_outputs_from_payload(payload))
    payload["passed"] = _status_ok_from_backend(status)
    _json_dump(result_json, payload)
    print(f"\nSaved result JSON: {result_json}")
    if workflow.reconstruction_result is not None:
        _print_backend_summary(workflow.reconstruction_result)
    return payload

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

def _matrix_metrics(payload: Mapping[str, Any], passed: bool) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"passed": 1.0 if passed else 0.0}
    validation_mode = str(payload.get("validation_mode") or "")
    if validation_mode:
        metrics["validation_mode"] = validation_mode
    for key in (
        "average_iou",
        "min_view_iou",
        "failed_required_view_count",
        "missing_required_metric_count",
    ):
        value = payload.get(key)
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if key == "average_iou" and validation_mode == "render-iou":
            set_metric_path(metrics, "render.average_iou", numeric)
            metrics[key] = numeric
        elif key == "min_view_iou" and validation_mode == "render-iou":
            set_metric_path(metrics, "render.min_view_iou", numeric)
            metrics[key] = numeric
        elif key in {
            "failed_required_view_count",
            "missing_required_metric_count",
        }:
            set_metric_path(metrics, f"render.{key}", numeric)
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
                numeric = float(value)
                set_metric_path(metrics, f"render.{key}", numeric)
                set_metric_path(metrics, f"silhouette.{key}", numeric)
                if key == "mean_boundary_iou":
                    set_metric_path(metrics, "render.boundary_iou_mean", numeric)
                elif key == "min_boundary_iou":
                    set_metric_path(metrics, "render.boundary_iou_min", numeric)
                elif key == "mean_signed_distance_loss":
                    set_metric_path(metrics, "render.signed_distance_loss_mean", numeric)
    novel = payload.get("novel_view")
    if isinstance(novel, Mapping):
        for key in ("psnr", "ssim", "lpips", "mse", "image_count"):
            value = novel.get(key)
            if isinstance(value, (int, float)):
                set_metric_path(metrics, f"novel_view.{key}", float(value))
    novel_summary = payload.get("novel_view_summary")
    if isinstance(novel_summary, Mapping):
        value = novel_summary.get("passed")
        if isinstance(value, bool):
            set_metric_path(metrics, "novel_view.passed", 1.0 if value else 0.0)
    cost_report = payload.get("cost_report")
    if isinstance(cost_report, Mapping):
        value = cost_report.get("combined_total_wall_ms")
        if isinstance(value, (int, float)):
            numeric = float(value)
            set_metric_path(metrics, "cost.total_wall_ms", numeric)
            metrics["cost_combined_total_wall_ms"] = numeric
        validation = cost_report.get("validation")
        if isinstance(validation, Mapping):
            value = validation.get("total_wall_ms")
            if isinstance(value, (int, float)):
                numeric = float(value)
                set_metric_path(metrics, "cost.stage.validation.wall_ms", numeric)
                metrics["cost_validation_total_wall_ms"] = numeric
        backend_cost = cost_report.get("backend")
        if isinstance(backend_cost, Mapping):
            value = backend_cost.get("total_wall_ms")
            if isinstance(value, (int, float)):
                numeric = float(value)
                set_metric_path(metrics, "cost.stage.backend.wall_ms", numeric)
                metrics["cost_backend_total_wall_ms"] = numeric
    cost_gate = payload.get("cost_gate")
    if isinstance(cost_gate, Mapping):
        value = cost_gate.get("passed")
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
            set_metric_path(metrics, "cost.gate_passed", numeric)
            metrics["cost_gate_passed"] = numeric
    views = payload.get("views", {})
    if isinstance(views, Mapping):
        for view, view_payload in views.items():
            if not isinstance(view_payload, Mapping):
                continue
            value = view_payload.get("iou", view_payload.get("area_iou"))
            if isinstance(value, (int, float)):
                numeric = float(value)
                set_metric_path(metrics, f"render.per_view.{view}.area_iou", numeric)
                metrics[f"{view}_iou"] = numeric
            for source_key, target_key in (
                ("boundary_iou", f"render.per_view.{view}.boundary_iou"),
                ("signed_distance_loss", f"render.per_view.{view}.signed_distance_loss"),
            ):
                raw = view_payload.get(source_key)
                if isinstance(raw, (int, float)):
                    set_metric_path(metrics, target_key, float(raw))
        required_values = [
            metrics.get(f"{view}_iou")
            for view in ("front", "side", "top")
            if isinstance(metrics.get(f"{view}_iou"), (int, float))
        ]
        if len(required_values) == 3:
            min_required = float(min(required_values))
            set_metric_path(metrics, "render.min_view_iou", min_required)
            if validation_mode == "render-iou":
                metrics["min_view_iou"] = min_required
        set_render_aggregate_metrics(metrics)
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
        _add_backend_metric_result(metrics, metric_result)
    elif isinstance(backend_payload.get("metric_result"), Mapping):
        _add_backend_metric_result(metrics, backend_payload["metric_result"])
    for bundle in _evaluation_bundle_sequence(payload):
        for group in bundle.get("metric_groups", ()) or ():
            if not isinstance(group, Mapping):
                continue
            for metric in group.get("metrics", ()) or ():
                if not isinstance(metric, Mapping):
                    continue
                name = str(metric.get("name", ""))
                value = metric.get("value")
                if name and isinstance(value, (int, float, bool)):
                    numeric = float(value)
                    metrics[name.replace(".", "_")] = numeric
                    set_metric_path(metrics, name, numeric)
                    alias = namespace_metric_key(name)
                    if alias != name:
                        set_metric_path(metrics, alias, numeric)
                    if name == "editability.editable_reconstruction_index":
                        metrics["editability_score"] = numeric
                    elif name == "topology.score":
                        metrics["topology_score"] = numeric
    return metrics


def _add_backend_metric_result(
    metrics: Dict[str, Any],
    metric_result: Any,
) -> None:
    if not isinstance(metric_result, Mapping):
        return
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
            set_metric_path(metrics, namespace_metric_key(key), float(value))

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
