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
from blender_blocking.e2e.backend_status import _candidate_status_payload, _print_backend_summary
from blender_blocking.e2e.console import _format_metric, _print_kv_table, _print_result_table, _print_rule, _print_section, _print_table, _status_icon
from blender_blocking.e2e.cost import _cost_gate_report, _cost_payload
from blender_blocking.e2e.novel_view import _aggregate_novel_reports, _load_novel_pair, _novel_view_gate
from blender_blocking.e2e.payloads import _evaluation_outputs_from_payload, _find_renderable_mesh, _json_dump, _mesh_path_from_backend_result, _ordered_unique, _render_filename_prefix
from blender_blocking.e2e.rendering import _import_obj_for_render


class E2EValidator:
    """End-to-end validation for 3D reconstruction accuracy."""

    def __init__(
        self,
        iou_threshold: float = 0.7,
        view_thresholds: Optional[Dict[str, float]] = None,
        boundary_iou_threshold: Optional[float] = None,
        signed_distance_loss_threshold: Optional[float] = None,
        render_config: Optional[RenderConfig] = None,
        workflow_config: Optional[BlockingConfig] = None,
        config_label: str = "default",
        validation_mode: str = "auto",
        render_output_dir: Optional[Path] = None,
        debug_output_dir: Optional[Path] = None,
        artifact_root: Optional[Path] = None,
        result_json: Optional[Path] = None,
        run_id: Optional[str] = None,
        novel_view_reference_paths: Optional[Mapping[str, str | Path]] = None,
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
        debug_artifact_policy: str = "all",
    ) -> None:
        """
        Initialize validator.

        Args:
            iou_threshold: Minimum IoU score to pass (0-1)
        """
        self.iou_threshold = iou_threshold
        self.view_thresholds = view_thresholds or {}
        self.boundary_iou_threshold = boundary_iou_threshold
        self.signed_distance_loss_threshold = signed_distance_loss_threshold
        self.workflow_config = workflow_config or BlockingConfig()
        self.render_config = render_config or self.workflow_config.render_silhouette
        self.config_label = config_label
        self.validation_mode = validation_mode
        default_render_dir = TEMP_OUTPUT_ROOT / "e2e" / "renders"
        self.render_output_dir = Path(render_output_dir or default_render_dir).resolve(
            strict=False
        )
        self.debug_output_dir = (
            Path(debug_output_dir).resolve(strict=False)
            if debug_output_dir is not None
            else None
        )
        self.artifact_root = (
            Path(artifact_root).resolve(strict=False)
            if artifact_root is not None
            else None
        )
        self.result_json = (
            Path(result_json).resolve(strict=False) if result_json is not None else None
        )
        self.run_id = run_id
        self.novel_view_reference_paths = {
            str(view): str(path)
            for view, path in (novel_view_reference_paths or {}).items()
        }
        self.novel_view_names = tuple(str(view) for view in novel_view_names)
        self.novel_compute_ssim = bool(novel_compute_ssim)
        self.novel_compute_lpips = bool(novel_compute_lpips)
        self.novel_psnr_threshold = novel_psnr_threshold
        self.novel_ssim_threshold = novel_ssim_threshold
        self.novel_lpips_threshold = novel_lpips_threshold
        self.cost_report_json = (
            Path(cost_report_json).resolve(strict=False)
            if cost_report_json is not None
            else None
        )
        self.cost_fail_max_wall_ms = cost_fail_max_wall_ms
        self.cost_fail_max_backend_wall_ms = cost_fail_max_backend_wall_ms
        self.cost_recorder = CostRecorder(track_memory=cost_track_memory)
        self.progress = progress
        self.debug_artifact_policy = str(debug_artifact_policy or "all")
        self.results = {}
        self.backend_result: Optional[Dict[str, Any]] = None

    def setup_render_settings(self) -> None:
        """Configure Blender for clean silhouette rendering."""
        scene = bpy.context.scene

        config = self.render_config

        # Transparent background for clean silhouettes
        scene.render.film_transparent = config.transparent_bg
        scene.render.image_settings.color_mode = config.color_mode

        # Resolution
        scene.render.resolution_x = int(config.resolution[0])
        scene.render.resolution_y = int(config.resolution[1])
        scene.render.resolution_percentage = 100

        # Fast rendering (we only need silhouettes)
        scene.render.engine = config.engine
        if config.engine == "BLENDER_EEVEE" and hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = int(config.samples)

    def extract_silhouette(self, image_path: str) -> np.ndarray:
        """
        Extract binary silhouette from image.

        Args:
            image_path: Path to image file

        Returns:
            Binary numpy array (0 or 255)
        """
        img = load_image(image_path)
        mask = mask_from_image_array(
            img, extract_config=self.workflow_config.silhouette_extract_ref
        )
        return (mask.astype(np.uint8) * 255).astype(np.uint8)

    def validate_reconstruction(
        self, reference_paths: Dict[str, str], num_slices: int = 12
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """
        Run full validation loop.

        Args:
            reference_paths: Dict with 'front', 'side', 'top' image paths
            num_slices: Number of slices for reconstruction

        Returns:
            Tuple of (passed: bool, results: dict)
        """
        mode = self.workflow_config.reconstruction.reconstruction_mode
        validation_mode = self.validation_mode
        if validation_mode == "auto":
            validation_mode = (
                "render-iou" if mode in RENDER_IOU_MODES else "backend-status"
            )

        _print_rule("BLENDSLOP E2E VALIDATION", width=72)
        _print_kv_table(
            (
                ("mode", mode),
                ("validation", validation_mode),
                ("label", self.config_label),
                ("run_id", self.run_id or "<auto>"),
                ("slices", num_slices),
                (
                    "render",
                    f"{self.render_config.resolution[0]}x{self.render_config.resolution[1]} {self.render_config.engine}",
                ),
            )
        )

        # Step 1: Generate 3D model
        _print_section("1/4 Reconstruct")
        print("Generating reconstruction from reference images...")
        from blender_blocking.main_integration import BlockingWorkflow

        context = (
            GenerationContext(run_id=self.run_id)
            if self.run_id
            else GenerationContext()
        )
        if self.artifact_root is not None:
            context.artifact_root = str(self.artifact_root)
        workflow = BlockingWorkflow(
            front_path=reference_paths.get("front"),
            side_path=reference_paths.get("side"),
            top_path=reference_paths.get("top"),
            config=self.workflow_config,
            context=context,
        )
        with self.cost_recorder.stage(
            "reconstruction_workflow",
            work_units={
                "reference_views": float(len(reference_paths)),
                "num_slices": float(num_slices),
            },
        ):
            payload = workflow.run_full_workflow(num_slices=num_slices)
        render_mesh = _find_renderable_mesh(payload) or _find_renderable_mesh(
            workflow.reconstruction_result
        )
        if render_mesh is None and workflow.reconstruction_result is not None:
            mesh_path = _mesh_path_from_backend_result(workflow.reconstruction_result)
            if mesh_path is not None:
                render_mesh = _import_obj_for_render(mesh_path)

        backend_payload: Dict[str, Any] = {}
        if workflow.reconstruction_result is not None:
            _, backend_payload = _candidate_status_payload(
                workflow.reconstruction_result
            )
            self.backend_result = backend_payload

        if render_mesh is None:
            if validation_mode in {"render-iou", "novel-view"}:
                print("ERROR: Reconstruction did not produce a renderable Blender mesh")
                if workflow.reconstruction_result is not None:
                    _print_backend_summary(workflow.reconstruction_result)
                failure_code = (
                    "missing_mesh_artifact"
                    if validation_mode == "novel-view"
                    else "missing_renderable_mesh"
                )
                result_payload, _cost_passed = self._attach_cost_outputs(
                    {
                        "mode": mode,
                        "validation_mode": validation_mode,
                        "status": "failed",
                        "error": failure_code,
                        "failure_code": failure_code,
                        "expected_artifact_key": "mesh_obj",
                        "backend_result": backend_payload,
                    },
                    backend_payload,
                )
                if self.result_json:
                    _json_dump(self.result_json, result_payload)
                    print(f"\nSaved result JSON: {self.result_json}")
                return False, {}
            if workflow.reconstruction_result is None:
                print("ERROR: Reconstruction returned no mesh and no backend result")
                return False, {}
            passed = _print_backend_summary(workflow.reconstruction_result)
            result_payload, cost_passed = self._attach_cost_outputs(
                backend_payload,
                backend_payload,
            )
            passed = passed and cost_passed
            if self.result_json:
                _json_dump(self.result_json, result_payload)
                print(f"\nSaved result JSON: {self.result_json}")
            return passed, {}

        print(f"{_status_icon(True)} Renderable mesh: {render_mesh.name}")
        if validation_mode == "backend-status":
            if workflow.reconstruction_result is not None:
                passed = _print_backend_summary(workflow.reconstruction_result)
                result_payload = backend_payload
            else:
                mesh_data = getattr(render_mesh, "data", None)
                vertex_count = (
                    len(mesh_data.vertices) if mesh_data is not None else None
                )
                face_count = len(mesh_data.polygons) if mesh_data is not None else None
                result_payload = {
                    "mode": mode,
                    "validation_mode": validation_mode,
                    "status": "success",
                    "mesh_name": render_mesh.name,
                    "vertices": vertex_count,
                    "faces": face_count,
                }
                _print_section("Mesh Result")
                _print_kv_table(
                    (
                        ("status", f"{_status_icon(True)} success"),
                        ("backend", mode),
                        ("mesh", render_mesh.name),
                        ("vertices", vertex_count),
                        ("faces", face_count),
                    )
                )
                passed = True
            result_payload, cost_passed = self._attach_cost_outputs(
                result_payload,
                backend_payload,
            )
            passed = passed and cost_passed
            if self.result_json:
                _json_dump(self.result_json, result_payload)
                print(f"\nSaved result JSON: {self.result_json}")
            return passed, {}

        # Step 2: Setup rendering
        _print_section("2/4 Render Setup")
        with self.cost_recorder.stage("render_setup"):
            self.setup_render_settings()
        print(f"{_status_icon(True)} Render settings configured")

        # Step 3: Render orthogonal views
        _print_section("3/4 Render Views")
        output_dir = self.render_output_dir
        silhouette_views = ["front", "side", "top"]
        render_views = list(
            _ordered_unique(
                tuple(silhouette_views)
                + tuple(self.novel_view_names)
                + tuple(self.novel_view_reference_paths.keys())
            )
        )
        base_name = None
        for view in ("front", "side", "top"):
            path = reference_paths.get(view)
            if path:
                stem = Path(path).stem
                for suffix in ("_front", "_side", "_top"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                base_name = stem
                break
        technique = (
            self.workflow_config.reconstruction.reconstruction_mode
            if self.workflow_config
            else "legacy"
        )
        config_label = self.config_label or "default"
        filename_prefix = _render_filename_prefix(base_name, technique, config_label)
        render_progress = progress_bar(
            len(render_views), desc="render_views", enabled=self.progress
        )
        with self.cost_recorder.stage(
            "render_views",
            work_units={"views": float(len(render_views))},
        ):
            rendered_paths = render_orthogonal_views(
                str(output_dir),
                views=render_views,
                target_objects=[render_mesh] if render_mesh else None,
                resolution=self.render_config.resolution,
                margin_frac=self.render_config.margin_frac,
                transparent_bg=self.render_config.transparent_bg,
                color_mode=self.render_config.color_mode,
                force_material=self.render_config.force_material,
                background_color=self.render_config.background_color,
                silhouette_color=self.render_config.silhouette_color,
                camera_distance_factor=self.render_config.camera_distance_factor,
                party_mode=self.render_config.party_mode,
                filename_prefix=filename_prefix,
                start_index=1,
                progress_callback=render_progress.update,
            )
        render_progress.close()

        if not rendered_paths:
            print("ERROR: Failed to render views")
            return False, {}

        for view, path in rendered_paths.items():
            print(f"{_status_icon(True)} Rendered {view:<5} {path}")

        if validation_mode == "novel-view":
            passed, novel_payload = self._validate_novel_views(
                mode=mode,
                rendered_paths=rendered_paths,
                reference_paths=reference_paths,
                backend_payload=backend_payload,
                render_output_dir=output_dir,
            )
            return passed, novel_payload.get("views", {})

        # Step 4: Compare with references
        _print_section("4/4 Compare Silhouettes")
        self.results = {}
        table_rows = []
        silhouette_gate = SilhouetteGateConfig(
            min_area_iou=self.iou_threshold,
            per_view_min_area_iou=self.view_thresholds,
            min_boundary_iou=self.boundary_iou_threshold,
            max_signed_distance_loss=self.signed_distance_loss_threshold,
            required_views=tuple(silhouette_views),
        )

        compare_progress = progress_bar(
            len(silhouette_views), desc="compare_views", enabled=self.progress
        )
        for view in silhouette_views:
            if view not in reference_paths or view not in rendered_paths:
                reason = "missing_reference_or_render"
                print(f"{_status_icon(False)} {view} {reason}")
                payload = missing_silhouette_view(
                    view,
                    reason=reason,
                    config=silhouette_gate,
                    required=True,
                )
                self.results[view] = payload
                table_rows.append(
                    {
                        "view": view,
                        "iou": "0.000",
                        "threshold": f"{silhouette_gate.threshold_for_view(view):.3f}",
                        "status": "FAIL",
                        "intersection": 0,
                        "union": 0,
                    }
                )
                compare_progress.update(1)
                continue

            ref_image = load_image(reference_paths[view])
            render_image = load_image(rendered_paths[view])

            ref_mask = mask_from_image_array(
                ref_image, extract_config=self.workflow_config.silhouette_extract_ref
            )
            render_mask = mask_from_image_array(
                render_image,
                extract_config=self.workflow_config.silhouette_extract_render,
            )

            canonical = self.workflow_config.canonicalize
            anchor = "center" if view == "top" else canonical.anchor
            ref_canon = canonicalize_mask(
                ref_mask,
                output_size=canonical.output_size,
                padding_frac=canonical.padding_frac,
                anchor=anchor,
            )
            render_canon = canonicalize_mask(
                render_mask,
                output_size=canonical.output_size,
                padding_frac=canonical.padding_frac,
                anchor=anchor,
            )

            threshold = self.view_thresholds.get(view, self.iou_threshold)
            payload = evaluate_silhouette_pair(
                ref_canon,
                render_canon,
                view=view,
                config=silhouette_gate,
                required=True,
            )

            if PIL_AVAILABLE and self._should_write_debug_artifacts(payload):
                debug_dir = (
                    self.debug_output_dir
                    or TEMP_OUTPUT_ROOT / "e2e" / "dbg"
                )
                debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(ref_mask.astype(np.uint8) * 255).save(
                    debug_dir / f"{view}_ref_silhouette.png"
                )
                Image.fromarray(render_mask.astype(np.uint8) * 255).save(
                    debug_dir / f"{view}_render_silhouette.png"
                )
                Image.fromarray(ref_canon.astype(np.uint8) * 255).save(
                    debug_dir / f"{view}_ref_canon.png"
                )
                Image.fromarray(render_canon.astype(np.uint8) * 255).save(
                    debug_dir / f"{view}_render_canon.png"
                )
                diff = np.logical_xor(ref_canon, render_canon).astype(np.uint8) * 255
                Image.fromarray(diff).save(debug_dir / f"{view}_diff.png")

            self.results[view] = payload

            table_rows.append(
                {
                    "view": view,
                    "iou": f"{float(payload['area_iou']):.3f}",
                    "threshold": f"{threshold:.3f}",
                    "status": "PASS" if payload["passed"] else "FAIL",
                    "intersection": payload["intersection"],
                    "union": payload["union"],
                }
            )
            compare_progress.update(1)
        compare_progress.close()
        _print_result_table(table_rows)

        # Calculate overall result
        if self.results:
            silhouette_summary = summarize_silhouette_views(
                self.results,
                config=silhouette_gate,
            )
            avg_iou = float(silhouette_summary["average_iou"])
            min_iou = float(silhouette_summary["min_view_iou"])
            passed = bool(silhouette_summary["passed"])

            _print_section("Summary")
            _print_kv_table(
                (
                    ("average_iou", f"{avg_iou:.3f}"),
                    ("min_view_iou", f"{min_iou:.3f}"),
                    ("threshold", f"{self.iou_threshold:.3f}"),
                    (
                        "required_views",
                        "PASS"
                        if silhouette_summary["required_views_passed"]
                        else (
                            "FAIL "
                            + ",".join(
                                str(view)
                                for view in silhouette_summary[
                                    "failed_required_views"
                                ]
                            )
                        ),
                    ),
                    (
                        "result",
                        f"{_status_icon(passed)} {'PASSED' if passed else 'FAILED'}",
                    ),
                    ("render_output", output_dir),
                )
            )
            if workflow.reconstruction_result is not None:
                print()
                _print_backend_summary(workflow.reconstruction_result)
            payload_out = {
                "mode": mode,
                "validation_mode": validation_mode,
                "passed": passed,
                "average_iou": avg_iou,
                "min_view_iou": min_iou,
                "required_views_passed": silhouette_summary[
                    "required_views_passed"
                ],
                "failed_required_view_count": silhouette_summary[
                    "failed_required_view_count"
                ],
                "missing_required_metric_count": silhouette_summary[
                    "missing_required_metric_count"
                ],
                "silhouette_summary": silhouette_summary,
                "views": self.results,
                "backend_result": backend_payload,
                "rendered_paths": rendered_paths,
            }
            payload_out.update(_evaluation_outputs_from_payload(backend_payload))
            payload_out, cost_passed = self._attach_cost_outputs(
                payload_out,
                backend_payload,
            )
            passed = passed and cost_passed
            payload_out["passed"] = passed
            if self.result_json:
                _json_dump(self.result_json, payload_out)
                print(f"\nSaved result JSON: {self.result_json}")

            return passed, self.results
        else:
            print("ERROR: No views to compare")
            return False, {}

    def _should_write_debug_artifacts(self, payload: Mapping[str, Any]) -> bool:
        policy = self.debug_artifact_policy
        if policy == "none":
            return False
        if policy == "all":
            return True
        if policy in {"failures", "top"}:
            return not bool(payload.get("passed"))
        return True

    def _validate_novel_views(
        self,
        *,
        mode: str,
        rendered_paths: Mapping[str, str],
        reference_paths: Mapping[str, str],
        backend_payload: Mapping[str, Any],
        render_output_dir: Path,
    ) -> Tuple[bool, Dict[str, Any]]:
        _print_section("4/4 Compare Novel/Image Views")
        from blender_blocking.evaluation.novel_view import image_pair_report

        references = self._novel_reference_map(reference_paths)
        pair_reports: Dict[str, Dict[str, Any]] = {}
        missing_views: list[str] = [
            view for view in self.novel_view_names if view not in references
        ]
        table_rows: list[dict[str, object]] = []

        for view, reference_path in references.items():
            rendered_path = rendered_paths.get(view)
            if not rendered_path:
                missing_views.append(view)
                continue
            try:
                ref_image, render_image, warnings = _load_novel_pair(
                    reference_path,
                    rendered_path,
                )
                report = image_pair_report(
                    ref_image,
                    render_image,
                    compute_ssim=self.novel_compute_ssim,
                    compute_lpips=self.novel_compute_lpips,
                ).to_dict()
            except ValueError as exc:
                report = {
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "mse": None,
                    "image_count": 0,
                    "warnings": (),
                    "failure_code": "image_size_mismatch",
                    "error": str(exc),
                }
                warnings = ()
            except Exception as exc:
                report = {
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "mse": None,
                    "image_count": 0,
                    "warnings": (),
                    "failure_code": "render_failed",
                    "error": str(exc),
                }
                warnings = ()
            if warnings:
                report["warnings"] = list(report.get("warnings", [])) + list(warnings)
            gate = _novel_view_gate(
                report,
                psnr_threshold=self.novel_psnr_threshold,
                ssim_threshold=self.novel_ssim_threshold,
                lpips_threshold=self.novel_lpips_threshold,
            )
            report["gate"] = gate
            if not gate["passed"] and "failure_code" not in report:
                report["failure_code"] = _novel_failure_code(report, gate)
            report["reference_path"] = str(reference_path)
            report["rendered_path"] = str(rendered_path)
            pair_reports[view] = report
            table_rows.append(
                {
                    "view": view,
                    "psnr": _format_metric(report.get("psnr"), precision=2),
                    "ssim": _format_metric(report.get("ssim"), precision=3),
                    "lpips": _format_metric(report.get("lpips"), precision=3),
                    "status": "PASS" if gate["passed"] else "FAIL",
                }
            )

        if table_rows:
            _print_table(table_rows, ("view", "psnr", "ssim", "lpips", "status"))

        summary = _aggregate_novel_reports(
            pair_reports,
            missing_views=missing_views,
            psnr_threshold=self.novel_psnr_threshold,
            ssim_threshold=self.novel_ssim_threshold,
            lpips_threshold=self.novel_lpips_threshold,
        )
        passed = bool(summary["passed"])
        _print_section("Summary")
        _print_kv_table(
            (
                ("image_pairs", summary["image_count"]),
                ("missing_views", ",".join(missing_views) if missing_views else None),
                ("psnr", _format_metric(summary.get("psnr"), precision=2)),
                ("ssim", _format_metric(summary.get("ssim"), precision=3)),
                ("lpips", _format_metric(summary.get("lpips"), precision=3)),
                (
                    "result",
                    f"{_status_icon(passed)} {'PASSED' if passed else 'FAILED'}",
                ),
                ("render_output", render_output_dir),
            )
        )
        if backend_payload:
            _print_backend_summary(backend_payload)

        payload_out = {
            "mode": mode,
            "validation_mode": "novel-view",
            "passed": passed,
            "novel_view": {
                "psnr": summary.get("psnr"),
                "ssim": summary.get("ssim"),
                "lpips": summary.get("lpips"),
                "mse": summary.get("mse"),
                "image_count": summary.get("image_count", 0),
                "warnings": summary.get("warnings", []),
                "dependency_state": summary.get("dependency_state", {}),
            },
            "novel_view_summary": summary,
            "views": pair_reports,
            "missing_views": missing_views,
            "backend_result": dict(backend_payload),
            "rendered_paths": dict(rendered_paths),
        }
        payload_out.update(_evaluation_outputs_from_payload(backend_payload))
        payload_out, cost_passed = self._attach_cost_outputs(
            payload_out,
            backend_payload,
        )
        passed = passed and cost_passed
        payload_out["passed"] = passed
        if self.result_json:
            _json_dump(self.result_json, payload_out)
            print(f"\nSaved result JSON: {self.result_json}")
        self.results = pair_reports
        return passed, payload_out

    def _novel_reference_map(
        self,
        reference_paths: Mapping[str, str],
    ) -> Dict[str, str]:
        novel_references: Dict[str, str] = dict(self.novel_view_reference_paths)
        reference_dir: Path | None = None
        for path in reference_paths.values():
            if path:
                reference_dir = Path(path).parent
                break
        for view in self.novel_view_names:
            if view in novel_references:
                continue
            if reference_dir is None:
                continue
            inferred = reference_dir / f"{view}.png"
            if inferred.exists():
                novel_references[view] = str(inferred)
        if novel_references or self.novel_view_names:
            return novel_references
        return {
            view: str(path)
            for view, path in reference_paths.items()
            if view in {"front", "side", "top"} and path
        }

    def _attach_cost_outputs(
        self,
        payload: Mapping[str, Any],
        backend_payload: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], bool]:
        validation_report = self.cost_recorder.report().to_dict()
        cost_report = _cost_payload(validation_report, backend_payload)
        gate = _cost_gate_report(
            cost_report,
            max_wall_ms=self.cost_fail_max_wall_ms,
            max_backend_wall_ms=self.cost_fail_max_backend_wall_ms,
        )
        merged = dict(payload)
        merged["cost_report"] = cost_report
        merged["cost_gate"] = gate
        if self.cost_report_json is not None:
            _json_dump(self.cost_report_json, cost_report)
            print(f"\nSaved cost report JSON: {self.cost_report_json}")
        if not gate["passed"]:
            print("\nCost gate: FAIL")
            for failure in gate["failures"]:
                print(f"  - {failure}")
        elif self.cost_fail_max_wall_ms is not None or self.cost_fail_max_backend_wall_ms is not None:
            print("\nCost gate: PASS")
        return merged, bool(gate["passed"])

    def print_detailed_results(self) -> None:
        """Print detailed comparison results."""
        if not self.results:
            print("No results to display")
            return

        first = next(iter(self.results.values()))
        if isinstance(first, Mapping) and "psnr" in first:
            print("\nDetailed Novel/Image Metrics:")
            _print_table(
                [
                    {
                        "view": view,
                        "psnr": _format_metric(metrics.get("psnr"), precision=2),
                        "ssim": _format_metric(metrics.get("ssim"), precision=3),
                        "lpips": _format_metric(metrics.get("lpips"), precision=3),
                        "mse": _format_metric(metrics.get("mse"), precision=5),
                    }
                    for view, metrics in self.results.items()
                ],
                ("view", "psnr", "ssim", "lpips", "mse"),
            )
            return

        print("\nDetailed Results:")
        print("-" * 60)
        print(
            f"{'View':<10} {'IoU':>8} {'Intersection':>12} {'Union':>10} {'PixDiff':>10}"
        )
        print("-" * 60)

        for view, metrics in self.results.items():
            print(
                f"{view:<10} "
                f"{metrics['iou']:>8.3f} "
                f"{metrics['intersection']:>12d} "
                f"{metrics['union']:>10d} "
                f"{metrics['pixel_difference']:>10.2f}"
            )

        print("-" * 60)


def _novel_failure_code(
    report: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    failures = " ".join(str(item) for item in gate.get("failures", ()) or ())
    dependency_state = report.get("dependency_state", {})
    lpips_state = (
        dependency_state.get("lpips")
        if isinstance(dependency_state, Mapping)
        else None
    )
    if "LPIPS" in failures and isinstance(lpips_state, Mapping):
        dependency = lpips_state.get("lpips")
        if isinstance(dependency, Mapping) and not dependency.get("available", False):
            return "lpips_dependency_missing"
    if "missing" in failures:
        return "metric_below_threshold"
    return "metric_below_threshold"

def test_with_sample_images(
    *,
    num_slices: int = 120,
    iou_threshold: float = 0.7,
    view_thresholds: Optional[Dict[str, float]] = None,
    boundary_iou_threshold: Optional[float] = None,
    signed_distance_loss_threshold: Optional[float] = None,
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
    validation_mode: str = "auto",
    render_output_dir: Optional[Path] = None,
    debug_output_dir: Optional[Path] = None,
    artifact_root: Optional[Path] = None,
    result_json: Optional[Path] = None,
    run_id: Optional[str] = None,
    novel_view_reference_paths: Optional[Mapping[str, str | Path]] = None,
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
    debug_artifact_policy: str = "all",
) -> bool:
    """Test with built-in sample images."""
    base_dir = BLENDER_BLOCKING_ROOT
    test_images_dir = base_dir / "test_images"

    # Check if test images exist
    if not test_images_dir.exists():
        print("Creating test images...")
        import subprocess

        result = subprocess.run(
            [sys.executable, str(base_dir / "create_test_images.py")],
            capture_output=True,
            text=True,
            cwd=base_dir,
        )
        if result.returncode != 0:
            print("ERROR: Failed to create test images")
            print(result.stderr)
            return False

    # Use vase test images
    reference_paths = {
        "front": str(test_images_dir / "vase_front.png"),
        "side": str(test_images_dir / "vase_side.png"),
        "top": str(test_images_dir / "vase_top.png"),
    }

    # Run validation with many slices to capture profile details
    validator = E2EValidator(
        iou_threshold=iou_threshold,
        view_thresholds=view_thresholds,
        boundary_iou_threshold=boundary_iou_threshold,
        signed_distance_loss_threshold=signed_distance_loss_threshold,
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        validation_mode=validation_mode,
        render_output_dir=render_output_dir,
        debug_output_dir=debug_output_dir,
        artifact_root=artifact_root,
        result_json=result_json,
        run_id=run_id,
        novel_view_reference_paths=novel_view_reference_paths,
        novel_view_names=novel_view_names,
        novel_compute_ssim=novel_compute_ssim,
        novel_compute_lpips=novel_compute_lpips,
        novel_psnr_threshold=novel_psnr_threshold,
        novel_ssim_threshold=novel_ssim_threshold,
        novel_lpips_threshold=novel_lpips_threshold,
        cost_report_json=cost_report_json,
        cost_track_memory=cost_track_memory,
        cost_fail_max_wall_ms=cost_fail_max_wall_ms,
        cost_fail_max_backend_wall_ms=cost_fail_max_backend_wall_ms,
        progress=progress,
        debug_artifact_policy=debug_artifact_policy,
    )
    passed, results = validator.validate_reconstruction(
        reference_paths, num_slices=num_slices
    )

    # Print detailed results
    if validator.results:
        validator.print_detailed_results()

    return passed

def test_with_custom_images(
    front: str,
    side: str,
    top: str,
    *,
    num_slices: int = 12,
    iou_threshold: float = 0.7,
    view_thresholds: Optional[Dict[str, float]] = None,
    boundary_iou_threshold: Optional[float] = None,
    signed_distance_loss_threshold: Optional[float] = None,
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
    validation_mode: str = "auto",
    render_output_dir: Optional[Path] = None,
    debug_output_dir: Optional[Path] = None,
    artifact_root: Optional[Path] = None,
    result_json: Optional[Path] = None,
    run_id: Optional[str] = None,
    novel_view_reference_paths: Optional[Mapping[str, str | Path]] = None,
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
    debug_artifact_policy: str = "all",
) -> bool:
    """
    Test with custom reference images.

    Args:
        front: Path to front view image
        side: Path to side view image
        top: Path to top view image

    Returns:
        bool: Test passed
    """
    reference_paths = {"front": front, "side": side, "top": top}

    validator = E2EValidator(
        iou_threshold=iou_threshold,
        view_thresholds=view_thresholds,
        boundary_iou_threshold=boundary_iou_threshold,
        signed_distance_loss_threshold=signed_distance_loss_threshold,
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        validation_mode=validation_mode,
        render_output_dir=render_output_dir,
        debug_output_dir=debug_output_dir,
        artifact_root=artifact_root,
        result_json=result_json,
        run_id=run_id,
        novel_view_reference_paths=novel_view_reference_paths,
        novel_view_names=novel_view_names,
        novel_compute_ssim=novel_compute_ssim,
        novel_compute_lpips=novel_compute_lpips,
        novel_psnr_threshold=novel_psnr_threshold,
        novel_ssim_threshold=novel_ssim_threshold,
        novel_lpips_threshold=novel_lpips_threshold,
        cost_report_json=cost_report_json,
        cost_track_memory=cost_track_memory,
        cost_fail_max_wall_ms=cost_fail_max_wall_ms,
        cost_fail_max_backend_wall_ms=cost_fail_max_backend_wall_ms,
        progress=progress,
        debug_artifact_policy=debug_artifact_policy,
    )
    passed, results = validator.validate_reconstruction(
        reference_paths, num_slices=num_slices
    )

    if validator.results:
        validator.print_detailed_results()

    return passed
