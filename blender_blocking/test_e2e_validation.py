"""
End-to-End Validation Test

Validates that 3D reconstruction accurately represents input reference images
by rendering the generated mesh and comparing to original inputs.

Usage:
    # In Blender (with GUI)
    Run this script in Blender's scripting workspace

    # Headless (for CI/CD)
    blender --background --python test_e2e_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Add both import roots used by this legacy codebase:
# - repo root for ``blender_blocking.*`` package imports
# - blender_blocking/ for existing direct imports like ``integration.*``
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from blender_blocking.verify_setup import configure_dependency_paths

configure_dependency_paths()

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

import numpy as np
from blender_blocking.config import BlockingConfig
from blender_blocking.config import CandidateConfig as ConfigCandidateConfig
from blender_blocking.config import RenderConfig
from blender_blocking.utils.generation_context import GenerationContext
from blender_blocking.utils.progress import progress_bar
from blender_blocking.integration.blender_ops.render_utils import (
    render_orthogonal_views,
)
from blender_blocking.integration.image_processing.image_loader import load_image
from blender_blocking.validation.silhouette_iou import (
    canonicalize_mask,
    compute_mask_iou,
    mask_from_image_array,
)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


ALL_RECONSTRUCTION_MODES = (
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "ensemble",
)
BACKEND_MODES = {
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "ensemble",
}
RENDER_IOU_MODES = {
    "legacy",
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
}
DEFAULT_ENSEMBLE_CANDIDATES = (
    "visual_hull_voxel",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
)


def _status_icon(ok: bool) -> str:
    return "✓" if ok else "✗"


def _print_rule(title: str = "", width: int = 72) -> None:
    if title:
        label = f" {title} "
        fill = max(0, width - len(label))
        left = fill // 2
        right = fill - left
        print("=" * left + label + "=" * right)
    else:
        print("=" * width)


def _print_section(title: str) -> None:
    print()
    _print_rule(title, width=72)


def _print_kv_table(rows: Iterable[Tuple[str, object]], *, title: str = "") -> None:
    rows = tuple((str(key), value) for key, value in rows if value is not None)
    if not rows:
        return
    if title:
        print(title)
    key_width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"  {key:<{key_width}} : {value}")


def _print_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    if not rows:
        return
    widths = {
        column: max(
            len(column),
            *[len(str(row.get(column, ""))) for row in rows],
        )
        for column in columns
    }
    header = "  " + "  ".join(f"{column:<{widths[column]}}" for column in columns)
    print(header)
    print("  " + "  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  "
            + "  ".join(
                f"{str(row.get(column, '')):<{widths[column]}}"
                for column in columns
            )
        )


def _print_result_table(rows: Sequence[Mapping[str, object]]) -> None:
    _print_table(rows, ("view", "iou", "threshold", "status", "intersection", "union"))


def _print_candidate_table(rows: Sequence[Mapping[str, object]]) -> None:
    _print_table(
        rows,
        ("candidate", "backend", "status", "score", "warnings", "errors", "artifact"),
    )


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _is_renderable_mesh(value: object) -> bool:
    return getattr(value, "type", None) == "MESH"


def _find_renderable_mesh(value: object) -> Optional[object]:
    if _is_renderable_mesh(value):
        return value
    if isinstance(value, Mapping):
        for key in ("profile_payload", "mesh", "object", "payload"):
            if key in value:
                found = _find_renderable_mesh(value[key])
                if found is not None:
                    return found
        for nested in value.values():
            found = _find_renderable_mesh(nested)
            if found is not None:
                return found
    if hasattr(value, "selected"):
        selected = getattr(value, "selected", None)
        found = _find_renderable_mesh(selected)
        if found is not None:
            return found
    if hasattr(value, "payload"):
        return _find_renderable_mesh(getattr(value, "payload"))
    return None


def _selected_result_dict(result: object) -> Dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "selected"):
        selected = getattr(result, "selected", None)
        return selected.to_dict() if hasattr(selected, "to_dict") else {}
    return result.to_dict() if hasattr(result, "to_dict") else {}


def _mesh_path_from_backend_result(result: object) -> Optional[Path]:
    data = _selected_result_dict(result)
    mesh_path = data.get("mesh_path") or data.get("artifacts", {}).get("mesh_obj")
    if not mesh_path:
        return None
    path = Path(str(mesh_path))
    return path if path.exists() else None


def _import_obj_for_render(path: Path) -> Optional[object]:
    before = {obj.name for obj in bpy.context.scene.objects}
    try:
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    except Exception as exc:
        print(f"Warning: failed to import OBJ for render validation: {exc}")
        return None
    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name not in before and getattr(obj, "type", None) == "MESH"
    ]
    if imported:
        return imported[0]
    active = bpy.context.active_object
    return active if _is_renderable_mesh(active) else None


def _candidate_status_payload(result: object) -> Tuple[str, Dict[str, Any]]:
    if result is None:
        return "missing", {}
    if hasattr(result, "selected") and hasattr(result, "candidates"):
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        selected = getattr(result, "selected", None)
        status = getattr(selected, "status", "missing") if selected else "missing"
        return str(status), data
    if hasattr(result, "status"):
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        return str(getattr(result, "status")), data
    return "unstructured", {"type": type(result).__name__, "repr": repr(result)}


def _print_backend_summary(result: object) -> bool:
    status, data = _candidate_status_payload(result)
    selected = data.get("selected") if isinstance(data, dict) else None
    if selected:
        summary_source = selected
    else:
        summary_source = data
    passed = status in {"success", "degraded"}

    _print_section("Backend Result")
    _print_kv_table(
        (
            ("status", f"{_status_icon(passed)} {status}"),
            ("backend", summary_source.get("backend_name") if summary_source else None),
            ("candidate", summary_source.get("candidate_id") if summary_source else None),
            ("mesh", summary_source.get("mesh_path") if summary_source else None),
            ("primitives", summary_source.get("primitive_path") if summary_source else None),
            ("volume", summary_source.get("volume_path") if summary_source else None),
            ("warnings", len(summary_source.get("warnings", [])) if summary_source else None),
            ("errors", len(summary_source.get("errors", [])) if summary_source else None),
        )
    )

    if data.get("candidates"):
        print("\nCandidates:")
        rows = []
        for candidate in data["candidates"]:
            metric_result = candidate.get("metric_result", {}) or {}
            mesh_path = candidate.get("mesh_path") or candidate.get("artifacts", {}).get(
                "mesh_obj", ""
            )
            rows.append(
                {
                    "candidate": candidate.get("candidate_id", ""),
                    "backend": candidate.get("backend_name", ""),
                    "status": candidate.get("status", ""),
                    "score": metric_result.get("scalar_score", ""),
                    "warnings": len(candidate.get("warnings", [])),
                    "errors": len(candidate.get("errors", [])),
                    "artifact": mesh_path,
                }
            )
        _print_candidate_table(rows)

    metrics = summary_source.get("metric_result", {}) if summary_source else {}
    extras = metrics.get("extras", {}) if isinstance(metrics, dict) else {}
    if metrics:
        _print_kv_table(
            (
                ("area_iou_mean", metrics.get("area_iou_mean")),
                ("area_iou_min", metrics.get("area_iou_min")),
                ("topology_score", metrics.get("topology_score")),
                ("editability_score", metrics.get("editability_score")),
                ("complexity_penalty", metrics.get("complexity_penalty")),
                ("elapsed_s", metrics.get("elapsed_s")),
                ("occupied_voxels", extras.get("occupied_voxels")),
                ("primitive_count", extras.get("primitive_count")),
                ("coverage_score", extras.get("coverage_score")),
                ("loss_total", extras.get("loss_total")),
            ),
            title="\nMetrics:",
        )
    return passed


class E2EValidator:
    """End-to-end validation for 3D reconstruction accuracy."""

    def __init__(
        self,
        iou_threshold: float = 0.7,
        view_thresholds: Optional[Dict[str, float]] = None,
        render_config: Optional[RenderConfig] = None,
        workflow_config: Optional[BlockingConfig] = None,
        config_label: str = "default",
        validation_mode: str = "auto",
        render_output_dir: Optional[Path] = None,
        result_json: Optional[Path] = None,
        run_id: Optional[str] = None,
        progress: bool = False,
    ) -> None:
        """
        Initialize validator.

        Args:
            iou_threshold: Minimum IoU score to pass (0-1)
        """
        self.iou_threshold = iou_threshold
        self.view_thresholds = view_thresholds or {}
        self.workflow_config = workflow_config or BlockingConfig()
        self.render_config = render_config or self.workflow_config.render_silhouette
        self.config_label = config_label
        self.validation_mode = validation_mode
        self.render_output_dir = (
            render_output_dir
            or Path(__file__).parent / "test_output" / "e2e_renders"
        )
        self.result_json = result_json
        self.run_id = run_id
        self.progress = progress
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
            validation_mode = "render-iou" if mode in RENDER_IOU_MODES else "backend-status"

        _print_rule("BLENDSLOP E2E VALIDATION", width=72)
        _print_kv_table(
            (
                ("mode", mode),
                ("validation", validation_mode),
                ("label", self.config_label),
                ("run_id", self.run_id or "<auto>"),
                ("slices", num_slices),
                ("render", f"{self.render_config.resolution[0]}x{self.render_config.resolution[1]} {self.render_config.engine}"),
            )
        )

        # Step 1: Generate 3D model
        _print_section("1/4 Reconstruct")
        print("Generating reconstruction from reference images...")
        from blender_blocking.main_integration import BlockingWorkflow

        context = GenerationContext(run_id=self.run_id) if self.run_id else None
        workflow = BlockingWorkflow(
            front_path=reference_paths.get("front"),
            side_path=reference_paths.get("side"),
            top_path=reference_paths.get("top"),
            config=self.workflow_config,
            context=context,
        )
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
            _, backend_payload = _candidate_status_payload(workflow.reconstruction_result)
            self.backend_result = backend_payload

        if render_mesh is None:
            if validation_mode == "render-iou":
                print("ERROR: Reconstruction did not produce a renderable Blender mesh")
                if workflow.reconstruction_result is not None:
                    _print_backend_summary(workflow.reconstruction_result)
                return False, {}
            if workflow.reconstruction_result is None:
                print("ERROR: Reconstruction returned no mesh and no backend result")
                return False, {}
            passed = _print_backend_summary(workflow.reconstruction_result)
            if self.result_json:
                _json_dump(self.result_json, backend_payload)
                print(f"\nSaved result JSON: {self.result_json}")
            return passed, {}

        print(f"{_status_icon(True)} Renderable mesh: {render_mesh.name}")
        if validation_mode == "backend-status":
            if workflow.reconstruction_result is not None:
                passed = _print_backend_summary(workflow.reconstruction_result)
                result_payload = backend_payload
            else:
                mesh_data = getattr(render_mesh, "data", None)
                vertex_count = len(mesh_data.vertices) if mesh_data is not None else None
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
            if self.result_json:
                _json_dump(self.result_json, result_payload)
                print(f"\nSaved result JSON: {self.result_json}")
            return passed, {}

        # Step 2: Setup rendering
        _print_section("2/4 Render Setup")
        self.setup_render_settings()
        print(f"{_status_icon(True)} Render settings configured")

        # Step 3: Render orthogonal views
        _print_section("3/4 Render Views")
        output_dir = self.render_output_dir
        views = ["front", "side", "top"]
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
        if base_name:
            filename_prefix = f"{base_name}_{technique}_{config_label}_"
        else:
            filename_prefix = f"{technique}_{config_label}_"
        render_progress = progress_bar(
            len(views), desc="render_views", enabled=self.progress
        )
        rendered_paths = render_orthogonal_views(
            str(output_dir),
            views=views,
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

        # Step 4: Compare with references
        _print_section("4/4 Compare Silhouettes")
        self.results = {}
        ious = []
        table_rows = []

        compare_progress = progress_bar(
            len(views), desc="compare_views", enabled=self.progress
        )
        for view in views:
            if view not in reference_paths or view not in rendered_paths:
                print(f"⚠ Skipping {view} (not available)")
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

            result = compute_mask_iou(ref_canon, render_canon)
            iou = result.iou

            threshold = self.view_thresholds.get(view, self.iou_threshold)

            if PIL_AVAILABLE:
                debug_dir = Path(__file__).parent / "test_output" / "debug_silhouettes"
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

            self.results[view] = {
                "iou": iou,
                "intersection": result.intersection,
                "union": result.union,
                "pixel_difference": float(
                    np.abs(ref_canon.astype(float) - render_canon.astype(float)).mean()
                ),
                "warnings": "; ".join(result.warnings) if result.warnings else "",
            }

            ious.append(iou)
            table_rows.append(
                {
                    "view": view,
                    "iou": f"{iou:.3f}",
                    "threshold": f"{threshold:.3f}",
                    "status": "PASS" if iou >= threshold else "FAIL",
                    "intersection": result.intersection,
                    "union": result.union,
                }
            )
            compare_progress.update(1)
        compare_progress.close()
        _print_result_table(table_rows)

        # Calculate overall result
        if ious:
            avg_iou = sum(ious) / len(ious)
            passed = avg_iou >= self.iou_threshold

            _print_section("Summary")
            _print_kv_table(
                (
                    ("average_iou", f"{avg_iou:.3f}"),
                    ("threshold", f"{self.iou_threshold:.3f}"),
                    ("result", f"{_status_icon(passed)} {'PASSED' if passed else 'FAILED'}"),
                    ("render_output", output_dir),
                )
            )
            if workflow.reconstruction_result is not None:
                print()
                _print_backend_summary(workflow.reconstruction_result)
            if self.result_json:
                payload_out = {
                    "mode": mode,
                    "validation_mode": validation_mode,
                    "passed": passed,
                    "average_iou": avg_iou,
                    "views": self.results,
                    "backend_result": backend_payload,
                    "rendered_paths": rendered_paths,
                }
                _json_dump(self.result_json, payload_out)
                print(f"\nSaved result JSON: {self.result_json}")

            return passed, self.results
        else:
            print("ERROR: No views to compare")
            return False, {}

    def print_detailed_results(self) -> None:
        """Print detailed comparison results."""
        if not self.results:
            print("No results to display")
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


def test_with_sample_images(
    *,
    num_slices: int = 120,
    iou_threshold: float = 0.7,
    view_thresholds: Optional[Dict[str, float]] = None,
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
    validation_mode: str = "auto",
    render_output_dir: Optional[Path] = None,
    result_json: Optional[Path] = None,
    run_id: Optional[str] = None,
    progress: bool = False,
) -> bool:
    """Test with built-in sample images."""
    base_dir = Path(__file__).parent
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
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        validation_mode=validation_mode,
        render_output_dir=render_output_dir,
        result_json=result_json,
        run_id=run_id,
        progress=progress,
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
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
    validation_mode: str = "auto",
    render_output_dir: Optional[Path] = None,
    result_json: Optional[Path] = None,
    run_id: Optional[str] = None,
    progress: bool = False,
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
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        validation_mode=validation_mode,
        render_output_dir=render_output_dir,
        result_json=result_json,
        run_id=run_id,
        progress=progress,
    )
    passed, results = validator.validate_reconstruction(
        reference_paths, num_slices=num_slices
    )

    if validator.results:
        validator.print_detailed_results()

    return passed


def _parse_resolution(value: str) -> Tuple[int, int]:
    if "x" in value:
        parts = value.lower().split("x", 1)
    elif "," in value:
        parts = value.split(",", 1)
    else:
        parts = [value]
    try:
        if len(parts) == 1:
            size = int(parts[0])
            return (size, size)
        width = int(parts[0])
        height = int(parts[1])
        return (width, height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "resolution must be N or WxH (e.g., 512 or 1024x1024)"
        ) from exc


def _parse_csv(value: str) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _parse_rgba(value: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("RGBA must be r,g,b,a")
    try:
        rgba = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGBA values must be numbers") from exc
    if any(channel < 0.0 or channel > 1.0 for channel in rgba):
        raise argparse.ArgumentTypeError("RGBA values must be in [0, 1]")
    return rgba  # type: ignore[return-value]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate blendslop reconstruction modes from reference silhouettes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode legacy --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode loft_profile --mesh-radial-segments 32 --profile-samples 120 --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode visual_hull_voxel --validation-mode backend-status --vh-resolution 32 --vh-backend chunked --vh-chunk-size 16 --vh-mesh-method points --no-progress
  blender --background --python blender_blocking/test_e2e_validation.py -- --reconstruction-mode ensemble --validation-mode backend-status --ensemble-candidates visual_hull_voxel,primitive_fit_refine,gaussian_ellipsoid_proxy --primitive-max 6 --gaussian-count 8 --no-progress
  python blender_blocking/test_e2e_validation.py --reconstruction-mode ensemble --print-config --dry-run

Modes:
  {", ".join(ALL_RECONSTRUCTION_MODES)}

Default ensemble:
  {", ".join(DEFAULT_ENSEMBLE_CANDIDATES)}
""",
    )

    core = parser.add_argument_group("core")
    core.add_argument(
        "--list-modes",
        action="store_true",
        help="Print available reconstruction modes and exit.",
    )
    core.add_argument(
        "--config-json",
        type=str,
        default=None,
        help='Inline JSON overrides for BlockingConfig (e.g. \'{"reconstruction": {"num_slices": 160}}\')',
    )
    core.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to JSON file with BlockingConfig overrides",
    )
    core.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config before running.",
    )
    core.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate CLI/config overrides, print config if requested, then exit.",
    )
    core.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Write validation/backend result JSON to this path.",
    )
    core.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Deterministic run id used for manifests and backend artifacts.",
    )
    core.add_argument(
        "--config-label",
        type=str,
        default=None,
        help="Label included in render filenames. Defaults to config path/inline/default.",
    )
    core.add_argument(
        "--front",
        type=str,
        default=None,
        help="Path to front view image (PNG/JPG)",
    )
    core.add_argument(
        "--side",
        type=str,
        default=None,
        help="Path to side view image (PNG/JPG)",
    )
    core.add_argument(
        "--top",
        type=str,
        default=None,
        help="Path to top view image (PNG/JPG)",
    )
    core.add_argument(
        "--validation-mode",
        choices=("auto", "render-iou", "backend-status"),
        default="auto",
        help="auto renders mesh modes and checks backend status for artifact-only modes.",
    )
    core.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Average IoU threshold for render-iou validation.",
    )
    core.add_argument("--front-threshold", type=float, default=None)
    core.add_argument("--side-threshold", type=float, default=None)
    core.add_argument("--top-threshold", type=float, default=None)
    core.add_argument(
        "--reconstruction-mode",
        choices=ALL_RECONSTRUCTION_MODES,
        default="legacy",
        help="Reconstruction mode to validate.",
    )
    core.add_argument(
        "--num-slices",
        type=int,
        default=120,
        help="Number of slices used for legacy/loft/profile workflows.",
    )
    core.add_argument(
        "--unit-scale",
        type=float,
        default=None,
        help="World units per pixel.",
    )

    render = parser.add_argument_group("render")
    render.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(512, 512),
        help="Render resolution (N or WxH, e.g., 512 or 1024x1024)",
    )
    render.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Render samples (EEVEE only)",
    )
    render.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE", "WORKBENCH"),
        default="BLENDER_EEVEE",
        help="Render engine",
    )
    render.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Camera framing margin as fraction of bounds",
    )
    render.add_argument("--color-mode", choices=("BW", "RGBA"), default=None)
    render.add_argument("--transparent-bg", action=argparse.BooleanOptionalAction, default=None)
    render.add_argument("--force-material", action=argparse.BooleanOptionalAction, default=None)
    render.add_argument("--background-color", type=_parse_rgba, default=None, help="RGBA as r,g,b,a")
    render.add_argument("--silhouette-color", type=_parse_rgba, default=None, help="RGBA as r,g,b,a")
    render.add_argument("--camera-distance-factor", type=float, default=None)
    render.add_argument("--party-mode", action=argparse.BooleanOptionalAction, default=None)
    render.add_argument(
        "--render-output-dir",
        type=Path,
        default=None,
        help="Directory for rendered validation views.",
    )

    profile = parser.add_argument_group("profile and loft")
    profile.add_argument(
        "--profile-samples",
        type=int,
        default=None,
        help="Profile sample count (profile_sampling.num_samples)",
    )
    profile.add_argument(
        "--profile-sample-policy",
        choices=("endpoints", "cell_centers"),
        default=None,
        help="Profile sampling policy",
    )
    profile.add_argument(
        "--profile-fill-strategy",
        choices=("interp_linear", "interp_nearest", "constant"),
        default=None,
        help="Profile fill strategy",
    )
    profile.add_argument(
        "--profile-smoothing-window",
        type=int,
        default=None,
        help="Median filter window for profile smoothing",
    )
    profile.add_argument(
        "--mesh-radial-segments",
        type=int,
        default=None,
        help="Loft mesh radial segments",
    )
    profile.add_argument(
        "--mesh-cap-mode",
        choices=("fan", "none", "ngon"),
        default=None,
        help="Loft mesh cap mode",
    )
    profile.add_argument(
        "--mesh-min-radius",
        type=float,
        default=None,
        help="Minimum radius (world units) for loft mesh slices",
    )
    profile.add_argument(
        "--mesh-merge-threshold",
        type=float,
        default=None,
        help="Merge threshold for loft mesh (world units)",
    )
    profile.add_argument(
        "--mesh-recalc-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recalculate loft mesh normals",
    )
    profile.add_argument(
        "--mesh-shade-smooth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable smooth shading on loft mesh",
    )
    profile.add_argument(
        "--mesh-weld-degenerate-rings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Weld degenerate rings in loft mesh",
    )
    profile.add_argument("--mesh-adaptive-radial-segments", action=argparse.BooleanOptionalAction, default=None)
    profile.add_argument("--mesh-min-adaptive-radial-segments", type=int, default=None)
    profile.add_argument("--mesh-max-adaptive-radial-segments", type=int, default=None)
    profile.add_argument("--mesh-topology-strict", action=argparse.BooleanOptionalAction, default=None)
    profile.add_argument("--mesh-research-allow-low-radial-segments", action=argparse.BooleanOptionalAction, default=None)

    silhouette = parser.add_argument_group("silhouette extraction")
    silhouette.add_argument("--ref-prefer-alpha", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--render-prefer-alpha", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--ref-polarity", choices=("auto", "dark_foreground", "light_foreground", "alpha_foreground"), default=None)
    silhouette.add_argument("--render-polarity", choices=("auto", "dark_foreground", "light_foreground", "alpha_foreground"), default=None)
    silhouette.add_argument("--ref-invert-policy", choices=("auto", "invert", "no_invert"), default=None)
    silhouette.add_argument("--render-invert-policy", choices=("auto", "invert", "no_invert"), default=None)
    silhouette.add_argument("--ref-alpha-threshold", type=int, default=None)
    silhouette.add_argument("--render-alpha-threshold", type=int, default=None)
    silhouette.add_argument("--ref-alpha-min-coverage", type=float, default=None)
    silhouette.add_argument("--render-alpha-min-coverage", type=float, default=None)
    silhouette.add_argument("--ref-gray-threshold", type=int, default=None)
    silhouette.add_argument("--render-gray-threshold", type=int, default=None)
    silhouette.add_argument("--ref-morph-close", type=int, default=None)
    silhouette.add_argument("--render-morph-close", type=int, default=None)
    silhouette.add_argument("--ref-morph-open", type=int, default=None)
    silhouette.add_argument("--render-morph-open", type=int, default=None)
    silhouette.add_argument("--ref-fill-holes", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--render-fill-holes", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--ref-largest-component", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--render-largest-component", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--ref-min-area-frac", type=float, default=None)
    silhouette.add_argument("--render-min-area-frac", type=float, default=None)
    silhouette.add_argument("--ref-max-area-frac", type=float, default=None)
    silhouette.add_argument("--render-max-area-frac", type=float, default=None)
    silhouette.add_argument("--ref-max-border-contact-frac", type=float, default=None)
    silhouette.add_argument("--render-max-border-contact-frac", type=float, default=None)
    silhouette.add_argument("--ref-min-component-area-px", type=int, default=None)
    silhouette.add_argument("--render-min-component-area-px", type=int, default=None)
    silhouette.add_argument("--ref-candidate-scoring", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--render-candidate-scoring", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--ref-emit-uncertainty", action=argparse.BooleanOptionalAction, default=None)
    silhouette.add_argument("--render-emit-uncertainty", action=argparse.BooleanOptionalAction, default=None)

    canonical = parser.add_argument_group("canonicalization and IoU")
    canonical.add_argument("--canonical-output-size", type=int, default=None)
    canonical.add_argument("--canonical-padding-frac", type=float, default=None)
    canonical.add_argument("--canonical-anchor", choices=("center", "bottom_center"), default=None)
    canonical.add_argument("--canonical-interp", choices=("nearest",), default=None)
    canonical.add_argument("--canonical-cache", action=argparse.BooleanOptionalAction, default=None)
    canonical.add_argument("--canonical-digest", choices=("sha256",), default=None)
    canonical.add_argument("--canonical-fill-holes", action=argparse.BooleanOptionalAction, default=None)
    canonical.add_argument("--canonical-largest-component", action=argparse.BooleanOptionalAction, default=None)

    join = parser.add_argument_group("mesh join and silhouette intersection")
    join.add_argument("--mesh-join-mode", choices=("auto", "boolean", "voxel", "simple"), default=None)
    join.add_argument("--boolean-solver", choices=("auto", "EXACT", "MANIFOLD", "FLOAT", "FAST"), default=None)
    join.add_argument("--allow-degraded-simple-join", action=argparse.BooleanOptionalAction, default=None)
    join.add_argument("--record-join-attempts", action=argparse.BooleanOptionalAction, default=None)
    join.add_argument("--balanced-boolean-tree", action=argparse.BooleanOptionalAction, default=None)
    join.add_argument("--silhouette-extrude-distance", type=float, default=None)
    join.add_argument("--silhouette-contour-mode", choices=("external", "ccomp", "tree", "hierarchy"), default=None)
    join.add_argument("--silhouette-largest-component", action=argparse.BooleanOptionalAction, default=None)

    hull = parser.add_argument_group("visual hull and volume")
    hull.add_argument("--vh-backend", choices=("dense", "chunked", "sparse_hash", "openvdb"), default=None)
    hull.add_argument("--vh-resolution", type=int, default=None)
    hull.add_argument("--vh-max-resolution", type=int, default=None)
    hull.add_argument("--vh-chunk-size", type=int, default=None)
    hull.add_argument("--vh-adaptive-max-depth", type=int, default=None)
    hull.add_argument("--vh-boundary-refine", action=argparse.BooleanOptionalAction, default=None)
    hull.add_argument("--vh-mesh-method", choices=("marching_cubes", "lewiner", "dual_contouring", "points"), default=None)
    hull.add_argument("--vh-postprocess", choices=("none", "poisson", "screened_poisson"), default=None)
    hull.add_argument("--vh-memory-budget-mb", type=int, default=None)
    hull.add_argument("--vh-occupancy-threshold", type=float, default=None)
    hull.add_argument("--vh-uncertainty-aggregation", choices=("min", "product", "logit_sum"), default=None)
    hull.add_argument("--volume-backend", choices=("dense", "chunked", "sparse_hash", "openvdb"), default=None)
    hull.add_argument("--volume-sparse-chunk-size", type=int, default=None)
    hull.add_argument("--volume-serialization", choices=("npz",), default=None)
    hull.add_argument("--export-openvdb", action=argparse.BooleanOptionalAction, default=None)

    primitive = parser.add_argument_group("primitive fitting")
    primitive.add_argument("--primitive-families", type=_parse_csv, default=None)
    primitive.add_argument("--primitive-loss-weights-json", type=str, default=None)
    primitive.add_argument("--primitive-target-points", type=int, default=None)
    primitive.add_argument("--primitive-min", type=int, default=None)
    primitive.add_argument("--primitive-max", type=int, default=None)
    primitive.add_argument("--primitive-steps", type=int, default=None)
    primitive.add_argument("--primitive-checkpoint-cadence", type=int, default=None)
    primitive.add_argument("--primitive-fail-on-regression", action=argparse.BooleanOptionalAction, default=None)

    gaussian = parser.add_argument_group("gaussian and ellipsoid proxy")
    gaussian.add_argument("--gaussian-count", type=int, default=None)
    gaussian.add_argument("--gaussian-initialization", choices=("farthest_point", "kmeans", "grid"), default=None)
    gaussian.add_argument("--gaussian-min-radius", type=float, default=None)
    gaussian.add_argument("--gaussian-max-radius", type=float, default=None)
    gaussian.add_argument("--gaussian-opacity-min", type=float, default=None)
    gaussian.add_argument("--gaussian-opacity-max", type=float, default=None)
    gaussian.add_argument("--gaussian-renderer", choices=("cpu_projected_ellipse", "gpu_splat"), default=None)
    gaussian.add_argument("--gaussian-export-mesh-proxy", action=argparse.BooleanOptionalAction, default=None)

    diff = parser.add_argument_group("differentiable refinement")
    diff.add_argument("--diff-backend", choices=("cpu_soft_silhouette", "blender_finite_difference", "nvdiffrast"), default=None)
    diff.add_argument("--diff-optional-policy", choices=("skip", "fail"), default=None)
    diff.add_argument("--diff-gradient-mode", choices=("finite_difference", "backend"), default=None)
    diff.add_argument("--diff-epsilon", type=float, default=None)
    diff.add_argument("--diff-loss-weights-json", type=str, default=None)

    ensemble = parser.add_argument_group("ensemble")
    ensemble.add_argument("--ensemble-candidates", type=_parse_csv, default=None, help="Comma-separated backend list.")
    ensemble.add_argument("--ensemble-policy", choices=("best_score", "quality_first", "editability_first", "fast_preview", "pareto"), default=None)
    ensemble.add_argument("--ensemble-max-parallel", type=int, default=None)
    ensemble.add_argument("--ensemble-timeout", type=float, default=None)
    ensemble.add_argument("--ensemble-total-timeout", type=float, default=None)
    ensemble.add_argument("--ensemble-keep-artifacts", action=argparse.BooleanOptionalAction, default=None)
    ensemble.add_argument("--ensemble-fail-if-no-required-views", action=argparse.BooleanOptionalAction, default=None)

    misc = parser.add_argument_group("constraints and quality")
    misc.add_argument("--constraint-file", action="append", default=None)
    misc.add_argument("--fail-on-hard-constraints", action=argparse.BooleanOptionalAction, default=None)
    misc.add_argument("--use-constraints-for-scoring", action=argparse.BooleanOptionalAction, default=None)
    misc.add_argument("--quality-budget-json", type=str, default=None)
    misc.add_argument("--quality-compare-baseline", type=str, default=None)
    misc.add_argument("--quality-fail-on-regression", action=argparse.BooleanOptionalAction, default=None)
    misc.add_argument("--environment-compatibility", choices=("warn", "strict", "ignore"), default=None)
    misc.add_argument("--synthetic-suite", type=str, default=None)
    misc.add_argument("--synthetic-seed", type=int, default=None)
    misc.add_argument("--synthetic-output-root", type=str, default=None)
    misc.add_argument("--synthetic-commit-small-fixtures-only", action=argparse.BooleanOptionalAction, default=None)
    misc.add_argument("--synthetic-keep-heavy-artifacts", action=argparse.BooleanOptionalAction, default=None)
    misc.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress bars",
    )
    return parser.parse_args(argv)


def _coerce_override_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _candidate_configs(names: Sequence[str]) -> Tuple[ConfigCandidateConfig, ...]:
    return tuple(
        ConfigCandidateConfig(backend_name=name, candidate_id=f"{name}_{index:02d}")
        for index, name in enumerate(names)
    )


def _apply_dataclass_overrides(target: object, values: Mapping[str, Any]) -> None:
    valid_fields = {field.name for field in fields(target)}
    for key, value in values.items():
        if key not in valid_fields:
            raise ValueError(f"unknown config field {type(target).__name__}.{key}")
        current = getattr(target, key)
        if key == "candidates" and isinstance(value, (list, tuple)):
            candidates = []
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    candidates.append(ConfigCandidateConfig(**dict(item)))
                else:
                    candidates.append(
                        ConfigCandidateConfig(
                            backend_name=str(item),
                            candidate_id=f"{item}_{index:02d}",
                        )
                    )
            setattr(target, key, tuple(candidates))
        elif is_dataclass(current) and isinstance(value, Mapping):
            _apply_dataclass_overrides(current, value)
        else:
            setattr(target, key, _coerce_override_value(current, value))


def _apply_overrides(cfg: BlockingConfig, overrides: Dict[str, Any]) -> None:
    """Apply JSON/config-file overrides to any BlockingConfig group."""
    if not overrides:
        return
    for group_name, values in overrides.items():
        if not hasattr(cfg, group_name):
            raise ValueError(f"unknown config group {group_name!r}")
        target = getattr(cfg, group_name)
        if is_dataclass(target) and isinstance(values, Mapping):
            _apply_dataclass_overrides(target, values)
        else:
            setattr(cfg, group_name, values)


def _set_if_not_none(target: object, name: str, value: Any) -> None:
    if value is not None:
        setattr(target, name, value)


def _apply_silhouette_cli_args(cfg: BlockingConfig, args: argparse.Namespace) -> None:
    for prefix, target in (
        ("ref", cfg.silhouette_extract_ref),
        ("render", cfg.silhouette_extract_render),
    ):
        _set_if_not_none(target, "prefer_alpha", getattr(args, f"{prefix}_prefer_alpha"))
        _set_if_not_none(target, "polarity", getattr(args, f"{prefix}_polarity"))
        _set_if_not_none(
            target, "invert_policy", getattr(args, f"{prefix}_invert_policy")
        )
        _set_if_not_none(
            target, "alpha_threshold", getattr(args, f"{prefix}_alpha_threshold")
        )
        _set_if_not_none(
            target,
            "alpha_min_coverage",
            getattr(args, f"{prefix}_alpha_min_coverage"),
        )
        _set_if_not_none(
            target, "gray_threshold", getattr(args, f"{prefix}_gray_threshold")
        )
        _set_if_not_none(
            target, "morph_close_px", getattr(args, f"{prefix}_morph_close")
        )
        _set_if_not_none(
            target, "morph_open_px", getattr(args, f"{prefix}_morph_open")
        )
        _set_if_not_none(target, "fill_holes", getattr(args, f"{prefix}_fill_holes"))
        _set_if_not_none(
            target,
            "largest_component_only",
            getattr(args, f"{prefix}_largest_component"),
        )
        _set_if_not_none(
            target, "min_area_frac", getattr(args, f"{prefix}_min_area_frac")
        )
        _set_if_not_none(
            target, "max_area_frac", getattr(args, f"{prefix}_max_area_frac")
        )
        _set_if_not_none(
            target,
            "max_border_contact_frac",
            getattr(args, f"{prefix}_max_border_contact_frac"),
        )
        _set_if_not_none(
            target,
            "min_component_area_px",
            getattr(args, f"{prefix}_min_component_area_px"),
        )
        _set_if_not_none(
            target, "candidate_scoring", getattr(args, f"{prefix}_candidate_scoring")
        )
        _set_if_not_none(
            target, "emit_uncertainty", getattr(args, f"{prefix}_emit_uncertainty")
        )


def _apply_cli_args(cfg: BlockingConfig, args: argparse.Namespace) -> None:
    cfg.reconstruction.reconstruction_mode = args.reconstruction_mode
    cfg.reconstruction.num_slices = int(args.num_slices)
    _set_if_not_none(cfg.reconstruction, "unit_scale", args.unit_scale)

    cfg.render_silhouette.resolution = args.resolution
    cfg.render_silhouette.engine = args.engine
    cfg.render_silhouette.samples = int(args.samples)
    cfg.render_silhouette.margin_frac = float(args.margin)
    _set_if_not_none(cfg.render_silhouette, "color_mode", args.color_mode)
    _set_if_not_none(cfg.render_silhouette, "transparent_bg", args.transparent_bg)
    _set_if_not_none(cfg.render_silhouette, "force_material", args.force_material)
    _set_if_not_none(cfg.render_silhouette, "background_color", args.background_color)
    _set_if_not_none(cfg.render_silhouette, "silhouette_color", args.silhouette_color)
    _set_if_not_none(
        cfg.render_silhouette,
        "camera_distance_factor",
        args.camera_distance_factor,
    )
    _set_if_not_none(cfg.render_silhouette, "party_mode", args.party_mode)

    _set_if_not_none(cfg.profile_sampling, "num_samples", args.profile_samples)
    _set_if_not_none(cfg.profile_sampling, "sample_policy", args.profile_sample_policy)
    _set_if_not_none(cfg.profile_sampling, "fill_strategy", args.profile_fill_strategy)
    _set_if_not_none(
        cfg.profile_sampling, "smoothing_window", args.profile_smoothing_window
    )

    _set_if_not_none(cfg.mesh_from_profile, "radial_segments", args.mesh_radial_segments)
    _set_if_not_none(cfg.mesh_from_profile, "cap_mode", args.mesh_cap_mode)
    _set_if_not_none(cfg.mesh_from_profile, "min_radius_u", args.mesh_min_radius)
    _set_if_not_none(
        cfg.mesh_from_profile, "merge_threshold_u", args.mesh_merge_threshold
    )
    _set_if_not_none(cfg.mesh_from_profile, "recalc_normals", args.mesh_recalc_normals)
    _set_if_not_none(cfg.mesh_from_profile, "shade_smooth", args.mesh_shade_smooth)
    _set_if_not_none(
        cfg.mesh_from_profile,
        "weld_degenerate_rings",
        args.mesh_weld_degenerate_rings,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "adaptive_radial_segments",
        args.mesh_adaptive_radial_segments,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "min_adaptive_radial_segments",
        args.mesh_min_adaptive_radial_segments,
    )
    _set_if_not_none(
        cfg.mesh_from_profile,
        "max_adaptive_radial_segments",
        args.mesh_max_adaptive_radial_segments,
    )
    _set_if_not_none(cfg.mesh_from_profile, "topology_strict", args.mesh_topology_strict)
    _set_if_not_none(
        cfg.mesh_from_profile,
        "research_allow_low_radial_segments",
        args.mesh_research_allow_low_radial_segments,
    )

    _apply_silhouette_cli_args(cfg, args)

    _set_if_not_none(cfg.canonicalize, "output_size", args.canonical_output_size)
    _set_if_not_none(cfg.canonicalize, "padding_frac", args.canonical_padding_frac)
    _set_if_not_none(cfg.canonicalize, "anchor", args.canonical_anchor)
    _set_if_not_none(cfg.canonicalize, "interp", args.canonical_interp)
    _set_if_not_none(cfg.canonicalize, "use_cache", args.canonical_cache)
    _set_if_not_none(cfg.canonicalize, "digest_algorithm", args.canonical_digest)
    _set_if_not_none(cfg.canonicalize, "fill_holes", args.canonical_fill_holes)
    _set_if_not_none(
        cfg.canonicalize, "largest_component_only", args.canonical_largest_component
    )

    _set_if_not_none(cfg.mesh_join, "mode", args.mesh_join_mode)
    _set_if_not_none(cfg.mesh_join, "boolean_solver", args.boolean_solver)
    _set_if_not_none(
        cfg.mesh_join, "allow_degraded_simple_join", args.allow_degraded_simple_join
    )
    _set_if_not_none(cfg.mesh_join, "record_attempts", args.record_join_attempts)
    _set_if_not_none(
        cfg.mesh_join, "balanced_boolean_tree", args.balanced_boolean_tree
    )
    _set_if_not_none(
        cfg.silhouette_intersection,
        "extrude_distance",
        args.silhouette_extrude_distance,
    )
    _set_if_not_none(
        cfg.silhouette_intersection, "contour_mode", args.silhouette_contour_mode
    )
    _set_if_not_none(
        cfg.silhouette_intersection,
        "largest_component_only",
        args.silhouette_largest_component,
    )

    _set_if_not_none(cfg.visual_hull, "backend", args.vh_backend)
    _set_if_not_none(cfg.visual_hull, "resolution", args.vh_resolution)
    _set_if_not_none(cfg.visual_hull, "max_resolution", args.vh_max_resolution)
    _set_if_not_none(cfg.visual_hull, "chunk_size", args.vh_chunk_size)
    _set_if_not_none(cfg.visual_hull, "adaptive_max_depth", args.vh_adaptive_max_depth)
    _set_if_not_none(cfg.visual_hull, "boundary_refine", args.vh_boundary_refine)
    _set_if_not_none(cfg.visual_hull, "mesh_method", args.vh_mesh_method)
    _set_if_not_none(cfg.visual_hull, "postprocess", args.vh_postprocess)
    _set_if_not_none(cfg.visual_hull, "memory_budget_mb", args.vh_memory_budget_mb)
    _set_if_not_none(
        cfg.visual_hull, "occupancy_threshold", args.vh_occupancy_threshold
    )
    _set_if_not_none(
        cfg.visual_hull,
        "uncertainty_aggregation",
        args.vh_uncertainty_aggregation,
    )
    _set_if_not_none(cfg.volume, "backend", args.volume_backend)
    _set_if_not_none(cfg.volume, "sparse_chunk_size", args.volume_sparse_chunk_size)
    _set_if_not_none(cfg.volume, "serialization", args.volume_serialization)
    _set_if_not_none(cfg.volume, "export_openvdb", args.export_openvdb)

    _set_if_not_none(cfg.primitive_fit, "primitive_families", args.primitive_families)
    if args.primitive_loss_weights_json:
        cfg.primitive_fit.loss_weights = json.loads(args.primitive_loss_weights_json)
    _set_if_not_none(cfg.primitive_fit, "target_point_count", args.primitive_target_points)
    _set_if_not_none(cfg.primitive_fit, "min_primitives", args.primitive_min)
    _set_if_not_none(cfg.primitive_fit, "max_primitives", args.primitive_max)
    _set_if_not_none(cfg.primitive_fit, "optimization_steps", args.primitive_steps)
    _set_if_not_none(
        cfg.primitive_fit, "checkpoint_cadence", args.primitive_checkpoint_cadence
    )
    _set_if_not_none(
        cfg.primitive_fit, "fail_on_regression", args.primitive_fail_on_regression
    )

    _set_if_not_none(cfg.gaussian_ellipsoid, "primitive_count", args.gaussian_count)
    _set_if_not_none(
        cfg.gaussian_ellipsoid, "initialization", args.gaussian_initialization
    )
    _set_if_not_none(cfg.gaussian_ellipsoid, "min_radius", args.gaussian_min_radius)
    _set_if_not_none(cfg.gaussian_ellipsoid, "max_radius", args.gaussian_max_radius)
    _set_if_not_none(cfg.gaussian_ellipsoid, "opacity_min", args.gaussian_opacity_min)
    _set_if_not_none(cfg.gaussian_ellipsoid, "opacity_max", args.gaussian_opacity_max)
    _set_if_not_none(cfg.gaussian_ellipsoid, "renderer", args.gaussian_renderer)
    _set_if_not_none(
        cfg.gaussian_ellipsoid,
        "export_mesh_proxy",
        args.gaussian_export_mesh_proxy,
    )

    _set_if_not_none(cfg.differentiable_render, "backend", args.diff_backend)
    _set_if_not_none(
        cfg.differentiable_render,
        "optional_dependency_policy",
        args.diff_optional_policy,
    )
    _set_if_not_none(cfg.differentiable_render, "gradient_mode", args.diff_gradient_mode)
    _set_if_not_none(
        cfg.differentiable_render, "finite_difference_epsilon", args.diff_epsilon
    )
    if args.diff_loss_weights_json:
        cfg.differentiable_render.loss_weights = json.loads(args.diff_loss_weights_json)

    if args.ensemble_candidates:
        cfg.ensemble.candidates = _candidate_configs(args.ensemble_candidates)
    _set_if_not_none(cfg.ensemble, "selection_policy", args.ensemble_policy)
    _set_if_not_none(cfg.ensemble, "max_parallel_candidates", args.ensemble_max_parallel)
    _set_if_not_none(cfg.ensemble, "per_candidate_timeout_s", args.ensemble_timeout)
    _set_if_not_none(cfg.ensemble, "total_timeout_s", args.ensemble_total_timeout)
    _set_if_not_none(cfg.ensemble, "keep_all_artifacts", args.ensemble_keep_artifacts)
    _set_if_not_none(
        cfg.ensemble,
        "fail_if_no_candidate_passes_required_views",
        args.ensemble_fail_if_no_required_views,
    )

    if args.constraint_file:
        cfg.constraints.constraint_files = tuple(args.constraint_file)
    _set_if_not_none(
        cfg.constraints,
        "fail_on_unsatisfied_hard_constraints",
        args.fail_on_hard_constraints,
    )
    _set_if_not_none(
        cfg.constraints,
        "use_constraints_for_candidate_scoring",
        args.use_constraints_for_scoring,
    )
    _set_if_not_none(cfg.quality_budget, "budget_json", args.quality_budget_json)
    _set_if_not_none(
        cfg.quality_budget, "compare_baseline", args.quality_compare_baseline
    )
    _set_if_not_none(
        cfg.quality_budget, "fail_on_regression", args.quality_fail_on_regression
    )
    _set_if_not_none(
        cfg.quality_budget,
        "environment_compatibility",
        args.environment_compatibility,
    )

    _set_if_not_none(cfg.synthetic_factory, "suite", args.synthetic_suite)
    _set_if_not_none(cfg.synthetic_factory, "seed", args.synthetic_seed)
    _set_if_not_none(
        cfg.synthetic_factory, "output_root", args.synthetic_output_root
    )
    _set_if_not_none(
        cfg.synthetic_factory,
        "commit_small_fixtures_only",
        args.synthetic_commit_small_fixtures_only,
    )
    _set_if_not_none(
        cfg.synthetic_factory,
        "keep_heavy_artifacts",
        args.synthetic_keep_heavy_artifacts,
    )


def _derive_config_label(args: argparse.Namespace) -> str:
    if args.config_path:
        stem = Path(args.config_path).stem
        for mode in ALL_RECONSTRUCTION_MODES:
            prefix = f"{mode}-"
            if stem.startswith(prefix):
                return stem[len(prefix) :]
        parts = stem.split("-")
        return parts[-1] if len(parts) > 1 else stem
    if args.config_json:
        return "inline"
    return "default"


if __name__ == "__main__":
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    elif not BLENDER_AVAILABLE:
        argv = sys.argv[1:]
    args = _parse_args(argv)

    if args.list_modes:
        _print_rule("RECONSTRUCTION MODES", width=72)
        for mode in ALL_RECONSTRUCTION_MODES:
            kind = "render-iou" if mode in RENDER_IOU_MODES else "backend-status"
            surface = "backend artifact" if mode in BACKEND_MODES else "Blender mesh"
            print(f"  {mode:<28} {kind:<16} {surface}")
        print(f"\nDefault ensemble candidates: {', '.join(DEFAULT_ENSEMBLE_CANDIDATES)}")
        sys.exit(0)

    workflow_config = BlockingConfig()
    _apply_cli_args(workflow_config, args)

    overrides: Dict[str, Any] = {}
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
    if args.config_json:
        inline = json.loads(args.config_json)
        if overrides:
            overrides.update(inline)
        else:
            overrides = inline
    _apply_overrides(workflow_config, overrides)
    workflow_config.validate()
    render_config = workflow_config.render_silhouette
    config_label = args.config_label or _derive_config_label(args)

    if args.print_config:
        _print_rule("RESOLVED CONFIG", width=72)
        print(json.dumps(workflow_config.to_dict(), indent=2, sort_keys=True))

    if args.dry_run:
        print("\nDry run complete: config resolved and validated.")
        sys.exit(0)

    if not BLENDER_AVAILABLE:
        print("ERROR: This validation CLI must be run inside Blender.")
        print("Run: blender --background --python blender_blocking/test_e2e_validation.py -- [options]")
        sys.exit(1)

    custom_paths = [args.front, args.side, args.top]
    if any(custom_paths) and not all(custom_paths):
        print("ERROR: --front, --side, and --top must be provided together.")
        sys.exit(2)

    thresholds = {
        key: value
        for key, value in {
            "front": args.front_threshold,
            "side": args.side_threshold,
            "top": args.top_threshold,
        }.items()
        if value is not None
    }

    if all(custom_paths):
        success = test_with_custom_images(
            args.front,
            args.side,
            args.top,
            num_slices=args.num_slices,
            iou_threshold=args.iou_threshold,
            view_thresholds=thresholds,
            render_config=render_config,
            workflow_config=workflow_config,
            config_label=config_label,
            validation_mode=args.validation_mode,
            render_output_dir=args.render_output_dir,
            result_json=args.result_json,
            run_id=args.run_id,
            progress=args.progress,
        )
    else:
        # Run test with sample images
        validator = E2EValidator(
            iou_threshold=args.iou_threshold,
            view_thresholds=thresholds,
            render_config=render_config,
            workflow_config=workflow_config,
            config_label=config_label,
            validation_mode=args.validation_mode,
            render_output_dir=args.render_output_dir,
            result_json=args.result_json,
            run_id=args.run_id,
            progress=args.progress,
        )
        base_dir = Path(__file__).parent
        test_images_dir = base_dir / "test_images"
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
                sys.exit(1)
        success, _ = validator.validate_reconstruction(
            {
                "front": str(test_images_dir / "vase_front.png"),
                "side": str(test_images_dir / "vase_side.png"),
                "top": str(test_images_dir / "vase_top.png"),
            },
            num_slices=args.num_slices,
        )
        if validator.results:
            validator.print_detailed_results()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
