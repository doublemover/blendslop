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
from typing import Any, Dict, Optional, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blender_blocking.verify_setup import configure_dependency_paths

configure_dependency_paths()

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("ERROR: This test must be run inside Blender")
    sys.exit(1)

import numpy as np
from blender_blocking.config import BlockingConfig
from blender_blocking.config import RenderConfig
from blender_blocking.main_integration import BlockingWorkflow
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


class E2EValidator:
    """End-to-end validation for 3D reconstruction accuracy."""

    def __init__(
        self,
        iou_threshold: float = 0.7,
        view_thresholds: Optional[Dict[str, float]] = None,
        render_config: Optional[RenderConfig] = None,
        workflow_config: Optional[BlockingConfig] = None,
        config_label: str = "default",
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
        self.progress = progress
        self.results = {}

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
        print("=" * 60)
        print("E2E VALIDATION TEST")
        print("=" * 60)

        # Step 1: Generate 3D model
        print("\n[1/4] Generating 3D model from reference images...")
        workflow = BlockingWorkflow(
            front_path=reference_paths.get("front"),
            side_path=reference_paths.get("side"),
            top_path=reference_paths.get("top"),
            config=self.workflow_config,
        )
        mesh = workflow.run_full_workflow(num_slices=num_slices)

        if not mesh:
            print("ERROR: Failed to generate 3D model")
            return False, {}

        print(f"✓ Generated mesh: {mesh.name}")

        # Step 2: Setup rendering
        print("\n[2/4] Setting up render configuration...")
        self.setup_render_settings()
        print("✓ Render settings configured")

        # Step 3: Render orthogonal views
        print("\n[3/4] Rendering orthogonal views...")
        output_dir = Path(__file__).parent / "test_output" / "e2e_renders"
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
            target_objects=[mesh] if mesh else None,
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
            print(f"✓ Rendered {view}: {path}")

        # Step 4: Compare with references
        print("\n[4/4] Comparing rendered views to reference images...")
        self.results = {}
        ious = []

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

            anchor = "center" if view == "top" else "bottom_center"
            ref_canon = canonicalize_mask(
                ref_mask, output_size=256, padding_frac=0.1, anchor=anchor
            )
            render_canon = canonicalize_mask(
                render_mask, output_size=256, padding_frac=0.1, anchor=anchor
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
            print(f"  {view:8s} IoU: {iou:.3f}", end="")
            print(f"  {'✓ PASS' if iou >= threshold else '✗ FAIL'}")
            compare_progress.update(1)
        compare_progress.close()

        # Calculate overall result
        if ious:
            avg_iou = sum(ious) / len(ious)
            passed = avg_iou >= self.iou_threshold

            print("\n" + "=" * 60)
            print(f"Average IoU: {avg_iou:.3f}")
            print(f"Threshold:   {self.iou_threshold:.3f}")
            print(f"Result:      {'✓ PASSED' if passed else '✗ FAILED'}")
            print("=" * 60)

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
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
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
        iou_threshold=0.7,
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        progress=progress,
    )
    passed, results = validator.validate_reconstruction(
        reference_paths, num_slices=num_slices
    )

    # Print detailed results
    validator.print_detailed_results()

    return passed


def test_with_custom_images(
    front: str,
    side: str,
    top: str,
    *,
    num_slices: int = 12,
    render_config: Optional[RenderConfig] = None,
    workflow_config: Optional[BlockingConfig] = None,
    config_label: str = "default",
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
        iou_threshold=0.7,
        render_config=render_config,
        workflow_config=workflow_config,
        config_label=config_label,
        progress=progress,
    )
    passed, results = validator.validate_reconstruction(
        reference_paths, num_slices=num_slices
    )

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


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E validation test settings")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help='Inline JSON overrides for BlockingConfig (e.g. \'{"reconstruction": {"num_slices": 160}}\')',
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to JSON file with BlockingConfig overrides",
    )
    parser.add_argument(
        "--front",
        type=str,
        default=None,
        help="Path to front view image (PNG/JPG)",
    )
    parser.add_argument(
        "--side",
        type=str,
        default=None,
        help="Path to side view image (PNG/JPG)",
    )
    parser.add_argument(
        "--top",
        type=str,
        default=None,
        help="Path to top view image (PNG/JPG)",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(512, 512),
        help="Render resolution (N or WxH, e.g., 512 or 1024x1024)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Render samples (EEVEE only)",
    )
    parser.add_argument(
        "--engine",
        choices=("BLENDER_EEVEE", "WORKBENCH"),
        default="BLENDER_EEVEE",
        help="Render engine",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.08,
        help="Camera framing margin as fraction of bounds",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=120,
        help="Number of slices used for reconstruction",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=("legacy", "loft_profile", "silhouette_intersection"),
        default="legacy",
        help="Reconstruction mode used for the blockout",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=None,
        help="World units per pixel (reconstruction.unit_scale)",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=None,
        help="Profile sample count (profile_sampling.num_samples)",
    )
    parser.add_argument(
        "--profile-sample-policy",
        choices=("endpoints", "cell_centers"),
        default=None,
        help="Profile sampling policy",
    )
    parser.add_argument(
        "--profile-fill-strategy",
        choices=("interp_linear", "interp_nearest", "constant"),
        default=None,
        help="Profile fill strategy",
    )
    parser.add_argument(
        "--profile-smoothing-window",
        type=int,
        default=None,
        help="Median filter window for profile smoothing",
    )
    parser.add_argument(
        "--mesh-radial-segments",
        type=int,
        default=None,
        help="Loft mesh radial segments",
    )
    parser.add_argument(
        "--mesh-cap-mode",
        choices=("fan", "none", "ngon"),
        default=None,
        help="Loft mesh cap mode",
    )
    parser.add_argument(
        "--mesh-min-radius",
        type=float,
        default=None,
        help="Minimum radius (world units) for loft mesh slices",
    )
    parser.add_argument(
        "--mesh-merge-threshold",
        type=float,
        default=None,
        help="Merge threshold for loft mesh (world units)",
    )
    parser.add_argument(
        "--mesh-recalc-normals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recalculate loft mesh normals",
    )
    parser.add_argument(
        "--mesh-shade-smooth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable smooth shading on loft mesh",
    )
    parser.add_argument(
        "--mesh-weld-degenerate-rings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Weld degenerate rings in loft mesh",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Disable progress bars",
    )
    return parser.parse_args(argv)


def _apply_overrides(cfg: BlockingConfig, overrides: Dict[str, Any]) -> None:
    if not overrides:
        return

    recon = overrides.get("reconstruction", {})
    if "reconstruction_mode" in recon:
        cfg.reconstruction.reconstruction_mode = recon["reconstruction_mode"]
    if "num_slices" in recon:
        cfg.reconstruction.num_slices = int(recon["num_slices"])
    if "unit_scale" in recon:
        cfg.reconstruction.unit_scale = float(recon["unit_scale"])

    profile = overrides.get("profile_sampling", {})
    if "num_samples" in profile:
        cfg.profile_sampling.num_samples = int(profile["num_samples"])
    if "sample_policy" in profile:
        cfg.profile_sampling.sample_policy = profile["sample_policy"]
    if "fill_strategy" in profile:
        cfg.profile_sampling.fill_strategy = profile["fill_strategy"]
    if "smoothing_window" in profile:
        cfg.profile_sampling.smoothing_window = int(profile["smoothing_window"])

    mesh = overrides.get("mesh_from_profile", {})
    if "radial_segments" in mesh:
        cfg.mesh_from_profile.radial_segments = int(mesh["radial_segments"])
    if "cap_mode" in mesh:
        cfg.mesh_from_profile.cap_mode = mesh["cap_mode"]
    if "min_radius_u" in mesh:
        cfg.mesh_from_profile.min_radius_u = float(mesh["min_radius_u"])
    if "merge_threshold_u" in mesh:
        cfg.mesh_from_profile.merge_threshold_u = float(mesh["merge_threshold_u"])
    if "recalc_normals" in mesh:
        cfg.mesh_from_profile.recalc_normals = bool(mesh["recalc_normals"])
    if "shade_smooth" in mesh:
        cfg.mesh_from_profile.shade_smooth = bool(mesh["shade_smooth"])
    if "weld_degenerate_rings" in mesh:
        cfg.mesh_from_profile.weld_degenerate_rings = bool(
            mesh["weld_degenerate_rings"]
        )

    join = overrides.get("mesh_join", {})
    if "mode" in join:
        cfg.mesh_join.mode = join["mode"]
    if "boolean_solver" in join:
        cfg.mesh_join.boolean_solver = join["boolean_solver"]

    sil_ref = overrides.get("silhouette_extract_ref", {})
    if "prefer_alpha" in sil_ref:
        cfg.silhouette_extract_ref.prefer_alpha = bool(sil_ref["prefer_alpha"])
    if "alpha_threshold" in sil_ref:
        cfg.silhouette_extract_ref.alpha_threshold = int(sil_ref["alpha_threshold"])
    if "gray_threshold" in sil_ref:
        cfg.silhouette_extract_ref.gray_threshold = (
            int(sil_ref["gray_threshold"])
            if sil_ref["gray_threshold"] is not None
            else None
        )
    if "invert_policy" in sil_ref:
        cfg.silhouette_extract_ref.invert_policy = sil_ref["invert_policy"]
    if "morph_close_px" in sil_ref:
        cfg.silhouette_extract_ref.morph_close_px = int(sil_ref["morph_close_px"])
    if "morph_open_px" in sil_ref:
        cfg.silhouette_extract_ref.morph_open_px = int(sil_ref["morph_open_px"])
    if "fill_holes" in sil_ref:
        cfg.silhouette_extract_ref.fill_holes = bool(sil_ref["fill_holes"])
    if "largest_component_only" in sil_ref:
        cfg.silhouette_extract_ref.largest_component_only = bool(
            sil_ref["largest_component_only"]
        )

    sil_render = overrides.get("silhouette_extract_render", {})
    if "prefer_alpha" in sil_render:
        cfg.silhouette_extract_render.prefer_alpha = bool(sil_render["prefer_alpha"])
    if "alpha_threshold" in sil_render:
        cfg.silhouette_extract_render.alpha_threshold = int(
            sil_render["alpha_threshold"]
        )
    if "gray_threshold" in sil_render:
        cfg.silhouette_extract_render.gray_threshold = (
            int(sil_render["gray_threshold"])
            if sil_render["gray_threshold"] is not None
            else None
        )
    if "invert_policy" in sil_render:
        cfg.silhouette_extract_render.invert_policy = sil_render["invert_policy"]
    if "morph_close_px" in sil_render:
        cfg.silhouette_extract_render.morph_close_px = int(sil_render["morph_close_px"])
    if "morph_open_px" in sil_render:
        cfg.silhouette_extract_render.morph_open_px = int(sil_render["morph_open_px"])
    if "fill_holes" in sil_render:
        cfg.silhouette_extract_render.fill_holes = bool(sil_render["fill_holes"])
    if "largest_component_only" in sil_render:
        cfg.silhouette_extract_render.largest_component_only = bool(
            sil_render["largest_component_only"]
        )

    sil_inter = overrides.get("silhouette_intersection", {})
    if "extrude_distance" in sil_inter:
        cfg.silhouette_intersection.extrude_distance = float(
            sil_inter["extrude_distance"]
        )
    if "contour_mode" in sil_inter:
        cfg.silhouette_intersection.contour_mode = sil_inter["contour_mode"]
    if "largest_component_only" in sil_inter:
        cfg.silhouette_intersection.largest_component_only = (
            bool(sil_inter["largest_component_only"])
            if sil_inter["largest_component_only"] is not None
            else None
        )
    if "silhouette_extract_override" in sil_inter:
        cfg.silhouette_intersection.silhouette_extract_override = sil_inter[
            "silhouette_extract_override"
        ]
    if "boolean_solver" in sil_inter:
        cfg.silhouette_intersection.boolean_solver = sil_inter["boolean_solver"]

    render = overrides.get("render_silhouette", {})
    if "resolution" in render:
        cfg.render_silhouette.resolution = tuple(render["resolution"])
    if "engine" in render:
        cfg.render_silhouette.engine = render["engine"]
    if "transparent_bg" in render:
        cfg.render_silhouette.transparent_bg = bool(render["transparent_bg"])
    if "samples" in render:
        cfg.render_silhouette.samples = int(render["samples"])
    if "margin_frac" in render:
        cfg.render_silhouette.margin_frac = float(render["margin_frac"])
    if "color_mode" in render:
        cfg.render_silhouette.color_mode = render["color_mode"]
    if "force_material" in render:
        cfg.render_silhouette.force_material = bool(render["force_material"])
    if "background_color" in render:
        cfg.render_silhouette.background_color = tuple(render["background_color"])
    if "silhouette_color" in render:
        cfg.render_silhouette.silhouette_color = tuple(render["silhouette_color"])
    if "camera_distance_factor" in render:
        cfg.render_silhouette.camera_distance_factor = float(
            render["camera_distance_factor"]
        )
    if "party_mode" in render:
        cfg.render_silhouette.party_mode = bool(render["party_mode"])

    canon = overrides.get("canonicalize", {})
    if "output_size" in canon:
        cfg.canonicalize.output_size = int(canon["output_size"])
    if "padding_frac" in canon:
        cfg.canonicalize.padding_frac = float(canon["padding_frac"])
    if "anchor" in canon:
        cfg.canonicalize.anchor = canon["anchor"]
    if "interp" in canon:
        cfg.canonicalize.interp = canon["interp"]


def _derive_config_label(args: argparse.Namespace) -> str:
    if args.config_path:
        stem = Path(args.config_path).stem
        for mode in ("legacy", "loft_profile", "silhouette_intersection"):
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
    args = _parse_args(argv)

    workflow_config = BlockingConfig()
    workflow_config.reconstruction.reconstruction_mode = args.reconstruction_mode

    if args.unit_scale is not None:
        workflow_config.reconstruction.unit_scale = float(args.unit_scale)
    if args.num_slices is not None:
        workflow_config.reconstruction.num_slices = int(args.num_slices)

    if args.profile_samples is not None:
        workflow_config.profile_sampling.num_samples = int(args.profile_samples)
    if args.profile_sample_policy is not None:
        workflow_config.profile_sampling.sample_policy = args.profile_sample_policy
    if args.profile_fill_strategy is not None:
        workflow_config.profile_sampling.fill_strategy = args.profile_fill_strategy
    if args.profile_smoothing_window is not None:
        workflow_config.profile_sampling.smoothing_window = int(
            args.profile_smoothing_window
        )

    if args.mesh_radial_segments is not None:
        workflow_config.mesh_from_profile.radial_segments = int(
            args.mesh_radial_segments
        )
    if args.mesh_cap_mode is not None:
        workflow_config.mesh_from_profile.cap_mode = args.mesh_cap_mode
    if args.mesh_min_radius is not None:
        workflow_config.mesh_from_profile.min_radius_u = float(args.mesh_min_radius)
    if args.mesh_merge_threshold is not None:
        workflow_config.mesh_from_profile.merge_threshold_u = float(
            args.mesh_merge_threshold
        )
    if args.mesh_recalc_normals is not None:
        workflow_config.mesh_from_profile.recalc_normals = bool(
            args.mesh_recalc_normals
        )
    if args.mesh_shade_smooth is not None:
        workflow_config.mesh_from_profile.shade_smooth = bool(args.mesh_shade_smooth)
    if args.mesh_weld_degenerate_rings is not None:
        workflow_config.mesh_from_profile.weld_degenerate_rings = bool(
            args.mesh_weld_degenerate_rings
        )

    workflow_config.render_silhouette.resolution = args.resolution
    workflow_config.render_silhouette.engine = args.engine
    workflow_config.render_silhouette.samples = args.samples
    workflow_config.render_silhouette.margin_frac = args.margin

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
    config_label = _derive_config_label(args)

    custom_paths = [args.front, args.side, args.top]
    if any(custom_paths) and not all(custom_paths):
        print("ERROR: --front, --side, and --top must be provided together.")
        sys.exit(2)

    if all(custom_paths):
        success = test_with_custom_images(
            args.front,
            args.side,
            args.top,
            num_slices=args.num_slices,
            render_config=render_config,
            workflow_config=workflow_config,
            config_label=config_label,
            progress=args.progress,
        )
    else:
        # Run test with sample images
        success = test_with_sample_images(
            num_slices=args.num_slices,
            render_config=render_config,
            workflow_config=workflow_config,
            config_label=config_label,
            progress=args.progress,
        )

    # Exit with appropriate code
    sys.exit(0 if success else 1)
