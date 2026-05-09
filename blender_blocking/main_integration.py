"""
Main integration script for Blender automated blocking tool.

This script ties together image processing, shape matching, and 3D operations
to create rough 3D blockouts from orthogonal reference images.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def check_setup() -> None:
    """
    Verify that dependencies are properly installed and compatible.

    Raises helpful error messages if setup is incomplete or incorrect.
    """
    try:
        # Try importing PIL and accessing C extension
        from PIL import Image

        try:
            from PIL import _imaging
        except ImportError as e:
            print("\n" + "=" * 70)
            print("❌ SETUP ERROR: Pillow C extensions not compatible")
            print("=" * 70)
            print(
                "\nYour Pillow installation is not compatible with this Python version."
            )
            print("This typically happens when:")
            print("  - You installed Pillow in a venv with Python 3.13")
            print("  - But Blender is using a different Python version (e.g., 3.11)")
            print("\nPillow includes compiled C extensions that must match your Python")
            print("version exactly. Virtual environment packages won't work.")
            print("\n🔧 REQUIRED FIX:")
            print("Install dependencies directly into Blender's Python:")
            print("\n  # Find Blender's Python path:")
            print("  # In Blender console: import sys; print(sys.executable)")
            print("\n  # Then install:")
            print(
                "  /path/to/blender/python -m pip install -r blender_blocking/requirements.txt"
            )
            print("\n📖 See BLENDER_SETUP.md for detailed instructions")
            print("=" * 70 + "\n")
            raise SystemExit(1)

        # Check other critical imports
        import cv2
        import scipy
        import skimage

    except ImportError as e:
        if "PIL" not in str(e):
            print("\n" + "=" * 70)
            print("❌ SETUP ERROR: Missing dependencies")
            print("=" * 70)
            print(f"\nCould not import required package: {e}")
            print("\n🔧 REQUIRED FIX:")
            print("Install dependencies into Blender's Python:")
            print("\n  # Find Blender's Python:")
            print("  # In Blender console: import sys; print(sys.executable)")
            print("\n  # Then install:")
            print(
                "  /path/to/blender/python -m pip install -r blender_blocking/requirements.txt"
            )
            print("\n📖 See BLENDER_SETUP.md for complete setup guide")
            print("=" * 70 + "\n")
            raise SystemExit(1)


# Import integration modules
from integration.image_processing.image_loader import load_orthogonal_views
from integration.image_processing.image_processor import process_image
from integration.shape_matching.contour_analyzer import find_contours, analyze_shape
from integration.blender_ops.profile_loft_mesh import create_loft_mesh_from_slices
from geometry.profile_models import PixelScale
from geometry.dual_profile import build_elliptical_profile_from_views
from geometry.silhouette import extract_binary_silhouette
from geometry.slicing import sample_elliptical_slices
from config import BlockingConfig
from utils.generation_context import GenerationContext
from utils.manifest import apply_object_tags, build_manifest, write_manifest
from reconstruction.ensemble import CandidateConfig as BackendCandidateConfig
from reconstruction.ensemble import EnsembleRunner
from reconstruction.registry import get_backend, register_builtin_backends
from reconstruction.target_builder import TargetBuildResult, build_target_from_images
from reconstruction.types import CandidateBudget, CandidateMetrics, CandidateRequest, CandidateResult


BACKEND_MODE_ALIASES = {
    "loft_profile": "profile_loft",
}

BACKEND_RECONSTRUCTION_MODES = {
    "loft_profile",
    "profile_loft",
    "silhouette_intersection",
    "visual_hull_voxel",
    "hybrid_loft_hull",
    "primitive_fit_refine",
    "gaussian_ellipsoid_proxy",
    "differentiable_refine",
    "shape_program",
    "ensemble",
}

# Import Blender modules (only available when running in Blender)
try:
    import bpy
    from primitives.primitives import spawn_cube, spawn_sphere, spawn_cylinder
    from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner
    from integration.blender_ops.scene_setup import (
        setup_scene,
        add_camera,
        add_lighting,
    )
    from utils.blender_version import get_boolean_solver

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: Blender API not available. Running in analysis-only mode.")


class BlockingWorkflow:
    """Main workflow class for automated blocking from reference images."""

    def __init__(
        self,
        front_path: Optional[str] = None,
        side_path: Optional[str] = None,
        top_path: Optional[str] = None,
        config: Optional[BlockingConfig] = None,
        context: Optional[GenerationContext] = None,
    ) -> None:
        """
        Initialize the blocking workflow.

        Args:
            front_path: Path to front view reference image
            side_path: Path to side view reference image
            top_path: Path to top view reference image
        """
        self.front_path = front_path
        self.side_path = side_path
        self.top_path = top_path
        self.config = config or BlockingConfig()
        self.context = context or GenerationContext(
            unit_scale=self.config.reconstruction.unit_scale,
            num_slices=self.config.reconstruction.num_slices,
            reconstruction_mode=self.config.reconstruction.reconstruction_mode,
        )
        self.context.config = self.config
        self.views: Dict[str, np.ndarray] = {}
        self.processed_views: Dict[str, np.ndarray] = {}
        self.contours: Dict[str, List[np.ndarray]] = {}
        self.shape_analysis: Dict[str, List[Dict[str, Any]]] = {}
        self.placement_data: Optional[List[Dict[str, Any]]] = None
        self.created_objects: List[Any] = []
        self.manifest: Optional[Dict[str, Any]] = None
        self.target_build: Optional[TargetBuildResult] = None
        self.reconstruction_result: Optional[Any] = None

    def load_images(self) -> Dict[str, np.ndarray]:
        """Load orthogonal reference images."""
        print("Loading images...")
        self.views = load_orthogonal_views(
            front_path=self.front_path, side_path=self.side_path, top_path=self.top_path
        )

        if not self.views:
            raise ValueError(
                "No images loaded. Please provide at least one reference image."
            )

        print(f"Loaded {len(self.views)} views: {', '.join(self.views.keys())}")
        return self.views

    def process_images(self) -> Dict[str, np.ndarray]:
        """Process images to extract edges and prepare for shape analysis."""
        print("Processing images...")

        for view_name, image in self.views.items():
            # Process image: normalize and extract edges
            processed = process_image(
                image, extract_edges_flag=True, normalize_flag=True
            )
            self.processed_views[view_name] = processed
            print(f"  Processed {view_name} view ({image.shape})")

        return self.processed_views

    def analyze_shapes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze shapes from processed images."""
        print("Analyzing shapes...")

        for view_name, edge_image in self.processed_views.items():
            # Find contours
            contours = find_contours(edge_image)
            self.contours[view_name] = contours

            # Analyze each contour
            shapes = []
            for i, contour in enumerate(contours):
                if len(contour) >= 5:  # Need at least 5 points for meaningful analysis
                    analysis = analyze_shape(contour)
                    if analysis["area"] > 100:  # Filter out tiny contours
                        shapes.append(analysis)

            self.shape_analysis[view_name] = shapes
            print(
                f"  {view_name}: Found {len(contours)} contours, {len(shapes)} significant shapes"
            )

        return self.shape_analysis

    def determine_primitive_type(self, shape_info: Dict[str, Any]) -> str:
        """
        Determine which primitive type best matches a shape.

        Args:
            shape_info: Dictionary with shape properties

        Returns:
            String indicating primitive type ('CUBE', 'SPHERE', 'CYLINDER')
        """
        circularity = shape_info.get("circularity", 0)
        aspect_ratio = shape_info.get("aspect_ratio", 1.0)

        # High circularity -> sphere or cylinder
        if circularity > 0.8:
            return "SPHERE"
        elif circularity > 0.5:
            return "CYLINDER"
        else:
            # Low circularity, check aspect ratio
            if 0.8 < aspect_ratio < 1.2:
                return "CUBE"
            else:
                return "CYLINDER"

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Compute an exclusive bounding box from a binary mask."""
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        return x0, y0, x1, y1

    def _calculate_bounds_from_silhouettes(
        self, silhouettes: Dict[str, np.ndarray]
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Calculate bounds from available silhouettes, using scale-aware rules."""
        if not silhouettes:
            return None

        scale = self.config.reconstruction.unit_scale
        front_bbox = (
            self._bbox_from_mask(silhouettes.get("front"))
            if "front" in silhouettes
            else None
        )
        side_bbox = (
            self._bbox_from_mask(silhouettes.get("side"))
            if "side" in silhouettes
            else None
        )
        top_bbox = (
            self._bbox_from_mask(silhouettes.get("top"))
            if "top" in silhouettes
            else None
        )

        if not any([front_bbox, side_bbox, top_bbox]):
            return None

        width = None
        depth = None
        height = None

        if front_bbox:
            _, _, x1, y1 = front_bbox
            x0, y0 = front_bbox[0], front_bbox[1]
            width = (x1 - x0) * scale
            height = (y1 - y0) * scale
        if side_bbox:
            _, _, x1, y1 = side_bbox
            x0, y0 = side_bbox[0], side_bbox[1]
            depth = (x1 - x0) * scale
            height = max(height or 0.0, (y1 - y0) * scale)
        if top_bbox:
            _, _, x1, y1 = top_bbox
            x0, y0 = top_bbox[0], top_bbox[1]
            if width is None:
                width = (x1 - x0) * scale
            if depth is None:
                depth = (y1 - y0) * scale

        fallback_min, fallback_max = self.calculate_bounds_from_shapes()
        fallback_width = fallback_max[0] - fallback_min[0]
        fallback_depth = fallback_max[1] - fallback_min[1]
        fallback_height = fallback_max[2] - fallback_min[2]

        if width is None:
            width = fallback_width
        if depth is None:
            depth = fallback_depth
        if height is None:
            height = fallback_height

        if width is None or depth is None or height is None:
            return None

        bounds_min = (-width / 2, -depth / 2, 0)
        bounds_max = (width / 2, depth / 2, height)
        return bounds_min, bounds_max

    def calculate_bounds_from_shapes(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculate 3D bounds based on analyzed shapes from multiple views.

        Returns:
            Tuple of (bounds_min, bounds_max) as vectors
        """
        # Use front and side views to estimate 3D dimensions
        front_shapes = self.shape_analysis.get("front", [])
        side_shapes = self.shape_analysis.get("side", [])
        top_shapes = self.shape_analysis.get("top", [])

        if not front_shapes and not side_shapes and not top_shapes:
            # Default bounds if no shapes found
            return ((-2, -2, 0), (2, 2, 4))

        # Get largest shape from each view
        def get_largest_bbox(
            shapes: List[Dict[str, Any]],
        ) -> Optional[Tuple[int, int, int, int]]:
            """Return the bounding box for the largest-area shape, if any."""
            if not shapes:
                return None
            largest = max(shapes, key=lambda s: s["area"])
            return largest["bounding_box"]

        front_bbox = get_largest_bbox(front_shapes)
        side_bbox = get_largest_bbox(side_shapes)
        top_bbox = get_largest_bbox(top_shapes)

        # Debug: Show detected bounding boxes
        if front_bbox:
            print(f"    DEBUG: Front bbox: {front_bbox}")
        if side_bbox:
            print(f"    DEBUG: Side bbox: {side_bbox}")
        if top_bbox:
            print(f"    DEBUG: Top bbox: {top_bbox}")

        scale = self.config.reconstruction.unit_scale

        width = None
        depth = None
        height = None

        if front_bbox:
            _, _, fw, fh = front_bbox
            width = fw * scale
            height = fh * scale
        if side_bbox:
            _, _, sw, sh = side_bbox
            depth = sw * scale
            height = max(height or 0.0, sh * scale)
        if top_bbox:
            _, _, tw, th = top_bbox
            if width is None:
                width = tw * scale
            if depth is None:
                depth = th * scale

        if width is None and depth is not None:
            width = depth
        if depth is None and width is not None:
            depth = width
        if height is None:
            height = 4.0

        # Center the object at origin
        bounds_min = (-width / 2, -depth / 2, 0)
        bounds_max = (width / 2, depth / 2, height)

        return bounds_min, bounds_max

    def create_3d_blockout(
        self, num_slices: Optional[int] = None, primitive_type: str = "CYLINDER"
    ) -> Optional[Any]:
        """
        Create 3D blockout in Blender based on analyzed shapes.

        Args:
            num_slices: Number of vertical slices for reconstruction
            primitive_type: Default primitive type to use

        Returns:
            Final joined mesh object
        """
        if not BLENDER_AVAILABLE:
            print("Error: Blender API not available. Cannot create 3D blockout.")
            return None

        if num_slices is None:
            num_slices = self.config.reconstruction.num_slices
        else:
            self.config.reconstruction.num_slices = num_slices

        self.context.num_slices = num_slices
        self.context.unit_scale = self.config.reconstruction.unit_scale
        self.context.reconstruction_mode = (
            self.config.reconstruction.reconstruction_mode
        )

        print("Creating 3D blockout in Blender...")

        # Setup clean Blender scene
        setup_scene(clear_existing=True)

        silhouettes: Dict[str, np.ndarray] = {}
        vertical_profile = None

        # Extract vertical profile from reference images first (we'll use it for bounds too)
        # Use original images (not edge-detected) for better silhouette extraction
        if "front" in self.views:
            from integration.shape_matching.profile_extractor import (
                extract_vertical_profile,
                extract_silhouette_from_image,
            )

            try:
                silhouette = extract_silhouette_from_image(self.views["front"])
                silhouettes["front"] = silhouette

                vertical_profile = extract_vertical_profile(
                    self.views["front"], num_samples=num_slices
                )
                print(
                    f"  Extracted vertical profile from front view ({len(vertical_profile)} samples)"
                )
                radii = [r for h, r in vertical_profile]
                print(f"  Profile radius range: {min(radii):.3f} to {max(radii):.3f}")
            except Exception as e:
                print(f"  Warning: Could not extract profile from front view: {e}")
        elif "side" in self.views:
            from integration.shape_matching.profile_extractor import (
                extract_vertical_profile,
                extract_silhouette_from_image,
            )

            try:
                silhouette = extract_silhouette_from_image(self.views["side"])
                silhouettes["side"] = silhouette

                vertical_profile = extract_vertical_profile(
                    self.views["side"], num_samples=num_slices
                )
                print(
                    f"  Extracted vertical profile from side view ({len(vertical_profile)} samples)"
                )
                radii = [r for h, r in vertical_profile]
                print(f"  Profile radius range: {min(radii):.3f} to {max(radii):.3f}")
            except Exception as e:
                print(f"  Warning: Could not extract profile from side view: {e}")

        for view_name in ("front", "side", "top"):
            if view_name in self.views and view_name not in silhouettes:
                from integration.shape_matching.profile_extractor import (
                    extract_silhouette_from_image,
                )

                try:
                    silhouettes[view_name] = extract_silhouette_from_image(
                        self.views[view_name]
                    )
                except Exception as e:
                    print(
                        f"  Warning: Could not extract silhouette for {view_name}: {e}"
                    )

        bounds = self._calculate_bounds_from_silhouettes(silhouettes)
        if bounds:
            bounds_min, bounds_max = bounds
            print(f"  Bounds from silhouettes: {bounds_min} to {bounds_max}")
        else:
            bounds_min, bounds_max = self.calculate_bounds_from_shapes()
            print(f"  Bounds from shape analysis: {bounds_min} to {bounds_max}")

        width = bounds_max[0] - bounds_min[0]
        depth = bounds_max[1] - bounds_min[1]
        height = bounds_max[2] - bounds_min[2]
        print(f"  Dimensions: {width:.3f} x {depth:.3f} x {height:.3f}")

        warnings: List[str] = []
        if width <= 0 or depth <= 0 or height <= 0:
            warnings.append("Degenerate bounds detected from silhouettes or shapes.")

        if self.context.dry_run:
            outputs = {
                "primitives": {"count": 0, "names": []},
                "final_mesh": {"count": 0, "name": None},
                "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
            }
            manifest = build_manifest(
                self.context, outputs=outputs, warnings=warnings, errors=[]
            )
            write_manifest(bpy.context.scene, manifest)
            self.manifest = manifest
            return None

        # Analyze slices with profile data
        print(f"  Analyzing {num_slices} slices...")
        analyzer = SliceAnalyzer(
            bounds_min,
            bounds_max,
            num_slices=num_slices,
            vertical_profile=vertical_profile,
        )
        slice_data = analyzer.get_all_slice_data()

        # Place primitives
        print("  Placing primitives...")
        placer = PrimitivePlacer()

        if vertical_profile:
            primitive_type = "CYLINDER"
            print("  Using CYLINDER primitives for profile-based reconstruction")
        elif self.shape_analysis:
            all_shapes = []
            for shapes in self.shape_analysis.values():
                all_shapes.extend(shapes)

            if all_shapes:
                largest_shape = max(all_shapes, key=lambda s: s["area"])
                primitive_type = self.determine_primitive_type(largest_shape)
                print(f"  Auto-selected primitive type: {primitive_type}")

        objects = placer.place_primitives_from_slices(
            slice_data, primitive_type=primitive_type, min_radius=analyzer.min_radius
        )
        self.created_objects = objects
        print(f"  Placed {len(objects)} primitives")

        for idx, obj in enumerate(objects):
            apply_object_tags(obj, role="primitive", context=self.context, index=idx)

        primitive_names = [obj.name for obj in objects]

        final_mesh = None
        if objects:
            print("  Joining meshes...")
            joiner = MeshJoiner()
            final_mesh = joiner.join(
                objects,
                target_name="Blockout_Mesh",
                mode=self.config.mesh_join.mode,
                solver=self.config.mesh_join.boolean_solver,
                allow_degraded_simple_join=(
                    self.config.mesh_join.allow_degraded_simple_join
                ),
            )

            if (
                self.views.get("front") is not None
                or self.views.get("side") is not None
            ):
                from integration.shape_matching.vertex_refinement import (
                    refine_mesh_to_silhouettes,
                )

                final_mesh = refine_mesh_to_silhouettes(
                    final_mesh,
                    front_silhouette=self.views.get("front"),
                    side_silhouette=self.views.get("side"),
                    subdivision_levels=1,
                )

            apply_object_tags(final_mesh, role="final", context=self.context)

            print("  Setting up camera and lighting...")
            add_camera()
            add_lighting()

        outputs = {
            "primitives": {"count": len(objects), "names": primitive_names},
            "final_mesh": {
                "count": 1 if final_mesh else 0,
                "name": final_mesh.name if final_mesh else None,
            },
            "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
        }
        manifest = build_manifest(
            self.context, outputs=outputs, warnings=warnings, errors=[]
        )
        write_manifest(bpy.context.scene, manifest)
        self.manifest = manifest

        if final_mesh:
            print(f"✓ Created blockout mesh: {final_mesh.name}")
        else:
            print("  Warning: No primitives were created")
        return final_mesh

    def create_3d_blockout_loft(
        self,
        num_slices: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Create 3D blockout using the loft profile pipeline.

        Args:
            num_slices: Number of slices for loft sampling

        Returns:
            Final lofted mesh object
        """
        if not BLENDER_AVAILABLE:
            print("Error: Blender API not available. Cannot create 3D blockout.")
            return None

        if num_slices is None:
            num_slices = self.config.reconstruction.num_slices
        else:
            self.config.reconstruction.num_slices = num_slices

        self.context.num_slices = num_slices
        self.context.unit_scale = self.config.reconstruction.unit_scale
        self.context.reconstruction_mode = (
            self.config.reconstruction.reconstruction_mode
        )

        print("Creating 3D blockout with loft pipeline...")

        setup_scene(clear_existing=True)

        warnings: List[str] = []
        front_mask = None
        side_mask = None

        if "front" in self.views:
            try:
                front_mask = extract_binary_silhouette(self.views["front"])
            except Exception as exc:
                warnings.append(f"Failed to extract front silhouette: {exc}")

        if "side" in self.views:
            try:
                side_mask = extract_binary_silhouette(self.views["side"])
            except Exception as exc:
                warnings.append(f"Failed to extract side silhouette: {exc}")

        if front_mask is None and side_mask is None:
            warnings.append(
                "No front/side silhouettes available; falling back to legacy pipeline."
            )
            print(
                "  Warning: No front/side silhouettes available, falling back to legacy."
            )
            return self.create_3d_blockout(num_slices=num_slices)

        if front_mask is None or side_mask is None:
            warnings.append(
                "Only one silhouette view available; using circular fallback."
            )

        scale = PixelScale(unit_per_px=self.config.reconstruction.unit_scale)
        profile = build_elliptical_profile_from_views(
            front_mask,
            side_mask,
            scale,
            num_samples=self.config.profile_sampling.num_samples,
            z0=0.0,
            height_strategy="front",
            fallback_policy="circular",
            min_radius_u=self.config.mesh_from_profile.min_radius_u,
            sample_policy=self.config.profile_sampling.sample_policy,
            fill_strategy=self.config.profile_sampling.fill_strategy,
            smoothing_window=self.config.profile_sampling.smoothing_window,
            enable_offsets=False,
        )

        slices = sample_elliptical_slices(
            profile,
            num_slices=num_slices,
            sampling=self.config.profile_sampling.sample_policy,
        )

        max_rx = max(slice_data.rx for slice_data in slices)
        max_ry = max(slice_data.ry for slice_data in slices)
        bounds_min = (-max_rx, -max_ry, profile.z0)
        bounds_max = (max_rx, max_ry, profile.z0 + profile.world_height)

        if self.context.dry_run:
            outputs = {
                "primitives": {"count": 0, "names": []},
                "final_mesh": {"count": 0, "name": None},
                "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
            }
            manifest = build_manifest(
                self.context, outputs=outputs, warnings=warnings, errors=[]
            )
            write_manifest(bpy.context.scene, manifest)
            self.manifest = manifest
            return None

        final_mesh = create_loft_mesh_from_slices(
            slices,
            name="Blockout_Mesh",
            radial_segments=self.config.mesh_from_profile.radial_segments,
            adaptive_radial_segments=(
                self.config.mesh_from_profile.adaptive_radial_segments
            ),
            cap_mode=self.config.mesh_from_profile.cap_mode,
            min_radius_u=self.config.mesh_from_profile.min_radius_u,
            merge_threshold_u=self.config.mesh_from_profile.merge_threshold_u,
            recalc_normals=self.config.mesh_from_profile.recalc_normals,
            shade_smooth=self.config.mesh_from_profile.shade_smooth,
            weld_degenerate_rings=self.config.mesh_from_profile.weld_degenerate_rings,
        )

        if final_mesh is not None:
            apply_object_tags(final_mesh, role="final", context=self.context)
            add_camera()
            add_lighting()

        outputs = {
            "primitives": {"count": 0, "names": []},
            "final_mesh": {
                "count": 1 if final_mesh else 0,
                "name": final_mesh.name if final_mesh else None,
            },
            "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
        }
        manifest = build_manifest(
            self.context, outputs=outputs, warnings=warnings, errors=[]
        )
        write_manifest(bpy.context.scene, manifest)
        self.manifest = manifest

        if final_mesh:
            print(f"✓ Created lofted mesh: {final_mesh.name}")
        else:
            print("  Warning: Loft mesh generation failed")

        return final_mesh

    def create_3d_blockout_silhouette_intersection(
        self,
        num_slices: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Create 3D blockout by intersecting extruded front/side silhouettes.

        This preserves concavities visible in the front silhouette (e.g. leg gaps)
        while staying consistent with side depth.
        """
        if not BLENDER_AVAILABLE:
            print("Error: Blender API not available. Cannot create 3D blockout.")
            return None

        if "front" not in self.views or "side" not in self.views:
            print("Warning: Front and side views are required for intersection mode.")
            return self.create_3d_blockout(num_slices=num_slices)

        setup_scene(clear_existing=True)

        from geometry.silhouette import extract_binary_silhouette
        from integration.shape_matching.contour_analyzer import find_contours
        from integration.blender_ops.mesh_generator import (
            create_mesh_from_contours,
            extrude_profile,
            center_extrusion,
            clean_mesh_for_boolean,
            triangulate_object,
        )
        from utils.blender_version import resolve_boolean_solver
        import cv2

        def _resolve_extract_config() -> Dict[str, object]:
            base_cfg = self.config.silhouette_extract_ref.to_dict()
            override = (
                self.config.silhouette_intersection.silhouette_extract_override or {}
            )
            for key, value in override.items():
                if value is not None:
                    base_cfg[key] = value
            if self.config.silhouette_intersection.largest_component_only is not None:
                base_cfg["largest_component_only"] = (
                    self.config.silhouette_intersection.largest_component_only
                )
            supported = {
                "prefer_alpha",
                "alpha_threshold",
                "gray_threshold",
                "invert_policy",
                "morph_close_px",
                "morph_open_px",
                "fill_holes",
                "largest_component_only",
            }
            return {key: value for key, value in base_cfg.items() if key in supported}

        extract_cfg = _resolve_extract_config()
        front_mask = extract_binary_silhouette(self.views["front"], **extract_cfg)
        side_mask = extract_binary_silhouette(self.views["side"], **extract_cfg)
        front_bbox_px = self._bbox_from_mask(front_mask)
        side_bbox_px = self._bbox_from_mask(side_mask)

        bounds = self._calculate_bounds_from_silhouettes(
            {"front": front_mask, "side": side_mask}
        )
        if not bounds:
            print("Warning: Failed to compute bounds from silhouettes.")
            return self.create_3d_blockout(num_slices=num_slices)

        bounds_min, bounds_max = bounds
        width = bounds_max[0] - bounds_min[0]
        depth = bounds_max[1] - bounds_min[1]
        height = bounds_max[2] - bounds_min[2]

        if width <= 0 or depth <= 0 or height <= 0:
            print("Warning: Degenerate bounds for silhouette intersection.")
            return self.create_3d_blockout(num_slices=num_slices)

        contour_mode = self.config.silhouette_intersection.contour_mode
        front_contours, front_hierarchy = find_contours(
            (front_mask.astype(np.uint8) * 255),
            mode=contour_mode,
            return_hierarchy=True,
        )
        side_contours, side_hierarchy = find_contours(
            (side_mask.astype(np.uint8) * 255),
            mode=contour_mode,
            return_hierarchy=True,
        )
        if not front_contours or not side_contours:
            print("Warning: Missing contours for silhouette intersection.")
            return self.create_3d_blockout(num_slices=num_slices)

        def _split_contours(
            contours: List[np.ndarray], hierarchy: Optional[np.ndarray]
        ) -> Tuple[List[int], Dict[int, List[int]]]:
            if not contours:
                return [], {}
            if hierarchy is None or len(hierarchy) == 0:
                outer = list(range(len(contours)))
                return outer, {}
            outer = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]
            holes: Dict[int, List[int]] = {idx: [] for idx in outer}
            for idx, h in enumerate(hierarchy[0]):
                parent = h[3]
                if parent != -1 and parent in holes:
                    holes[parent].append(idx)
            if not outer:
                outer = list(range(len(contours)))
            return outer, holes

        def _apply_transforms(obj: object) -> None:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
            triangulate_object(obj)
            clean_mesh_for_boolean(obj)
            obj.select_set(False)

        def _mesh_counts(obj: object) -> Tuple[int, int]:
            if obj is None or getattr(obj, "type", None) != "MESH":
                return 0, 0
            return len(obj.data.vertices), len(obj.data.polygons)

        def _apply_boolean(
            base: object, other: object, operation: str, solver: str
        ) -> bool:
            if base is None or other is None:
                return False
            modifier = base.modifiers.new(name=f"{operation}_Op", type="BOOLEAN")
            modifier.operation = operation
            modifier.object = other
            modifier.solver = solver
            bpy.context.view_layer.objects.active = base
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(other, do_unlink=True)
            verts, faces = _mesh_counts(base)
            if verts == 0 or faces == 0:
                print(
                    f"Warning: Boolean {operation} with solver {solver} produced empty mesh"
                )
                return False
            return True

        largest_only = self.config.silhouette_intersection.largest_component_only
        if largest_only is None:
            largest_only = bool(extract_cfg.get("largest_component_only", False))

        def _build_silhouette_object(
            contours: List[np.ndarray],
            hierarchy: Optional[np.ndarray],
            *,
            name_prefix: str,
            scale: Tuple[float, float, float],
            rotation: Tuple[float, float, float],
            extrude_distance: float,
            solver: str,
            source_size: Tuple[int, int],
            normalize_bounds: Optional[Tuple[float, float, float, float]],
        ) -> Optional[object]:
            outer, holes = _split_contours(contours, hierarchy)
            if not outer:
                return None
            if largest_only:
                outer = [max(outer, key=lambda i: cv2.contourArea(contours[i]))]
            parts: List[object] = []
            for idx in outer:
                obj = create_mesh_from_contours(
                    [contours[idx]],
                    name=f"{name_prefix}_{idx}",
                    source_size=source_size,
                    normalize_bounds=normalize_bounds,
                )
                if obj is None:
                    print(
                        f"Warning: create_mesh_from_contours returned None for {name_prefix}_{idx}"
                    )
                    continue
                triangulate_object(obj)
                extrude_profile(obj, extrude_distance=extrude_distance)
                center_extrusion(obj, extrude_distance=extrude_distance)
                obj.scale = scale
                obj.rotation_euler = rotation
                _apply_transforms(obj)
                verts, faces = _mesh_counts(obj)
                print(
                    f"DEBUG silhouette_intersection {obj.name}: verts={verts} faces={faces}"
                )

                for hole_idx in holes.get(idx, []):
                    hole_obj = create_mesh_from_contours(
                        [contours[hole_idx]],
                        name=f"{name_prefix}_Hole_{hole_idx}",
                        source_size=source_size,
                        normalize_bounds=normalize_bounds,
                    )
                    if hole_obj is None:
                        print(
                            f"Warning: create_mesh_from_contours returned None for {name_prefix}_Hole_{hole_idx}"
                        )
                        continue
                    triangulate_object(hole_obj)
                    extrude_profile(hole_obj, extrude_distance=extrude_distance)
                    center_extrusion(hole_obj, extrude_distance=extrude_distance)
                    hole_obj.scale = scale
                    hole_obj.rotation_euler = rotation
                    _apply_transforms(hole_obj)
                    hole_ok = _apply_boolean(obj, hole_obj, "DIFFERENCE", solver)
                    if not hole_ok:
                        print(
                            f"Warning: Hole subtraction failed for {obj.name} using solver {solver}"
                        )

                parts.append(obj)

            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            base = parts[0]
            for extra in parts[1:]:
                ok = _apply_boolean(base, extra, "UNION", solver)
                if not ok:
                    print(
                        f"Warning: UNION failed while combining {base.name}; solver={solver}"
                    )
            return base

        solver_override = self.config.silhouette_intersection.boolean_solver
        if solver_override == "auto":
            solver_override = self.config.mesh_join.boolean_solver
        solver = resolve_boolean_solver(solver_override)

        extrude_distance = self.config.silhouette_intersection.extrude_distance
        front_size = (front_mask.shape[1], front_mask.shape[0])
        side_size = (side_mask.shape[1], side_mask.shape[0])
        front_obj = _build_silhouette_object(
            front_contours,
            front_hierarchy,
            name_prefix="Front_Silhouette",
            scale=(width / 2.0, height / 2.0, depth),
            rotation=(math.radians(-90.0), 0.0, 0.0),
            extrude_distance=extrude_distance,
            solver=solver,
            source_size=front_size,
            normalize_bounds=front_bbox_px,
        )
        side_obj = _build_silhouette_object(
            side_contours,
            side_hierarchy,
            name_prefix="Side_Silhouette",
            scale=(depth / 2.0, height / 2.0, width),
            rotation=(math.radians(-90.0), 0.0, math.radians(90.0)),
            extrude_distance=extrude_distance,
            solver=solver,
            source_size=side_size,
            normalize_bounds=side_bbox_px,
        )

        if front_obj is None or side_obj is None:
            print("Warning: Failed to create silhouette meshes.")
            return self.create_3d_blockout(num_slices=num_slices)

        front_verts, front_faces = _mesh_counts(front_obj)
        side_verts, side_faces = _mesh_counts(side_obj)
        print(
            f"DEBUG silhouette_intersection front_obj={front_obj.name} verts={front_verts} faces={front_faces}"
        )
        print(
            f"DEBUG silhouette_intersection side_obj={side_obj.name} verts={side_verts} faces={side_faces}"
        )

        bpy.context.view_layer.update()

        base_obj = front_obj
        base_obj.name = "Blockout_Mesh"

        clean_mesh_for_boolean(base_obj)
        clean_mesh_for_boolean(side_obj)

        modifier = base_obj.modifiers.new(name="Intersect", type="BOOLEAN")
        modifier.operation = "INTERSECT"
        modifier.object = side_obj
        modifier.solver = solver

        bpy.context.view_layer.objects.active = base_obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(side_obj, do_unlink=True)

        final_verts, final_faces = _mesh_counts(base_obj)
        print(
            f"DEBUG silhouette_intersection intersect result verts={final_verts} faces={final_faces} solver={solver}"
        )
        if final_verts == 0 or final_faces == 0:
            error = (
                "silhouette intersection produced an empty mesh; try "
                "silhouette_intersection.boolean_solver=EXACT or MANIFOLD"
            )
            print(f"Warning: {error}")
            outputs = {
                "primitives": {"count": 0, "names": []},
                "final_mesh": {"count": 0, "name": None},
                "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
            }
            manifest = build_manifest(
                self.context,
                outputs=outputs,
                warnings=[],
                errors=[error],
            )
            write_manifest(bpy.context.scene, manifest)
            self.manifest = manifest
            return None

        apply_object_tags(base_obj, role="final", context=self.context)
        add_camera()
        add_lighting()

        outputs = {
            "primitives": {"count": 0, "names": []},
            "final_mesh": {"count": 1, "name": base_obj.name},
            "bounds": {"min": list(bounds_min), "max": list(bounds_max)},
        }
        manifest = build_manifest(self.context, outputs=outputs, warnings=[], errors=[])
        write_manifest(bpy.context.scene, manifest)
        self.manifest = manifest

        print(f"✓ Created silhouette intersection mesh: {base_obj.name}")
        return base_obj

    def _artifact_root(self) -> Path:
        """Return the scoped artifact root for this workflow run."""
        artifact_root = getattr(self.context, "artifact_root", None)
        if artifact_root:
            return Path(str(artifact_root))
        return Path("test_output") / "reconstruction" / self.context.run_id

    def _backend_name_for_mode(self, mode: str) -> str:
        """Return the canonical backend name for a reconstruction mode."""
        return BACKEND_MODE_ALIASES.get(mode, mode)

    def _is_backend_mode(self, mode: str) -> bool:
        """Return whether a mode must route through the backend contract."""
        return mode in BACKEND_RECONSTRUCTION_MODES

    def _target_bounds_minmax(
        self, silhouettes: Optional[Mapping[str, np.ndarray]] = None
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if silhouettes:
            bounds = self._calculate_bounds_from_silhouettes(dict(silhouettes))
            if bounds:
                return bounds
        return self.calculate_bounds_from_shapes()

    def build_reconstruction_target(self) -> TargetBuildResult:
        """Build and cache the typed reconstruction target for backend modes."""
        if self.target_build is not None:
            return self.target_build
        artifact_root = self._artifact_root()
        self.target_build = build_target_from_images(
            self.views,
            config=self.config,
            constraint_files=self.config.constraints.constraint_files,
            artifact_root=artifact_root,
            profile_samples=self.config.profile_sampling.num_samples,
        )
        return self.target_build

    def _config_for_backend(self, backend_name: str) -> Dict[str, Any]:
        """Return backend-specific config payloads from BlockingConfig."""
        backend_name = self._backend_name_for_mode(backend_name)
        if backend_name == "legacy":
            return self.config.reconstruction.to_dict()
        if backend_name == "profile_loft":
            return {
                **self.config.reconstruction.to_dict(),
                **self.config.profile_sampling.to_dict(),
                **self.config.mesh_from_profile.to_dict(),
            }
        if backend_name == "silhouette_intersection":
            return self.config.silhouette_intersection.to_dict()
        if backend_name == "visual_hull_voxel":
            return self.config.visual_hull.to_dict()
        if backend_name == "hybrid_loft_hull":
            return {
                **self.config.visual_hull.to_dict(),
                **self.config.reconstruction.to_dict(),
                "profile_sampling": self.config.profile_sampling.to_dict(),
                "mesh_from_profile": self.config.mesh_from_profile.to_dict(),
            }
        if backend_name == "primitive_fit_refine":
            return {
                **self.config.primitive_fit.to_dict(),
                "visual_hull_resolution": min(
                    self.config.visual_hull.resolution,
                    self.config.visual_hull.max_resolution,
                ),
                "chunk_size": self.config.visual_hull.chunk_size,
            }
        if backend_name == "gaussian_ellipsoid_proxy":
            return {
                **self.config.gaussian_ellipsoid.to_dict(),
                "visual_hull_resolution": min(
                    self.config.visual_hull.resolution,
                    self.config.visual_hull.max_resolution,
                ),
                "chunk_size": self.config.visual_hull.chunk_size,
            }
        if backend_name == "differentiable_refine":
            config = self.config.differentiable_render.to_dict()
            config.setdefault(
                "visual_hull_resolution",
                min(
                    self.config.visual_hull.resolution,
                    self.config.visual_hull.max_resolution,
                ),
            )
            config.setdefault(
                "primitive_count",
                self.config.gaussian_ellipsoid.primitive_count,
            )
            config.setdefault("chunk_size", self.config.visual_hull.chunk_size)
            return config
        if backend_name == "shape_program":
            return self.config.shape_program.to_dict()
        return {}

    def _default_ensemble_candidates(self) -> Tuple[BackendCandidateConfig, ...]:
        names = [
            "profile_loft" if BLENDER_AVAILABLE else None,
            "silhouette_intersection" if BLENDER_AVAILABLE else None,
            "visual_hull_voxel",
            "primitive_fit_refine",
            "gaussian_ellipsoid_proxy",
            "differentiable_refine",
            "shape_program",
        ]
        candidates = []
        for name in names:
            if not name:
                continue
            candidates.append(
                BackendCandidateConfig(
                    backend_name=name,
                    candidate_id=name,
                    config=self._config_for_backend(name),
                )
            )
        return tuple(candidates)

    def _configured_ensemble_candidates(self) -> Tuple[BackendCandidateConfig, ...]:
        configured = self.config.ensemble.candidates
        if not configured:
            return self._default_ensemble_candidates()
        return tuple(
            BackendCandidateConfig(
                backend_name=candidate.backend_name,
                enabled=candidate.enabled,
                config={
                    **self._config_for_backend(candidate.backend_name),
                    **dict(candidate.config),
                },
                candidate_id=candidate.candidate_id,
            )
            for candidate in configured
        )

    def run_backend_reconstruction(
        self,
        *,
        mode: Optional[str] = None,
    ) -> Optional[Any]:
        """Run the typed backend registry path for ambitious modes."""
        selected_mode = mode or self.config.reconstruction.reconstruction_mode
        backend_name = self._backend_name_for_mode(selected_mode)
        register_builtin_backends()
        target_build = self.build_reconstruction_target()
        artifact_root = self._artifact_root()
        artifact_root.mkdir(parents=True, exist_ok=True)
        context = SimpleNamespace(
            workflow=self,
            blender_available=BLENDER_AVAILABLE,
            generation_context=self.context,
            requested_mode=selected_mode,
            backend_name=backend_name,
        )

        if selected_mode == "ensemble":
            runner = EnsembleRunner(
                selection_policy=self.config.ensemble.selection_policy
            )
            result = runner.run(
                target=target_build.target,
                candidates=self._configured_ensemble_candidates(),
                artifact_root=artifact_root / "candidates",
                context=context,
                budget=CandidateBudget(
                    timeout_s=self.config.ensemble.per_candidate_timeout_s,
                    memory_budget_mb=self.config.visual_hull.memory_budget_mb,
                ),
                total_timeout_s=self.config.ensemble.total_timeout_s,
            )
            self.reconstruction_result = result
            self._record_backend_manifest(
                result.to_dict(),
                target_build,
                result_obj=result,
                suite=selected_mode,
            )
            return result.selected.payload if result.selected else None

        backend = get_backend(backend_name)
        if backend.capabilities.requires_blender and not BLENDER_AVAILABLE:
            result = CandidateResult(
                candidate_id=backend_name,
                backend_name=backend_name,
                status="skipped",
                metric_result=CandidateMetrics(),
                warnings=(
                    f"{backend_name} requires Blender",
                    f"requested mode was {selected_mode}",
                ),
            )
        else:
            request = CandidateRequest(
                candidate_id=backend_name,
                backend_name=backend_name,
                target=target_build.target,
                config=self._config_for_backend(backend_name),
                budget=CandidateBudget(
                    timeout_s=(
                        self.config.primitive_fit.max_runtime_s
                        if backend_name == "primitive_fit_refine"
                        else None
                    ),
                    memory_budget_mb=self.config.visual_hull.memory_budget_mb
                ),
                artifact_root=artifact_root / "candidates",
                context=context,
            )
            errors = backend.validate_config(request.config)
            if errors:
                result = CandidateResult(
                    candidate_id=backend_name,
                    backend_name=backend_name,
                    status="failed",
                    errors=tuple(errors),
                    warnings=(f"requested mode was {selected_mode}",),
                )
            else:
                result = backend.reconstruct(request)

        self.reconstruction_result = result
        self._record_backend_manifest(
            result.to_dict(),
            target_build,
            result_obj=result,
            suite=selected_mode,
        )
        if BLENDER_AVAILABLE and getattr(result.payload, "name", None):
            apply_object_tags(result.payload, role="final", context=self.context)
        return result.payload

    def _record_backend_manifest(
        self,
        result_payload: Mapping[str, Any],
        target_build: TargetBuildResult,
        *,
        result_obj: Any = None,
        suite: str = "",
    ) -> None:
        outputs = {
            "backend_result": result_payload,
            "target": target_build.to_manifest_fragment(),
            "artifact_root": str(self._artifact_root()),
        }
        outputs.update(
            self._evaluation_manifest_outputs(
                result_obj=result_obj,
                target_build=target_build,
                suite=suite,
            )
        )
        warnings = list(target_build.warnings)
        errors = list(result_payload.get("errors", []))
        manifest = build_manifest(
            self.context,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
        )
        self.manifest = manifest
        if BLENDER_AVAILABLE:
            write_manifest(bpy.context.scene, manifest)

    def _evaluation_manifest_outputs(
        self,
        *,
        result_obj: Any,
        target_build: TargetBuildResult,
        suite: str,
    ) -> Dict[str, Any]:
        if result_obj is None:
            return {}
        if hasattr(result_obj, "evaluation_bundles"):
            bundles = getattr(result_obj, "evaluation_bundles", ()) or ()
            autopsy_packs = getattr(result_obj, "autopsy_packs", ()) or ()
            active_view_plans = [
                self._active_view_plan(bundle, target_build=target_build)
                for bundle in bundles
            ]
            return {
                "evaluation_bundles": [
                    bundle.to_dict() if hasattr(bundle, "to_dict") else bundle
                    for bundle in bundles
                ],
                "autopsy_packs": [
                    pack.to_dict() if hasattr(pack, "to_dict") else pack
                    for pack in autopsy_packs
                ],
                "active_view_plans": active_view_plans,
            }
        if hasattr(result_obj, "to_evaluation_bundle"):
            try:
                bundle = result_obj.to_evaluation_bundle(
                    target=target_build.target,
                    suite=suite,
                    run_id=str(getattr(result_obj, "candidate_id", suite)),
                )
                try:
                    from blender_blocking.evaluation.autopsy import (
                        autopsy_pack_from_bundle,
                    )
                except Exception:  # pragma: no cover - legacy script import path
                    from evaluation.autopsy import autopsy_pack_from_bundle  # type: ignore

                return {
                    "evaluation_bundle": bundle.to_dict(),
                    "autopsy_pack": autopsy_pack_from_bundle(bundle).to_dict(),
                    "active_view_plan": self._active_view_plan(
                        bundle,
                        target_build=target_build,
                    ),
                }
            except Exception as exc:
                return {"evaluation_bundle_error": str(exc)}
        return {}

    def _active_view_plan(
        self,
        bundle: Any,
        *,
        target_build: TargetBuildResult,
    ) -> Dict[str, Any]:
        try:
            from blender_blocking.evaluation.view_planning import (
                active_view_plan_payload,
            )
        except Exception:  # pragma: no cover - legacy script import path
            from evaluation.view_planning import active_view_plan_payload  # type: ignore

        return active_view_plan_payload(
            bundle,
            existing_views=target_build.target.views(),
        )

    def run_full_workflow(self, num_slices: Optional[int] = None) -> Optional[Any]:
        """
        Run the complete workflow from images to 3D blockout.

        Args:
            num_slices: Number of slices for 3D reconstruction

        Returns:
            Final mesh object (if Blender available)
        """
        print("=" * 60)
        print("BLENDER AUTOMATED BLOCKING WORKFLOW")
        print("=" * 60)

        check_setup()
        self.context.apply_seed()

        # Step 1: Load images
        with self.context.time_block("load_images"):
            self.load_images()

        # Step 2: Process images
        with self.context.time_block("process_images"):
            self.process_images()

        # Step 3: Analyze shapes
        with self.context.time_block("analyze_shapes"):
            self.analyze_shapes()

        # Step 4: Create 3D blockout or pure backend artifact.
        result = None
        mode = self.config.reconstruction.reconstruction_mode
        if BLENDER_AVAILABLE:
            if self._is_backend_mode(mode):
                with self.context.time_block(f"backend_{mode}"):
                    result = self.run_backend_reconstruction(mode=mode)
            else:
                with self.context.time_block("create_3d_blockout"):
                    result = self.create_3d_blockout(num_slices=num_slices)
        else:
            if self._is_backend_mode(mode):
                with self.context.time_block(f"backend_{mode}"):
                    result = self.run_backend_reconstruction(mode=mode)
            else:
                print("\nShape analysis complete. Run in Blender to create 3D blockout.")
                print(f"Analyzed views: {', '.join(self.shape_analysis.keys())}")
                for view, shapes in self.shape_analysis.items():
                    print(f"  {view}: {len(shapes)} shapes")

        print("=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)

        return result


def example_workflow_with_images(
    front_path: Optional[str] = None,
    side_path: Optional[str] = None,
    top_path: Optional[str] = None,
) -> BlockingWorkflow:
    """
    Run example workflow with provided image paths.

    Args:
        front_path: Path to front view image
        side_path: Path to side view image
        top_path: Path to top view image

    Returns:
        BlockingWorkflow instance
    """
    workflow = BlockingWorkflow(
        front_path=front_path, side_path=side_path, top_path=top_path
    )

    workflow.run_full_workflow(num_slices=12)

    return workflow


def example_workflow_no_images() -> Optional[Any]:
    """
    Run example workflow without images (generates procedural blockout).
    Useful for testing the 3D generation pipeline.
    """
    if not BLENDER_AVAILABLE:
        print("Error: Blender API required for procedural generation")
        return None

    print("=" * 60)
    print("PROCEDURAL BLOCKOUT (No Reference Images)")
    print("=" * 60)

    # Setup scene
    setup_scene(clear_existing=True)

    # Define bounds
    bounds_min = (-2, -2, 0)
    bounds_max = (2, 2, 6)

    # Analyze and place
    analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=12)
    slice_data = analyzer.get_all_slice_data()

    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type="CYLINDER")

    # Join
    joiner = MeshJoiner()
    final_mesh = joiner.join(
        objects,
        target_name="Procedural_Blockout",
        mode="boolean",
    )

    # Setup scene
    add_camera()
    add_lighting()

    print(f"✓ Created procedural blockout: {final_mesh.name}")
    print("=" * 60)

    return final_mesh


if __name__ == "__main__":
    # When run in Blender, execute procedural example
    if BLENDER_AVAILABLE:
        print("Running in Blender - creating procedural blockout...")
        result = example_workflow_no_images()
    else:
        print("Not running in Blender - image analysis only mode")
        print("\nTo use this script:")
        print("1. With reference images in Blender:")
        print(
            "   >>> from blender_blocking.main_integration import example_workflow_with_images"
        )
        print("   >>> example_workflow_with_images('front.png', 'side.png', 'top.png')")
        print("\n2. Procedural generation in Blender:")
        print(
            "   >>> from blender_blocking.main_integration import example_workflow_no_images"
        )
        print("   >>> example_workflow_no_images()")
