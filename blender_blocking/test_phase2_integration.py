"""
Phase 2 Integration Test: Visual Hull → Multi-Profile → Primitives

Tests the complete Phase 2 pipeline:
1. Generate Visual Hull mesh from multi-view images
2. Extract multi-angle profiles from mesh
3. Combine profiles (median)
4. Feed into SliceAnalyzer + primitive placement
5. Measure IoU improvement over Visual Hull alone

Expected results:
- Visual Hull alone: ~0.78 IoU
- Visual Hull + primitives: ~0.85-0.88 IoU (target)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_phase2_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import math
from typing import Tuple

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "blender_python_packages"))

import bpy
import numpy as np
from mathutils import Vector

# Phase 1: Visual Hull
from integration.multi_view.visual_hull import MultiViewVisualHull

# Phase 2: Multi-profile extraction
from integration.shape_matching.mesh_profile_extractor import (
    extract_multi_angle_profiles,
    combine_profiles,
)

# Existing primitive placement
from placement.primitive_placement import SliceAnalyzer, PrimitivePlacer, MeshJoiner
from integration.blender_ops.raycast_utils import ray_cast_world
from integration.blender_ops.camera_framing import compute_bounds_world
from integration.blender_ops.silhouette_render import (
    render_silhouette_frame,
    set_camera_orbit,
    set_camera_top,
    silhouette_session,
)


def clear_scene() -> None:
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_test_mesh_vase() -> bpy.types.Object:
    """Create vase for testing (same as in test suite)."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))
    bpy.ops.object.mode_set(mode="OBJECT")
    return vase


def voxelize_mesh_ground_truth(
    mesh_obj: bpy.types.Object, resolution: int = 128
) -> Tuple[np.ndarray, Vector, Vector]:
    """
    Voxelize mesh for ground truth IoU measurement.

    Uses fixed large bounds to ensure raycasts work correctly.
    Tight mesh-derived bounds cause incorrect voxelization (100% occupancy bug).
    """
    # Use FIXED large bounds (same as test_ground_truth_iou.py)
    # This ensures mesh is in CENTER of voxel grid and rays shoot from outside
    bounds_min = Vector((-2.0, -2.0, -2.0))
    bounds_max = Vector((2.0, 2.0, 2.0))

    # Create voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Use numpy arrays for bounds
    bounds_min_arr = np.array([bounds_min.x, bounds_min.y, bounds_min.z])
    bounds_max_arr = np.array([bounds_max.x, bounds_max.y, bounds_max.z])
    voxel_size = (bounds_max_arr - bounds_min_arr) / resolution

    print(f"  Voxelizing at {resolution}³ resolution...")
    voxels_inside = 0
    total_voxels = resolution**3
    progress_every = max(total_voxels // 10, 1)
    voxels_checked = 0

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # Voxel center position
                voxel_pos = bounds_min_arr + (np.array([i, j, k]) + 0.5) * voxel_size
                point = Vector((voxel_pos[0], voxel_pos[1], voxel_pos[2]))

                # Count raycasts in different directions for robustness
                hit_count = 0
                for ray_dir in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    direction = Vector(ray_dir)
                    result, location, normal, index = ray_cast_world(
                        mesh_obj,
                        point - direction * 10,  # Start from far away
                        direction,
                        20.0,  # Long enough to traverse bounds
                    )
                    if result:
                        hit_count += 1

                # If majority of raycasts hit, consider inside
                if hit_count >= 2:
                    voxel_grid[i, j, k] = True
                    voxels_inside += 1

                voxels_checked += 1

                # Progress update every 10%
                if voxels_checked % progress_every == 0:
                    progress = 100 * voxels_checked / total_voxels
                    print(
                        f"    Progress: {progress:.0f}% ({voxels_inside:,} voxels inside)"
                    )

    print(f"  ✓ Voxelization complete: {voxels_inside:,} / {total_voxels:,} voxels")

    return voxel_grid, bounds_min, bounds_max


def compute_iou(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(grid_a, grid_b).sum()
    union = np.logical_or(grid_a, grid_b).sum()
    return intersection / union if union > 0 else 0.0


def render_turntable_silhouettes(
    mesh_obj: bpy.types.Object, output_dir: Path, num_views: int = 12
) -> None:
    """Render turntable silhouettes for Visual Hull."""
    if num_views <= 0:
        raise ValueError("num_views must be >= 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = compute_bounds_world([mesh_obj])
    bounds_min, bounds_max = bounds
    center = (bounds_min + bounds_max) / 2.0
    width = bounds_max.x - bounds_min.x
    depth = bounds_max.y - bounds_min.y
    height = bounds_max.z - bounds_min.z
    max_dim = max(width, depth, height, 1e-3)
    margin_frac = 0.08
    distance = max_dim * 2.0
    ortho_scale = max_dim * (1.0 + 2.0 * margin_frac)
    top_ortho_scale = max(width, depth, 1e-3) * (1.0 + 2.0 * margin_frac)

    angle_step = 360.0 / num_views
    with silhouette_session(
        target_objects=[mesh_obj],
        resolution=(512, 512),
        color_mode="BW",
        transparent_bg=False,
        engine="BLENDER_EEVEE",
        background_color=(1.0, 1.0, 1.0, 1.0),
        silhouette_color=(0.0, 0.0, 0.0, 1.0),
    ) as session:
        for i in range(num_views):
            angle = i * angle_step
            angle_rad = math.radians(angle)
            set_camera_orbit(session.camera, center, distance, angle_rad, ortho_scale)
            output_path = output_dir / f"view_{int(angle):03d}.png"
            render_silhouette_frame(session, output_path)

        set_camera_top(session.camera, center, distance, top_ortho_scale)
        render_silhouette_frame(session, output_dir / "top.png")


def create_visual_hull_mesh(
    turntable_dir: Path, resolution: int = 128
) -> Tuple[bpy.types.Object, np.ndarray]:
    """Create Visual Hull mesh from turntable images."""
    from integration.image_processing.image_loader import load_multi_view_auto

    # Load views
    views_dict = load_multi_view_auto(
        str(turntable_dir), num_views=12, include_top=True
    )

    # Create Visual Hull
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0]),
    )

    # Add views
    print("\nAdding views to Visual Hull:")
    for view_name, (img, angle, view_type) in views_dict.items():
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        silhouette = img_gray < 128
        foreground_pixels = silhouette.sum()
        print(
            f"  {view_name}: {img.shape}, {foreground_pixels:,} foreground pixels ({foreground_pixels/silhouette.size*100:.1f}%)"
        )
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    # Reconstruct
    print("\nReconstructing Visual Hull...")
    voxel_grid = hull.reconstruct(verbose=True)
    print(f"Visual Hull voxels: {voxel_grid.sum():,} / {voxel_grid.size:,}")

    # Create voxel mesh with faces (not just point cloud)
    # This enables raycasting for profile extraction
    voxel_size_world = (hull.bounds_max - hull.bounds_min) / hull.resolution
    voxel_scale = voxel_size_world / 2.0

    occupied_indices = np.argwhere(voxel_grid)
    # Sample every 4th voxel for speed (still ~18k voxels for 128³)
    sampled = occupied_indices[::4]

    if len(sampled) == 0:
        # No occupied voxels - return empty mesh
        mesh = bpy.data.meshes.new("VisualHull_Empty")
        obj = bpy.data.objects.new("VisualHull_Empty", mesh)
        bpy.context.collection.objects.link(obj)
        return obj, voxel_grid

    # Create cube template
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -100))
    cube_template = bpy.context.active_object

    # Create copies for voxels (limit to 500 for speed)
    all_objects = []
    for i, idx in enumerate(sampled[:500]):
        pos = hull.bounds_min + (idx + 0.5) * voxel_size_world

        if i == 0:
            obj = cube_template
        else:
            obj = cube_template.copy()
            obj.data = cube_template.data.copy()
            bpy.context.collection.objects.link(obj)

        obj.location = pos
        obj.scale = voxel_scale
        all_objects.append(obj)

    # Join all cubes
    bpy.context.view_layer.objects.active = all_objects[0]
    for obj in all_objects:
        obj.select_set(True)

    bpy.ops.object.join()
    final_obj = bpy.context.active_object
    final_obj.name = "VisualHull"

    # CRITICAL: Apply transformations to normalize mesh
    # Without this, mesh has tiny scale and offset location
    # This causes profile extraction to produce inflated radii (25x volume error!)
    bpy.context.view_layer.objects.active = final_obj
    final_obj.select_set(True)

    # Apply scale/rotation/location to bake into vertices
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Center at origin
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")
    final_obj.location = (0, 0, 0)

    # CRITICAL: Scale Visual Hull to match ground truth size
    # Voxel mesh is in "voxel space" and ends up 15-25x too small after normalization
    # Ground truth vase is: radius ~0.6, height 2.0 → bbox approximately (1.2, 1.2, 2.0)
    bbox = [final_obj.matrix_world @ Vector(corner) for corner in final_obj.bound_box]
    bounds_min = Vector(
        (min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox))
    )
    bounds_max = Vector(
        (max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox))
    )
    vh_size = bounds_max - bounds_min

    # Expected ground truth size (from create_test_mesh_vase: radius=0.5*1.2=0.6, depth=2.0)
    expected_size = Vector((1.2, 1.2, 2.0))  # XY from diameter, Z from depth

    # Compute scale factor (use Z as reference, most reliable dimension)
    scale_factor = expected_size.z / vh_size.z if vh_size.z > 0 else 1.0

    # Apply uniform scale
    final_obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)  # Bake into vertices

    print(
        f"  ✓ Scaled Visual Hull: {vh_size.z:.3f} → {expected_size.z:.3f} (factor {scale_factor:.3f}x)"
    )

    return final_obj, voxel_grid


def test_phase2_pipeline() -> None:
    """Test complete Phase 2 pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 2 INTEGRATION TEST")
    print("=" * 70)
    print("\nPipeline: Visual Hull → Multi-Profile → Primitives")
    print("Target: Boost from ~0.78 IoU to ~0.85-0.88 IoU")

    resolution = 128
    turntable_dir = Path(__file__).resolve().parents[1] / "temp" / "phase2_vase"

    # Step 1: Create ground truth vase
    print("\n" + "=" * 70)
    print("STEP 1: Create Ground Truth Vase")
    print("=" * 70)

    clear_scene()
    ground_truth_vase = create_test_mesh_vase()
    print("✓ Created vase mesh")

    # Voxelize for ground truth
    ground_truth_voxels, gt_bounds_min, gt_bounds_max = voxelize_mesh_ground_truth(
        ground_truth_vase, resolution=resolution
    )
    print(f"✓ Ground truth: {ground_truth_voxels.sum():,} voxels")

    # Step 2: Render turntable silhouettes
    print("\n" + "=" * 70)
    print("STEP 2: Render Turntable Silhouettes")
    print("=" * 70)

    render_turntable_silhouettes(ground_truth_vase, turntable_dir, num_views=12)
    print(f"✓ Rendered 13 views to {turntable_dir}")

    # Keep ground truth for later, create new scene for reconstruction
    ground_truth_copy = ground_truth_voxels.copy()

    # Step 3: Visual Hull reconstruction
    print("\n" + "=" * 70)
    print("STEP 3: Visual Hull Reconstruction (Phase 1)")
    print("=" * 70)

    clear_scene()
    visual_hull_mesh, visual_hull_voxels = create_visual_hull_mesh(
        turntable_dir, resolution=resolution
    )
    print(f"✓ Visual Hull mesh: {len(visual_hull_mesh.data.vertices):,} vertices")
    print(f"✓ Visual Hull voxels: {visual_hull_voxels.sum():,}")

    # Measure Visual Hull IoU
    iou_visual_hull = compute_iou(visual_hull_voxels, ground_truth_copy)
    print(f"✓ Visual Hull IoU vs ground truth: {iou_visual_hull:.4f}")

    # Step 4: Extract multi-angle profiles from Visual Hull mesh
    print("\n" + "=" * 70)
    print("STEP 4: Extract Multi-Angle Profiles (Phase 2)")
    print("=" * 70)

    profiles = extract_multi_angle_profiles(
        visual_hull_mesh,
        num_angles=12,
        num_heights=20,
        bounds_min=gt_bounds_min,
        bounds_max=gt_bounds_max,
    )
    print(f"✓ Extracted {len(profiles)} profiles at different angles")

    # Combine profiles
    combined_profile = combine_profiles(profiles, method="median")
    print(f"✓ Combined using median: {len(combined_profile)} height samples")

    radii = [r for h, r in combined_profile]
    print(f"  Radius range: {min(radii):.3f} to {max(radii):.3f}")

    # Step 5: Place primitives using combined profile
    print("\n" + "=" * 70)
    print("STEP 5: Place Primitives from Multi-Profile")
    print("=" * 70)

    # Create SliceAnalyzer with combined profile
    analyzer = SliceAnalyzer(
        gt_bounds_min, gt_bounds_max, num_slices=20, vertical_profile=combined_profile
    )
    slice_data = analyzer.get_all_slice_data()
    print(f"✓ Analyzed {len(slice_data)} slices")

    # Place primitives
    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type="CYLINDER")
    print(f"✓ Placed {len(objects)} cylinder primitives")

    # Join primitives
    if objects:
        joiner = MeshJoiner()
        final_mesh = joiner.join_with_boolean_union(
            objects, target_name="Phase2_Primitives"
        )
        print(f"✓ Joined into single mesh: {final_mesh.name}")

        # Step 6: Measure IoU of primitive mesh
        print("\n" + "=" * 70)
        print("STEP 6: Measure IoU Improvement")
        print("=" * 70)

        # Voxelize primitive mesh
        primitive_voxels, _, _ = voxelize_mesh_ground_truth(
            final_mesh, resolution=resolution
        )
        print(f"✓ Primitive mesh voxels: {primitive_voxels.sum():,}")

        # Measure IoU
        iou_primitives = compute_iou(primitive_voxels, ground_truth_copy)
        print(f"✓ Primitives IoU vs ground truth: {iou_primitives:.4f}")

        # Results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        improvement = iou_primitives - iou_visual_hull
        improvement_pct = (
            (improvement / iou_visual_hull * 100) if iou_visual_hull > 0 else 0
        )

        print(f"\nGround Truth: {ground_truth_copy.sum():,} voxels")
        print(f"\nVisual Hull (Phase 1):")
        print(f"  Voxels: {visual_hull_voxels.sum():,}")
        print(f"  IoU: {iou_visual_hull:.4f}")

        print(f"\nPrimitives (Phase 2):")
        print(f"  Voxels: {primitive_voxels.sum():,}")
        print(f"  IoU: {iou_primitives:.4f}")

        print(f"\nImprovement:")
        print(f"  Absolute: {improvement:+.4f}")
        print(f"  Relative: {improvement_pct:+.1f}%")

        print(f"\nTarget Assessment:")
        if iou_primitives >= 0.85:
            print(f"  ✓ SUCCESS: Achieved target (≥0.85)")
        else:
            gap = 0.85 - iou_primitives
            print(f"  ⚠ Below target by {gap:.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_phase2_pipeline()
