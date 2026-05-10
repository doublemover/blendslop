"""
Ground Truth IoU Validation: 3-view vs 12-view vs Ground Truth.

Voxelizes the original vase mesh and compares both reconstruction methods
against ground truth to measure absolute IoU.

Expected results:
- 3-view vs ground truth: ~0.875 IoU
- 12-view vs ground truth: 0.88-0.92 IoU

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_ground_truth_iou.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import pickle
from typing import Optional, Tuple

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "blender_python_packages"))

import bpy
import numpy as np
from mathutils import Vector

from integration.image_processing.image_loader import load_multi_view_auto
from integration.multi_view.visual_hull import MultiViewVisualHull
from integration.blender_ops.raycast_utils import ray_cast_world


def clear_scene() -> None:
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_vase() -> bpy.types.Object:
    """Create the same vase used for turntable generation."""
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2.0, location=(0, 0, 0))
    vase = bpy.context.active_object

    # Enter edit mode and taper
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    # Simple scaling to create vase-like shape
    bpy.ops.transform.resize(value=(1.2, 1.2, 1.0), constraint_axis=(True, True, False))

    bpy.ops.object.mode_set(mode="OBJECT")

    return vase


def voxelize_mesh(
    mesh_obj: bpy.types.Object,
    resolution: int = 128,
    bounds_min: Optional[np.ndarray] = None,
    bounds_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Voxelize a Blender mesh object.

    Args:
        mesh_obj: Blender mesh object
        resolution: Voxel grid resolution
        bounds_min: Min bounds (default: [-2, -2, -2])
        bounds_max: Max bounds (default: [2, 2, 2])

    Returns:
        voxel_grid: Boolean numpy array of shape (resolution, resolution, resolution)
    """
    if bounds_min is None:
        bounds_min = np.array([-2.0, -2.0, -2.0])
    if bounds_max is None:
        bounds_max = np.array([2.0, 2.0, 2.0])

    print(f"\nVoxelizing mesh: {mesh_obj.name}")
    print(f"  Resolution: {resolution}³")
    print(f"  Bounds: {bounds_min} to {bounds_max}")

    # Create voxel grid
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    # Compute voxel size
    voxel_size = (bounds_max - bounds_min) / resolution

    # For each voxel center, check if inside mesh
    start_time = time.time()
    total_voxels = resolution**3
    progress_every = max(total_voxels // 10, 1)
    voxels_checked = 0
    voxels_inside = 0

    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # Voxel center position
                voxel_pos = bounds_min + (np.array([i, j, k]) + 0.5) * voxel_size

                # Check if inside mesh using raycast
                # Cast ray from voxel position in +X direction
                # If odd number of hits, point is inside
                point = Vector((voxel_pos[0], voxel_pos[1], voxel_pos[2]))
                direction = Vector((1, 0, 0))

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
                        f"  Progress: {progress:.0f}% ({voxels_inside:,} voxels inside)"
                    )

    elapsed = time.time() - start_time
    occupancy = voxels_inside / total_voxels

    print(f"\n  ✓ Voxelization complete in {elapsed:.1f}s")
    print(f"  Occupied voxels: {voxels_inside:,} / {total_voxels:,}")
    print(f"  Occupancy: {occupancy*100:.2f}%")

    return voxel_grid


def load_image_as_silhouette(image_path: Path) -> np.ndarray:
    """Load image and convert to binary silhouette."""
    from integration.image_processing.image_loader import load_image

    img = load_image(str(image_path))

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    # Threshold to binary (silhouette is dark on light background)
    silhouette = img < 128

    return silhouette


def reconstruct_3view(
    front_path: Path, side_path: Path, top_path: Path, resolution: int = 128
) -> Tuple[np.ndarray, float]:
    """Reconstruct using 3-view baseline."""
    print("\n" + "=" * 70)
    print("3-VIEW BASELINE RECONSTRUCTION")
    print("=" * 70)

    # Load silhouettes
    front_silhouette = load_image_as_silhouette(front_path)
    side_silhouette = load_image_as_silhouette(side_path)
    top_silhouette = load_image_as_silhouette(top_path)

    # Create Visual Hull
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0]),
    )

    hull.add_view_from_silhouette(front_silhouette, angle=0.0, view_type="lateral")
    hull.add_view_from_silhouette(side_silhouette, angle=90.0, view_type="lateral")
    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    print(f"\n✓ Added 3 views")

    # Reconstruct
    print(f"Reconstructing...")
    start_time = time.time()
    voxel_grid = hull.reconstruct(verbose=False)
    processing_time = time.time() - start_time

    stats = hull.get_stats()
    print(f"✓ Complete in {processing_time:.2f}s")
    print(f"  Occupied: {stats['occupied_voxels']:,} ({stats['occupancy']*100:.2f}%)")

    return voxel_grid, processing_time


def reconstruct_12view(
    turntable_dir: Path, resolution: int = 128
) -> Tuple[np.ndarray, float]:
    """Reconstruct using 12-view turntable."""
    print("\n" + "=" * 70)
    print("12-VIEW ENHANCED RECONSTRUCTION")
    print("=" * 70)

    # Load multi-view sequence
    views_dict = load_multi_view_auto(
        str(turntable_dir), num_views=12, include_top=True
    )

    print(f"\n✓ Loaded {len(views_dict)} views")

    # Create Visual Hull
    hull = MultiViewVisualHull(
        resolution=resolution,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 2.0]),
    )

    # Add all views
    for view_name, (img, angle, view_type) in views_dict.items():
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img

        silhouette = img_gray < 128
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type=view_type)

    # Reconstruct
    print(f"Reconstructing...")
    start_time = time.time()
    voxel_grid = hull.reconstruct(verbose=False)
    processing_time = time.time() - start_time

    stats = hull.get_stats()
    print(f"✓ Complete in {processing_time:.2f}s")
    print(f"  Occupied: {stats['occupied_voxels']:,} ({stats['occupancy']*100:.2f}%)")

    return voxel_grid, processing_time


def compute_iou(voxel_grid_a: np.ndarray, voxel_grid_b: np.ndarray) -> float:
    """Compute IoU between two voxel grids."""
    intersection = np.logical_and(voxel_grid_a, voxel_grid_b).sum()
    union = np.logical_or(voxel_grid_a, voxel_grid_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def main() -> int:
    """Run ground truth IoU validation."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH IoU VALIDATION")
    print("=" * 70)
    print("\nObjective: Validate 3-view and 12-view IoU against ground truth")
    print("Expected: 3-view ~0.875, 12-view 0.88-0.92")

    resolution = 128

    # Paths
    front_path = Path("test_images/vase_front.png")
    side_path = Path("test_images/vase_side.png")
    top_path = Path("test_images/vase_top.png")
    turntable_dir = Path("test_images/turntable_vase/")

    # Verify paths
    if not front_path.exists():
        print(f"\n❌ ERROR: Test images not found")
        return 1

    if not turntable_dir.exists():
        print(f"\n❌ ERROR: Turntable directory not found")
        return 1

    # Create ground truth mesh
    print("\n" + "=" * 70)
    print("STEP 1: Create Ground Truth Voxel Grid")
    print("=" * 70)

    clear_scene()
    vase = create_vase()
    print(f"\n✓ Created vase mesh")

    # Voxelize ground truth
    ground_truth_grid = voxelize_mesh(vase, resolution=resolution)

    # Save ground truth for later use
    cache_path = Path(__file__).resolve().parents[1] / "temp" / "ground_truth_voxels.pkl"
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(ground_truth_grid, f)
    print(f"\n✓ Saved ground truth to {cache_path}")

    # Reconstruct with 3-view
    print("\n" + "=" * 70)
    print("STEP 2: 3-View Reconstruction")
    print("=" * 70)

    voxel_3view, time_3view = reconstruct_3view(
        front_path, side_path, top_path, resolution=resolution
    )

    # Reconstruct with 12-view
    print("\n" + "=" * 70)
    print("STEP 3: 12-View Reconstruction")
    print("=" * 70)

    voxel_12view, time_12view = reconstruct_12view(turntable_dir, resolution=resolution)

    # Compute IoU scores
    print("\n" + "=" * 70)
    print("STEP 4: Compute IoU vs Ground Truth")
    print("=" * 70)

    iou_3view = compute_iou(voxel_3view, ground_truth_grid)
    iou_12view = compute_iou(voxel_12view, ground_truth_grid)
    iou_between = compute_iou(voxel_3view, voxel_12view)

    print(f"\nIoU Results:")
    print(f"  3-view vs ground truth:  {iou_3view:.4f}")
    print(f"  12-view vs ground truth: {iou_12view:.4f}")
    print(f"  3-view vs 12-view:       {iou_between:.4f}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nGround Truth:")
    print(f"  Occupied voxels: {ground_truth_grid.sum():,}")
    print(f"  Occupancy: {ground_truth_grid.mean()*100:.2f}%")

    print(f"\n3-View Baseline:")
    print(f"  Occupied voxels: {voxel_3view.sum():,}")
    print(f"  IoU vs ground truth: {iou_3view:.4f}")
    print(f"  Processing time: {time_3view:.2f}s")

    print(f"\n12-View Enhanced:")
    print(f"  Occupied voxels: {voxel_12view.sum():,}")
    print(f"  IoU vs ground truth: {iou_12view:.4f}")
    print(f"  Processing time: {time_12view:.2f}s")

    print(f"\nImprovement:")
    iou_gain = iou_12view - iou_3view
    print(f"  IoU gain: {iou_gain:+.4f} ({iou_gain/iou_3view*100:+.1f}%)")
    print(f"  Time cost: {time_12view/time_3view:.2f}x")

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if iou_12view >= 0.88 and iou_12view <= 0.92:
        print(f"\n✓ SUCCESS: 12-view IoU {iou_12view:.4f} is within target (0.88-0.92)")
    elif iou_12view > 0.92:
        print(f"\n✓ EXCELLENT: 12-view IoU {iou_12view:.4f} exceeds target!")
    else:
        print(f"\n⚠ WARNING: 12-view IoU {iou_12view:.4f} below target (0.88-0.92)")

    if iou_12view > iou_3view:
        print(f"✓ 12-view improves over 3-view baseline")
    else:
        print(f"⚠ 12-view does not improve over 3-view")

    if time_12view < 80.0:
        print(f"✓ Processing time {time_12view:.1f}s within target (<80s)")
    else:
        print(f"⚠ Processing time {time_12view:.1f}s exceeds target (80s)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
