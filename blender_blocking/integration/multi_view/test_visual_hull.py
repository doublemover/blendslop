"""
Test suite for Multi-View Visual Hull reconstruction.

Tests:
1. Basic instantiation
2. Single view reconstruction
3. 3-view reconstruction (baseline)
4. 8-view reconstruction
5. 12-view reconstruction
6. Camera projection accuracy
"""

from __future__ import annotations

import numpy as np
import sys
import unittest
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration.multi_view.visual_hull import MultiViewVisualHull, CameraView


def generate_cylinder_silhouette(
    width: int = 128, height: int = 128, radius_ratio: float = 0.4, angle: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic cylinder silhouette at given angle.

    For turntable views, cylinder appears as rectangle (constant width).

    Args:
        width: Image width
        height: Image height
        radius_ratio: Cylinder radius as fraction of image size
        angle: Viewing angle (for lateral views, doesn't affect cylinder)

    Returns:
        Binary silhouette (HxW)
    """
    silhouette = np.zeros((height, width), dtype=bool)

    # Cylinder appears as vertical rectangle in lateral view
    radius_pixels = int(width * radius_ratio)
    center_x = width // 2

    # Fill rectangle
    x_min = center_x - radius_pixels
    x_max = center_x + radius_pixels

    silhouette[:, max(0, x_min) : min(width, x_max)] = True

    return silhouette


def generate_top_cylinder_silhouette(
    width: int = 128, height: int = 128, radius_ratio: float = 0.4
) -> np.ndarray:
    """
    Generate synthetic cylinder silhouette from top view (circular).

    Args:
        width: Image width
        height: Image height
        radius_ratio: Cylinder radius as fraction of image size

    Returns:
        Binary silhouette (HxW)
    """
    silhouette = np.zeros((height, width), dtype=bool)

    center_x = width // 2
    center_y = height // 2
    radius = int(width * radius_ratio)

    # Create circle
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    silhouette[mask] = True

    return silhouette


def build_basic_hull(
    resolution: int = 16, image_size: int = 48, radius_ratio: float = 0.35
) -> MultiViewVisualHull:
    """Build a small multi-view hull for deterministic comparisons."""
    hull = MultiViewVisualHull(resolution=resolution)

    for angle in [0, 90]:
        silhouette = generate_cylinder_silhouette(
            image_size, image_size, radius_ratio=radius_ratio
        )
        hull.add_view_from_silhouette(
            silhouette, angle=float(angle), view_type="lateral"
        )

    top_silhouette = generate_top_cylinder_silhouette(
        image_size, image_size, radius_ratio=radius_ratio
    )
    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    return hull


def test_basic_instantiation() -> bool:
    """Test basic Visual Hull creation."""
    print("\n" + "=" * 70)
    print("TEST: Basic Instantiation")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=32)

    print("  OK Created Visual Hull")
    print(f"    Resolution: {hull.resolution}")
    print(f"    Bounds: {hull.bounds_min} to {hull.bounds_max}")

    assert hull.resolution == 32
    assert len(hull.views) == 0

    print("\nOK Basic instantiation test PASSED")
    return True


def test_camera_view() -> bool:
    """Test CameraView creation and projection."""
    print("\n" + "=" * 70)
    print("TEST: Camera View Creation")
    print("=" * 70)

    # Create simple silhouette (circle)
    silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)

    view = CameraView(silhouette=silhouette, angle=0.0, view_type="top")
    list_view = CameraView(silhouette=silhouette.tolist(), angle=0.0, view_type="top")

    print("  OK Created camera view")
    print(f"    Angle: {view.angle}°")
    print(f"    Type: {view.view_type}")
    print(f"    Size: {view.width}x{view.height}")

    # Test projection
    test_point = np.array([0.0, 0.0, 0.0])  # Center point
    u, v = view.project_point(test_point, (np.array([-1, -1, -1]), np.array([1, 1, 1])))

    print(f"  OK Projection test: (0,0,0) -> ({u}, {v})")
    print(f"    Expected near: ({view.width//2}, {view.height//2})")

    # Should project near center
    assert abs(u - view.width // 2) < 5
    assert abs(v - view.height // 2) < 5
    assert list_view.silhouette.shape == silhouette.shape

    print("\nOK Camera view test PASSED")
    return True


def test_projection_non_centered_bounds() -> bool:
    """Ensure projection uses bounds center for rotation."""
    print("\n" + "=" * 70)
    print("TEST: Projection with Non-Centered Bounds")
    print("=" * 70)

    silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
    view = CameraView(silhouette=silhouette, angle=45.0, view_type="lateral")

    bounds_min = np.array([2.0, 3.0, -1.0])
    bounds_max = np.array([4.0, 5.0, 1.0])
    center = (bounds_min + bounds_max) / 2.0

    u, v = view.project_point(center, (bounds_min, bounds_max))
    print(f"  Projection of center {center} -> ({u}, {v})")

    assert abs(u - view.width // 2) < 5
    assert abs(v - view.height // 2) < 5

    print("\nOK Non-centered bounds projection test PASSED")
    return True


def test_projection_respects_image_bounds() -> bool:
    """Ensure projection can map world bounds to the selected silhouette bbox."""
    print("\n" + "=" * 70)
    print("TEST: Projection Respects Image Bounds")
    print("=" * 70)

    silhouette = np.zeros((64, 64), dtype=bool)
    silhouette[16:48, 16:48] = True
    view = CameraView(
        silhouette=silhouette,
        angle=0.0,
        view_type="lateral",
        image_bounds=(16.0, 16.0, 48.0, 48.0),
    )

    bounds_min = np.array([-1.0, -1.0, -1.0])
    bounds_max = np.array([1.0, 1.0, 1.0])
    center = np.array([0.0, 0.0, 0.0])
    lower_left = np.array([-1.0, 0.0, -1.0])
    upper_right = np.array([1.0, 0.0, 1.0])

    center_u, center_v = view.project_point(center, (bounds_min, bounds_max))
    left_u, bottom_v = view.project_point(lower_left, (bounds_min, bounds_max))
    right_u, top_v = view.project_point(upper_right, (bounds_min, bounds_max))

    print(f"  Center projection -> ({center_u}, {center_v})")
    print(f"  ROI corners -> ({left_u}, {bottom_v}) and ({right_u}, {top_v})")

    assert abs(center_u - 32) <= 1
    assert abs(center_v - 32) <= 1
    assert left_u == 16
    assert top_v == 16
    assert right_u == 47
    assert bottom_v == 47

    print("\nOK Image-bounds projection test PASSED")
    return True


def test_side_projection_uses_rotated_extent() -> bool:
    """Ensure side views normalize against rotated X/Y footprint extent."""
    print("\n" + "=" * 70)
    print("TEST: Side Projection Uses Rotated Extent")
    print("=" * 70)

    silhouette = np.ones((64, 64), dtype=bool)
    view = CameraView(silhouette=silhouette, angle=90.0, view_type="lateral")

    bounds_min = np.array([-2.0, -0.5, -1.0])
    bounds_max = np.array([2.0, 0.5, 1.0])
    left_depth = np.array([0.0, -0.5, 0.0])
    right_depth = np.array([0.0, 0.5, 0.0])

    left_u, _left_v = view.project_point(left_depth, (bounds_min, bounds_max))
    right_u, _right_v = view.project_point(right_depth, (bounds_min, bounds_max))

    print(f"  Side depth endpoints -> u={left_u}, u={right_u}")

    assert left_u == 0
    assert right_u == 63

    print("\nOK Side rotated extent test PASSED")
    return True


def test_single_view_reconstruction() -> bool:
    """Test reconstruction with single view (should give cone)."""
    print("\n" + "=" * 70)
    print("TEST: Single View Reconstruction")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=32)

    # Add single lateral view (cylinder from side)
    silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(silhouette, angle=0.0, view_type="lateral")

    print(f"  Added 1 view")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=False)

    print("  OK Reconstruction complete")

    stats = hull.get_stats()
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats["num_views"] == 1
    assert stats["occupied_voxels"] > 0

    print("\nOK Single view test PASSED")
    return True


def test_three_view_reconstruction() -> bool:
    """Test reconstruction with 3 orthogonal views (baseline)."""
    print("\n" + "=" * 70)
    print("TEST: 3-View Reconstruction (Baseline)")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 3 orthogonal lateral views (0°, 90°, 180°)
    for angle in [0, 90, 180]:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(
            silhouette, angle=float(angle), view_type="lateral"
        )

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    print(f"  Added 4 views (3 lateral + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats["num_views"] == 4
    assert stats["occupied_voxels"] > 0

    # Extract surface points
    points = hull.extract_mesh_points(surface_only=True)
    print(f"    Surface points: {len(points):,}")

    print("\nOK 3-view reconstruction test PASSED")
    return True


def test_eight_view_reconstruction() -> bool:
    """Test reconstruction with 8 views (45° spacing)."""
    print("\n" + "=" * 70)
    print("TEST: 8-View Reconstruction")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 8 lateral views (45° spacing)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(
            silhouette, angle=float(angle), view_type="lateral"
        )

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    print(f"  Added 9 views (8 lateral at 45° + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats["num_views"] == 9
    assert stats["occupied_voxels"] > 0

    print("\nOK 8-view reconstruction test PASSED")
    return True


def test_twelve_view_reconstruction() -> bool:
    """Test reconstruction with 12 views (30° spacing) - target configuration."""
    print("\n" + "=" * 70)
    print("TEST: 12-View Reconstruction (Target Configuration)")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=32)

    # Add 12 lateral views (30° spacing)
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    for angle in angles:
        silhouette = generate_cylinder_silhouette(64, 64, radius_ratio=0.3)
        hull.add_view_from_silhouette(
            silhouette, angle=float(angle), view_type="lateral"
        )

    # Add top view
    top_silhouette = generate_top_cylinder_silhouette(64, 64, radius_ratio=0.3)
    hull.add_view_from_silhouette(top_silhouette, view_type="top")

    print(f"  Added 13 views (12 lateral at 30° + 1 top)")

    # Reconstruct
    voxel_grid = hull.reconstruct(verbose=True)

    stats = hull.get_stats()
    print(f"\n  Final statistics:")
    print(f"    Occupied voxels: {stats['occupied_voxels']:,}")
    print(f"    Occupancy: {stats['occupancy']*100:.2f}%")

    assert stats["num_views"] == 13
    assert stats["occupied_voxels"] > 0

    # Extract points
    points = hull.extract_mesh_points(surface_only=True)
    print(f"    Surface points: {len(points):,}")

    print("\nOK 12-view reconstruction test PASSED")
    return True


def test_vectorized_matches_loop() -> bool:
    """Ensure vectorized reconstruction matches loop baseline."""
    print("\n" + "=" * 70)
    print("TEST: Vectorized vs Loop Reconstruction")
    print("=" * 70)

    hull_vec = build_basic_hull()
    grid_vec = hull_vec.reconstruct(verbose=False, use_vectorized=True)

    hull_loop = build_basic_hull()
    grid_loop = hull_loop.reconstruct(verbose=False, use_vectorized=False)

    assert np.array_equal(grid_vec, grid_loop)

    print("\nOK Vectorized vs loop equivalence test PASSED")
    return True


def test_chunked_matches_full() -> bool:
    """Ensure chunked vectorized reconstruction matches full-grid."""
    print("\n" + "=" * 70)
    print("TEST: Chunked vs Full Reconstruction")
    print("=" * 70)

    hull_full = build_basic_hull()
    grid_full = hull_full.reconstruct(verbose=False, use_vectorized=True)

    hull_chunked = build_basic_hull()
    grid_chunked = hull_chunked.reconstruct(
        verbose=False, use_vectorized=True, chunk_size=4
    )

    assert np.array_equal(grid_full, grid_chunked)

    print("\nOK Chunked vs full equivalence test PASSED")
    return True


def test_empty_silhouette_early_exit() -> bool:
    """Ensure empty silhouettes short-circuit to empty grid."""
    print("\n" + "=" * 70)
    print("TEST: Empty Silhouette Early Exit")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=16)
    empty = np.zeros((32, 32), dtype=bool)
    hull.add_view_from_silhouette(empty, angle=0.0, view_type="lateral")

    grid = hull.reconstruct(verbose=False, use_vectorized=True)
    assert grid.sum() == 0

    print("\nOK Empty silhouette early exit test PASSED")
    return True


def test_surface_voxel_extraction() -> bool:
    """Ensure surface voxel extraction behaves on simple grids."""
    print("\n" + "=" * 70)
    print("TEST: Surface Voxel Extraction")
    print("=" * 70)

    hull = MultiViewVisualHull(resolution=3)

    full = np.ones((3, 3, 3), dtype=bool)
    surface_full = hull._extract_surface_voxels(full)
    assert int(surface_full.sum()) == 26

    single = np.zeros((3, 3, 3), dtype=bool)
    single[1, 1, 1] = True
    surface_single = hull._extract_surface_voxels(single)
    assert int(surface_single.sum()) == 1

    empty = np.zeros((3, 3, 3), dtype=bool)
    surface_empty = hull._extract_surface_voxels(empty)
    assert int(surface_empty.sum()) == 0

    print("\nOK Surface voxel extraction test PASSED")
    return True


def run_all_tests() -> int:
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("MULTI-VIEW VISUAL HULL TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Instantiation", test_basic_instantiation),
        ("Camera View", test_camera_view),
        ("Projection Non-Centered Bounds", test_projection_non_centered_bounds),
        ("Projection Image Bounds", test_projection_respects_image_bounds),
        ("Side Projection Rotated Extent", test_side_projection_uses_rotated_extent),
        ("Single View Reconstruction", test_single_view_reconstruction),
        ("3-View Reconstruction", test_three_view_reconstruction),
        ("8-View Reconstruction", test_eight_view_reconstruction),
        ("12-View Reconstruction", test_twelve_view_reconstruction),
        ("Vectorized vs Loop", test_vectorized_matches_loop),
        ("Chunked vs Full", test_chunked_matches_full),
        ("Empty Silhouette Early Exit", test_empty_silhouette_early_exit),
        ("Surface Voxel Extraction", test_surface_voxel_extraction),
    ]

    results = {}
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\nFAIL: {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results[name] = False

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name:.<50} {status}")

    print("=" * 70)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nTESTS FAILED")
        return 1
    else:
        print("\nALL TESTS PASSED")
        return 0


class TestVisualHull(unittest.TestCase):
    def test_basic_instantiation(self) -> None:
        self.assertTrue(test_basic_instantiation())

    def test_camera_view(self) -> None:
        self.assertTrue(test_camera_view())

    def test_projection_non_centered_bounds(self) -> None:
        self.assertTrue(test_projection_non_centered_bounds())

    def test_projection_respects_image_bounds(self) -> None:
        self.assertTrue(test_projection_respects_image_bounds())

    def test_side_projection_uses_rotated_extent(self) -> None:
        self.assertTrue(test_side_projection_uses_rotated_extent())

    def test_single_view_reconstruction(self) -> None:
        self.assertTrue(test_single_view_reconstruction())

    def test_three_view_reconstruction(self) -> None:
        self.assertTrue(test_three_view_reconstruction())

    def test_eight_view_reconstruction(self) -> None:
        self.assertTrue(test_eight_view_reconstruction())

    def test_twelve_view_reconstruction(self) -> None:
        self.assertTrue(test_twelve_view_reconstruction())

    def test_vectorized_matches_loop(self) -> None:
        self.assertTrue(test_vectorized_matches_loop())

    def test_chunked_matches_full(self) -> None:
        self.assertTrue(test_chunked_matches_full())

    def test_empty_silhouette_early_exit(self) -> None:
        self.assertTrue(test_empty_silhouette_early_exit())

    def test_surface_voxel_extraction(self) -> None:
        self.assertTrue(test_surface_voxel_extraction())


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
