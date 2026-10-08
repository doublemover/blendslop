"""
Multi-View Visual Hull Reconstruction.

Extends current 3-view pipeline to support N views (typically 8-12) for higher accuracy.
Based on research document: MULTI_VIEW_RECONSTRUCTION_RESEARCH.md

Algorithm:
1. Initialize 3D voxel grid (128³ or 256³)
2. For each of N views:
   - Back-project silhouette into 3D space (silhouette cone)
   - Mark voxels inside this view's cone
3. Intersect all N cones:
   - Voxel is INSIDE if inside ALL N silhouette cones
   - Voxel is OUTSIDE if outside ANY silhouette cone
4. Extract mesh from final voxel grid (Marching Cubes)

Expected performance:
- 3 views: ~30s
- 12 views: ~60-80s
- Complexity: O(R³ × N) where R = voxel resolution, N = views

References:
- Research doc: MULTI_VIEW_RECONSTRUCTION_RESEARCH.md
- Visual Hull: https://en.wikipedia.org/wiki/Visual_hull
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class CameraView:
    """
    Represents a single camera view for Visual Hull reconstruction.

    Attributes:
        silhouette: 2D binary silhouette (HxW, 1=object, 0=background)
        angle: Rotation angle in degrees (for turntable setup)
        view_type: 'lateral' or 'top'
        camera_distance: Distance from rotation center
        focal_length: Camera focal length (optional, for calibration)
    """

    def __init__(
        self,
        silhouette: np.ndarray,
        angle: float = 0.0,
        view_type: str = "lateral",
        camera_distance: float = 1.0,
        focal_length: Optional[float] = None,
        image_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Initialize camera view.

        Args:
            silhouette: Binary silhouette mask (HxW)
            angle: Turntable rotation angle in degrees (0-360)
            view_type: 'lateral' (horizontal) or 'top' (overhead)
            camera_distance: Distance from turntable center
            focal_length: Camera focal length (optional)
        """
        silhouette = np.asarray(silhouette)
        if silhouette.ndim != 2:
            raise ValueError("silhouette must be a 2D array (HxW)")

        self.silhouette = silhouette.astype(bool)
        self.angle = angle
        self.view_type = view_type
        self.camera_distance = camera_distance
        self.focal_length = focal_length or camera_distance  # Default to orthographic

        self.height, self.width = silhouette.shape
        self.image_bounds = self._coerce_image_bounds(image_bounds)

    def _coerce_image_bounds(
        self, image_bounds: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        if image_bounds is None:
            return (0.0, 0.0, float(self.width), float(self.height))
        x0, y0, x1, y1 = (float(value) for value in image_bounds)
        x0 = max(0.0, min(float(self.width - 1), x0))
        y0 = max(0.0, min(float(self.height - 1), y0))
        x1 = max(x0 + 1.0, min(float(self.width), x1))
        y1 = max(y0 + 1.0, min(float(self.height), y1))
        return (x0, y0, x1, y1)

    def is_point_in_silhouette_cone(
        self, point_3d: np.ndarray, voxel_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """
        Check if a 3D point projects inside this view's silhouette.

        Args:
            point_3d: 3D point (x, y, z)
            voxel_bounds: (min_bounds, max_bounds) for voxel grid

        Returns:
            True if point is inside silhouette cone, False otherwise
        """
        # Project 3D point to 2D silhouette coordinates
        u, v = self.project_point(point_3d, voxel_bounds)

        # Check if projection is within image bounds
        if u < 0 or u >= self.width or v < 0 or v >= self.height:
            return False

        # Check if projection falls inside silhouette
        return self.silhouette[v, u]

    def project_point(
        self, point_3d: np.ndarray, voxel_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[int, int]:
        """
        Project 3D point to 2D silhouette coordinates.

        Uses simplified camera model for turntable setup:
        - Orthographic projection (parallel rays)
        - Camera rotates around object (turntable)
        - Object centered at origin

        Args:
            point_3d: 3D point (x, y, z) in voxel space
            voxel_bounds: (min_bounds, max_bounds) for normalization

        Returns:
            (u, v) pixel coordinates in silhouette
        """
        min_bounds, max_bounds = voxel_bounds

        center = (min_bounds + max_bounds) / 2.0

        if self.view_type == "top":
            # Top view: camera above, looking down
            # Project (x, y, z) -> (x, y)
            world_x = point_3d[0]
            world_y = point_3d[1]
        else:
            # Lateral view: rotate point to camera's reference frame
            angle_rad = math.radians(self.angle)

            # Rotate point around Z axis by -angle (inverse of camera rotation)
            cos_a = math.cos(-angle_rad)
            sin_a = math.sin(-angle_rad)

            local_x = point_3d[0] - center[0]
            local_y = point_3d[1] - center[1]

            rotated_x = local_x * cos_a - local_y * sin_a
            rotated_y = local_x * sin_a + local_y * cos_a

            # After rotation, camera looks along +Y axis
            # Project (rotated_x, rotated_y, z) -> (rotated_x, z)
            world_x = rotated_x + center[0]
            world_y = point_3d[2]  # Z becomes vertical in image

        # Normalize to [0, 1] based on the projected horizontal extent.  For
        # side views this is the rotated X/Y footprint, not always X bounds.
        x_min, x_max = self.projected_horizontal_bounds(min_bounds, max_bounds)
        x_range = x_max - x_min
        y_range_idx = 1 if self.view_type == "top" else 2
        y_range = max_bounds[y_range_idx] - min_bounds[y_range_idx]

        x_norm = (world_x - x_min) / x_range if x_range > 0 else 0.5
        y_norm = (world_y - min_bounds[y_range_idx]) / y_range if y_range > 0 else 0.5
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        x0, y0, x1, y1 = self.image_bounds
        u_span = max(1.0, (x1 - x0) - 1.0)
        v_span = max(1.0, (y1 - y0) - 1.0)

        # Map to pixel coordinates.  Bounds are exclusive x1/y1 values, so a
        # normalized coordinate of 1.0 lands on the last pixel inside the ROI.
        u = int(x0 + x_norm * u_span + 1e-9)
        v = int(y0 + (1.0 - y_norm) * v_span + 1e-9)  # Flip Y (image origin top-left)

        return (u, v)

    def projected_horizontal_bounds(
        self, min_bounds: np.ndarray, max_bounds: np.ndarray
    ) -> Tuple[float, float]:
        """Return the world-space horizontal projection extent for this view."""
        if self.view_type == "top":
            return float(min_bounds[0]), float(max_bounds[0])

        center = (min_bounds + max_bounds) / 2.0
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        corners = (
            (min_bounds[0], min_bounds[1]),
            (min_bounds[0], max_bounds[1]),
            (max_bounds[0], min_bounds[1]),
            (max_bounds[0], max_bounds[1]),
        )
        projected = []
        for x, y in corners:
            local_x = float(x - center[0])
            local_y = float(y - center[1])
            projected.append(local_x * cos_a - local_y * sin_a + float(center[0]))
        return min(projected), max(projected)


class MultiViewVisualHull:
    """
    Multi-view Visual Hull reconstruction using voxel intersection.

    Supports N camera views (typically 8-12) to achieve higher accuracy
    than 3-view baseline.
    """

    def __init__(
        self,
        resolution: int = 128,
        bounds_min: Optional[np.ndarray] = None,
        bounds_max: Optional[np.ndarray] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Initialize Visual Hull reconstructor.

        Args:
            resolution: Voxel grid resolution (default: 128)
            bounds_min: Minimum bounds (x, y, z), default: (-1, -1, -1)
            bounds_max: Maximum bounds (x, y, z), default: (1, 1, 1)
            chunk_size: Optional z-slab chunk size for reconstruction
        """
        if resolution < 1:
            raise ValueError("resolution must be >= 1")

        self.resolution = resolution
        self.bounds_min = (
            np.array(bounds_min)
            if bounds_min is not None
            else np.array([-1.0, -1.0, -1.0])
        )
        self.bounds_max = (
            np.array(bounds_max)
            if bounds_max is not None
            else np.array([1.0, 1.0, 1.0])
        )

        if np.any(self.bounds_max <= self.bounds_min):
            raise ValueError("bounds_max must be greater than bounds_min on all axes")

        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        # Voxel grid will be created during reconstruction
        self.voxel_grid = None
        self.views: List[CameraView] = []
        self.chunk_size = chunk_size

    def add_view(self, view: CameraView) -> None:
        """Add a camera view to the reconstruction."""
        self.views.append(view)

    def add_view_from_silhouette(
        self,
        silhouette: np.ndarray,
        angle: float = 0.0,
        view_type: str = "lateral",
        camera_distance: float = 1.0,
        image_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Convenience method to add a view from silhouette image.

        Args:
            silhouette: Binary silhouette mask
            angle: Rotation angle in degrees
            view_type: 'lateral' or 'top'
            camera_distance: Distance from center
        """
        view = CameraView(
            silhouette,
            angle,
            view_type,
            camera_distance,
            image_bounds=image_bounds,
        )
        self.add_view(view)

    def reconstruct(
        self,
        verbose: bool = True,
        *,
        use_vectorized: bool = True,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reconstruct 3D volume using multi-view Visual Hull algorithm.

        Args:
            verbose: Whether to print progress output.
            use_vectorized: Use vectorized projection instead of triple nested loops.
            chunk_size: Optional z-slab chunk size for vectorized reconstruction.

        Returns:
            3D voxel grid (resolution³) with 1=inside, 0=outside
        """
        if len(self.views) == 0:
            raise ValueError("No views added. Call add_view() first.")

        total_voxels = self.resolution**3

        if verbose:
            print(f"\n{'='*70}")
            print("MULTI-VIEW VISUAL HULL RECONSTRUCTION")
            print(f"{'='*70}")
            print(f"  Views: {len(self.views)}")
            print(f"  Resolution: {self.resolution}³")
            print(f"  Bounds: {self.bounds_min} to {self.bounds_max}")

        if any(not view.silhouette.any() for view in self.views):
            if verbose:
                print("\n  Warning: one or more silhouettes are empty.")
                print("  Returning empty voxel grid.")
            self.voxel_grid = np.zeros(
                (self.resolution, self.resolution, self.resolution), dtype=bool
            )
            if verbose:
                print(f"\n  Final occupied voxels: 0 / {total_voxels:,}")
                print("  Occupancy: 0.00%")
                print(f"{'='*70}\n")
            return self.voxel_grid

        # Initialize voxel grid (all True initially, will be intersected)
        self.voxel_grid = np.ones(
            (self.resolution, self.resolution, self.resolution), dtype=bool
        )

        if verbose:
            print(f"\n  Processing {len(self.views)} views...")

        effective_chunk_size = chunk_size if chunk_size is not None else self.chunk_size

        if use_vectorized:
            self._reconstruct_vectorized(
                verbose=verbose, chunk_size=effective_chunk_size
            )
        else:
            self._reconstruct_loop(verbose=verbose)

        if verbose:
            final_occupied = int(self.voxel_grid.sum())
            print(f"\n  Final occupied voxels: {final_occupied:,} / {total_voxels:,}")
            print(f"  Occupancy: {final_occupied / total_voxels * 100:.2f}%")
            print(f"{'='*70}\n")

        return self.voxel_grid

    def _reconstruct_loop(self, verbose: bool = True) -> None:
        """Reconstruct using the original triple nested loops."""
        voxel_centers = self._compute_voxel_centers()

        for i, view in enumerate(self.views):
            if verbose:
                view_desc = (
                    f"{view.view_type} @ {view.angle}°"
                    if view.view_type == "lateral"
                    else "top view"
                )
                print(
                    f"    [{i+1}/{len(self.views)}] {view_desc}... ",
                    end="",
                    flush=True,
                )

            view_mask = np.zeros_like(self.voxel_grid, dtype=bool)

            for ix in range(self.resolution):
                for iy in range(self.resolution):
                    for iz in range(self.resolution):
                        point = voxel_centers[ix, iy, iz]
                        if view.is_point_in_silhouette_cone(
                            point, (self.bounds_min, self.bounds_max)
                        ):
                            view_mask[ix, iy, iz] = True

            self.voxel_grid &= view_mask

            occupied = int(view_mask.sum())
            if verbose:
                print(f"OK ({occupied:,} voxels inside)")

            if not self.voxel_grid.any():
                if verbose:
                    print("    No voxels remain after this view; stopping early.")
                break

    def _reconstruct_vectorized(
        self, *, verbose: bool = True, chunk_size: Optional[int] = None
    ) -> None:
        """Reconstruct using vectorized projection with optional chunking."""
        if chunk_size is None:
            chunk_size = self.resolution

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        if chunk_size > self.resolution:
            chunk_size = self.resolution

        bounds_min = self.bounds_min
        bounds_max = self.bounds_max
        center = (bounds_min + bounds_max) / 2.0

        x = np.linspace(bounds_min[0], bounds_max[0], self.resolution)
        y = np.linspace(bounds_min[1], bounds_max[1], self.resolution)
        z = np.linspace(bounds_min[2], bounds_max[2], self.resolution)

        xx = x[:, None, None]
        yy = y[None, :, None]

        for i, view in enumerate(self.views):
            if verbose:
                view_desc = (
                    f"{view.view_type} @ {view.angle}°"
                    if view.view_type == "lateral"
                    else "top view"
                )
                print(
                    f"    [{i+1}/{len(self.views)}] {view_desc}... ",
                    end="",
                    flush=True,
                )

            view_occupied = 0

            for z_start in range(0, self.resolution, chunk_size):
                z_end = min(self.resolution, z_start + chunk_size)
                chunk = self.voxel_grid[:, :, z_start:z_end]
                if not chunk.any():
                    continue

                zz = z[z_start:z_end][None, None, :]
                view_mask = self._project_view_mask(
                    view=view,
                    xx=xx,
                    yy=yy,
                    zz=zz,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    center=center,
                )
                view_occupied += int(view_mask.sum())

                chunk &= view_mask

            if verbose:
                print(f"OK ({view_occupied:,} voxels inside)")

            if not self.voxel_grid.any():
                if verbose:
                    print("    No voxels remain after this view; stopping early.")
                break

    def _project_view_mask(
        self,
        *,
        view: CameraView,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        """Project voxel centers for a single view into a boolean mask."""
        if view.view_type == "top":
            world_x = xx
            world_y = yy
            y_min = bounds_min[1]
            y_range = bounds_max[1] - bounds_min[1]
        else:
            angle_rad = math.radians(view.angle)
            cos_a = math.cos(-angle_rad)
            sin_a = math.sin(-angle_rad)

            local_x = xx - center[0]
            local_y = yy - center[1]
            rotated_x = local_x * cos_a - local_y * sin_a

            world_x = rotated_x + center[0]
            world_y = zz
            y_min = bounds_min[2]
            y_range = bounds_max[2] - bounds_min[2]

        x_min, x_max = view.projected_horizontal_bounds(bounds_min, bounds_max)
        x_range = x_max - x_min

        world_x, world_y = np.broadcast_arrays(world_x, world_y)
        x_norm = (world_x - x_min) / x_range
        y_norm = (world_y - y_min) / y_range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        x0, y0, x1, y1 = view.image_bounds
        u_span = max(1.0, (x1 - x0) - 1.0)
        v_span = max(1.0, (y1 - y0) - 1.0)
        u = np.floor(x0 + x_norm * u_span + 1e-9).astype(int)
        v = np.floor(y0 + (1.0 - y_norm) * v_span + 1e-9).astype(int)

        valid = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)

        if not valid.any():
            return np.zeros(u.shape, dtype=bool)

        mask = np.zeros(u.shape, dtype=bool)
        mask[valid] = view.silhouette[v[valid], u[valid]]
        return mask

    def _compute_voxel_centers(self) -> np.ndarray:
        """
        Compute 3D center coordinates for all voxels.

        Returns:
            Array of shape (resolution, resolution, resolution, 3) with voxel centers
        """
        # Create coordinate arrays
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], self.resolution)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], self.resolution)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], self.resolution)

        # Create meshgrid
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Stack into (resolution, resolution, resolution, 3) array
        voxel_centers = np.stack([xx, yy, zz], axis=-1)

        return voxel_centers

    def extract_mesh_points(self, surface_only: bool = True) -> np.ndarray:
        """
        Extract point cloud from voxel grid.

        Args:
            surface_only: If True, only extract surface voxels (faster)

        Returns:
            Nx3 array of 3D points
        """
        if self.voxel_grid is None:
            raise ValueError("No reconstruction yet. Call reconstruct() first.")

        if surface_only:
            # Extract surface voxels (adjacent to empty space)
            # Simple surface detection without scipy: check 6-connected neighbors
            mask = self._extract_surface_voxels(self.voxel_grid)
        else:
            mask = self.voxel_grid

        # Get voxel indices
        indices = np.argwhere(mask)

        # Convert to world coordinates
        voxel_centers = self._compute_voxel_centers()
        points = voxel_centers[indices[:, 0], indices[:, 1], indices[:, 2]]

        return points

    def _extract_surface_voxels(self, voxel_grid: np.ndarray) -> np.ndarray:
        """
        Extract surface voxels (those adjacent to empty space).

        Uses simple 6-connectivity check (no scipy required).

        Args:
            voxel_grid: 3D boolean array

        Returns:
            3D boolean array with surface voxels marked
        """
        if not voxel_grid.any():
            return np.zeros_like(voxel_grid, dtype=bool)

        try:
            from scipy import ndimage

            structure = ndimage.generate_binary_structure(3, 1)
            eroded = ndimage.binary_erosion(
                voxel_grid, structure=structure, border_value=0
            )
            return np.logical_and(voxel_grid, np.logical_not(eroded))
        except Exception:
            pass

        surface = np.zeros_like(voxel_grid, dtype=bool)
        r, c, d = voxel_grid.shape

        # Check each occupied voxel
        for i in range(r):
            for j in range(c):
                for k in range(d):
                    if not voxel_grid[i, j, k]:
                        continue

                    # Check if any 6-connected neighbor is empty (then it's surface)
                    is_surface = False

                    # Check -X neighbor
                    if i == 0 or not voxel_grid[i - 1, j, k]:
                        is_surface = True
                    # Check +X neighbor
                    elif i == r - 1 or not voxel_grid[i + 1, j, k]:
                        is_surface = True
                    # Check -Y neighbor
                    elif j == 0 or not voxel_grid[i, j - 1, k]:
                        is_surface = True
                    # Check +Y neighbor
                    elif j == c - 1 or not voxel_grid[i, j + 1, k]:
                        is_surface = True
                    # Check -Z neighbor
                    elif k == 0 or not voxel_grid[i, j, k - 1]:
                        is_surface = True
                    # Check +Z neighbor
                    elif k == d - 1 or not voxel_grid[i, j, k + 1]:
                        is_surface = True

                    if is_surface:
                        surface[i, j, k] = True

        return surface

    def get_stats(self) -> Dict[str, object]:
        """Get reconstruction statistics."""
        if self.voxel_grid is None:
            return {"reconstructed": False}

        total = self.voxel_grid.size
        occupied = self.voxel_grid.sum()

        return {
            "reconstructed": True,
            "num_views": len(self.views),
            "resolution": self.resolution,
            "total_voxels": total,
            "occupied_voxels": int(occupied),
            "occupancy": float(occupied) / total,
            "bounds_min": self.bounds_min.tolist(),
            "bounds_max": self.bounds_max.tolist(),
        }


def load_multi_view_turntable(
    silhouette_paths: List[str],
    angles: List[float],
    include_top: bool = True,
    top_path: Optional[str] = None,
) -> MultiViewVisualHull:
    """
    Convenience function to load turntable sequence.

    Args:
        silhouette_paths: List of paths to lateral view silhouettes
        angles: List of rotation angles (degrees) for each view
        include_top: Whether to include top view
        top_path: Path to top view silhouette (if include_top=True)

    Returns:
        MultiViewVisualHull instance with views loaded
    """
    from PIL import Image

    if len(silhouette_paths) != len(angles):
        raise ValueError(
            f"Number of paths ({len(silhouette_paths)}) must match angles ({len(angles)})"
        )

    hull = MultiViewVisualHull()

    # Load lateral views
    for path, angle in zip(silhouette_paths, angles):
        img = Image.open(path).convert("L")  # Grayscale
        silhouette = np.array(img) > 127  # Threshold to binary
        hull.add_view_from_silhouette(silhouette, angle=angle, view_type="lateral")

    # Load top view
    if include_top and top_path:
        img = Image.open(top_path).convert("L")
        silhouette = np.array(img) > 127
        hull.add_view_from_silhouette(silhouette, angle=0.0, view_type="top")

    return hull
