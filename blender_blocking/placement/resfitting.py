"""
Residual Primitive Fitting (ResFit) algorithm.

Based on arXiv:2512.09201 Section 3.2 - Residual Primitive Fitting

Simplified implementation:
1. Initialize primitives from voxel/slice analysis
2. Optimize primitive parameters using gradient descent on SDF
3. Compute residual error (uncovered regions)
4. Add new primitives in high-error regions
5. Repeat until convergence

References:
- https://arxiv.org/abs/2512.09201
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from primitives.superfrustum import SuperFrustum


class ResidualFitter:
    """
    Residual Primitive Fitting algorithm for optimal primitive placement.

    Iteratively fits primitives to 3D shapes by:
    1. Placing initial primitives
    2. Optimizing parameters via gradient descent
    3. Computing residual error
    4. Adding primitives in high-error regions
    """

    def __init__(
        self,
        max_primitives: int = 20,
        max_iterations: int = 10,
        error_threshold: float = 0.01,
        learning_rate: float = 0.01,
        optimization_steps: int = 50,
    ) -> None:
        """
        Initialize ResFit algorithm.

        Args:
            max_primitives: Maximum number of primitives to place
            max_iterations: Maximum fitting iterations
            error_threshold: Target error threshold for convergence
            learning_rate: Learning rate for gradient descent
            optimization_steps: Number of optimization steps per iteration
        """
        self.max_primitives = max_primitives
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.optimization_steps = optimization_steps

        self.primitives: List[SuperFrustum] = []
        self.errors: List[float] = []

    def initialize_from_slices(
        self, slice_data: List[Dict[str, Any]], num_initial: int = 5
    ) -> List[SuperFrustum]:
        """
        Initialize primitives from slice analysis data.

        Args:
            slice_data: List of slice analysis results from SliceAnalyzer
            num_initial: Number of initial primitives to place

        Returns:
            List of initialized SuperFrustum primitives
        """
        if num_initial <= 0:
            raise ValueError("num_initial must be >= 1")

        primitives = []

        # Group slices into segments for primitive placement
        slices_per_primitive = max(1, len(slice_data) // num_initial)

        for i in range(num_initial):
            start_idx = i * slices_per_primitive
            end_idx = min((i + 1) * slices_per_primitive, len(slice_data))

            if start_idx >= len(slice_data):
                break

            # Extract slice segment
            segment = slice_data[start_idx:end_idx]

            # Compute average properties
            avg_center = np.mean([s["center"] for s in segment], axis=0)
            avg_radius = np.mean([s["radius"] for s in segment])

            # Estimate bottom and top radii
            radius_bottom = segment[0]["radius"] if segment else avg_radius
            radius_top = segment[-1]["radius"] if len(segment) > 1 else avg_radius

            # Estimate height
            if len(segment) > 1:
                z_min = segment[0]["center"][2]
                z_max = segment[-1]["center"][2]
                height = abs(z_max - z_min)
            else:
                height = 2.0  # Default

            # Create SuperFrustum
            sf = SuperFrustum(
                position=tuple(avg_center),
                orientation=(0.0, 0.0),  # Aligned with Z axis
                radius_bottom=max(radius_bottom, 0.1),
                radius_top=max(radius_top, 0.1),
                height=max(height, 0.5),
            )

            primitives.append(sf)

        return primitives

    def initialize_from_voxels(
        self, voxel_grid: np.ndarray, num_initial: int = 5
    ) -> List[SuperFrustum]:
        """
        Initialize primitives from voxel grid (simplified MSD approach).

        Args:
            voxel_grid: 3D occupancy grid (1 = filled, 0 = empty)
            num_initial: Number of initial primitives

        Returns:
            List of initialized SuperFrustum primitives
        """
        if num_initial <= 0:
            raise ValueError("num_initial must be >= 1")

        primitives = []

        # Find occupied voxels
        occupied = np.argwhere(voxel_grid > 0.5)

        if len(occupied) == 0:
            return primitives

        # Simple k-means-like clustering to find initial placements
        # For now, use vertical slicing as a proxy
        z_min, z_max = occupied[:, 2].min(), occupied[:, 2].max()
        z_step = (z_max - z_min) / num_initial

        for i in range(num_initial):
            z_start = z_min + i * z_step
            z_end = z_min + (i + 1) * z_step

            # Get voxels in this z-range
            mask = (occupied[:, 2] >= z_start) & (occupied[:, 2] < z_end)
            segment_voxels = occupied[mask]

            if len(segment_voxels) == 0:
                continue

            # Compute center
            center = segment_voxels.mean(axis=0)

            # Estimate radius from XY extent
            xy_extent = segment_voxels[:, :2] - center[:2]
            radius = np.percentile(np.linalg.norm(xy_extent, axis=1), 75)

            # Estimate height
            height = z_end - z_start

            # Create primitive
            sf = SuperFrustum(
                position=tuple(center),
                orientation=(0.0, 0.0),
                radius_bottom=max(radius, 0.1),
                radius_top=max(radius * 0.8, 0.1),  # Slight taper
                height=max(height, 0.5),
            )

            primitives.append(sf)

        return primitives

    def optimize_primitives(
        self,
        primitives: List[SuperFrustum],
        target_points: np.ndarray,
        steps: int = 50,
        use_vectorized: bool = True,
        log_progress: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[SuperFrustum]:
        """
        Optimize primitive parameters using gradient descent.

        Args:
            primitives: List of SuperFrustum primitives to optimize
            target_points: Nx3 array of target surface points
            steps: Number of optimization steps
            use_vectorized: Use vectorized SDF/gradient evaluation when available
            log_progress: Use tqdm-friendly logging (keeps progress bar at bottom)
            progress_callback: Optional callback for per-step progress updates

        Returns:
            Optimized primitives
        """
        if steps <= 0 or not primitives:
            return primitives
        if target_points.size == 0:
            raise ValueError("target_points is empty")

        target_points = np.asarray(target_points, dtype=np.float64)

        def _log(message: str) -> None:
            if log_progress:
                from utils.progress import progress_print

                progress_print(message, enabled=True)
            else:
                print(message)

        if use_vectorized and target_points.ndim == 2 and target_points.shape[1] == 3:
            epsilon = 1e-5

            def _stack_primitives(
                primitives: List[SuperFrustum],
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                positions = np.stack([sf.position for sf in primitives]).astype(
                    np.float64
                )
                orientations = np.stack([sf.orientation for sf in primitives]).astype(
                    np.float64
                )
                radius_bottom = np.array(
                    [sf.radius_bottom for sf in primitives], dtype=np.float64
                )
                radius_top = np.array(
                    [sf.radius_top for sf in primitives], dtype=np.float64
                )
                heights = np.array([sf.height for sf in primitives], dtype=np.float64)
                return positions, orientations, radius_bottom, radius_top, heights

            def _axis_vectors(orientations: np.ndarray) -> np.ndarray:
                theta = orientations[:, 0]
                phi = orientations[:, 1]
                axis = np.stack(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ],
                    axis=1,
                )
                norms = np.linalg.norm(axis, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                return axis / norms

            def _rotation_to_z(axis: np.ndarray) -> np.ndarray:
                count = axis.shape[0]
                rot = np.tile(np.eye(3, dtype=np.float64), (count, 1, 1))
                dot = axis[:, 2]
                near = dot > 0.999999
                opposite = dot < -0.999999

                if np.any(opposite):
                    rot[opposite] = np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                        ],
                        dtype=np.float64,
                    )

                mask = ~(near | opposite)
                if np.any(mask):
                    v = np.stack(
                        [
                            axis[mask, 1],
                            -axis[mask, 0],
                            np.zeros(mask.sum(), dtype=np.float64),
                        ],
                        axis=1,
                    )
                    w = 1.0 + dot[mask]
                    norm = np.sqrt(w * w + np.sum(v * v, axis=1))
                    w = w / norm
                    v = v / norm[:, None]

                    qx, qy, qz = v[:, 0], v[:, 1], v[:, 2]
                    qw = w
                    xx = qx * qx
                    yy = qy * qy
                    zz = qz * qz
                    xy = qx * qy
                    xz = qx * qz
                    yz = qy * qz
                    wx = qw * qx
                    wy = qw * qy
                    wz = qw * qz

                    rot_m = np.empty((mask.sum(), 3, 3), dtype=np.float64)
                    rot_m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
                    rot_m[:, 0, 1] = 2.0 * (xy - wz)
                    rot_m[:, 0, 2] = 2.0 * (xz + wy)
                    rot_m[:, 1, 0] = 2.0 * (xy + wz)
                    rot_m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
                    rot_m[:, 1, 2] = 2.0 * (yz - wx)
                    rot_m[:, 2, 0] = 2.0 * (xz - wy)
                    rot_m[:, 2, 1] = 2.0 * (yz + wx)
                    rot_m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
                    rot[mask] = rot_m

                return rot

            def _sdf_batch_multi(
                points: np.ndarray,
                positions: np.ndarray,
                orientations: np.ndarray,
                radius_bottom: np.ndarray,
                radius_top: np.ndarray,
                heights: np.ndarray,
            ) -> np.ndarray:
                axis = _axis_vectors(orientations)
                rot = _rotation_to_z(axis)
                p = points[None, :, :] - positions[:, None, :]
                p_local = np.einsum("pni,pji->pnj", p, rot)

                r1 = radius_bottom[:, None]
                r2 = radius_top[:, None]
                h = heights[:, None] / 2.0

                q0 = np.linalg.norm(p_local[:, :, :2], axis=2)
                q1 = p_local[:, :, 2]
                q = np.stack([q0, q1], axis=2)

                k1 = np.stack([radius_top, heights / 2.0], axis=1)
                k2 = np.stack([radius_top - radius_bottom, heights], axis=1)

                r_edge = np.where(q1 < 0.0, r1, r2)
                ca0 = q0 - np.minimum(q0, r_edge)
                ca1 = np.abs(q1) - h
                ca = np.stack([ca0, ca1], axis=2)

                dot_k2 = np.sum(k2 * k2, axis=1)
                t = np.clip(
                    np.sum((k1[:, None, :] - q) * k2[:, None, :], axis=2)
                    / dot_k2[:, None],
                    0.0,
                    1.0,
                )
                cb = q - k1[:, None, :] + t[:, :, None] * k2[:, None, :]

                s = np.where((cb[:, :, 0] < 0.0) & (ca[:, :, 1] < 0.0), -1.0, 1.0)
                dist = np.sqrt(
                    np.minimum(
                        np.sum(ca * ca, axis=2),
                        np.sum(cb * cb, axis=2),
                    )
                )
                return s * dist

            positions, orientations, radius_bottom, radius_top, heights = (
                _stack_primitives(primitives)
            )

            for step in range(steps):
                sdf_values = _sdf_batch_multi(
                    target_points,
                    positions,
                    orientations,
                    radius_bottom,
                    radius_top,
                    heights,
                )

                total_loss = float(np.sum(np.mean(sdf_values**2, axis=1)))
                weight = 2.0 * sdf_values

                grad_pos = np.zeros((len(primitives), len(target_points), 3))
                for i in range(3):
                    pos_eps = positions.copy()
                    pos_eps[:, i] += epsilon
                    sdf_eps = _sdf_batch_multi(
                        target_points,
                        pos_eps,
                        orientations,
                        radius_bottom,
                        radius_top,
                        heights,
                    )
                    grad_pos[:, :, i] = (sdf_eps - sdf_values) / epsilon

                grad_orient = np.zeros((len(primitives), len(target_points), 2))
                for i in range(2):
                    orient_eps = orientations.copy()
                    orient_eps[:, i] += epsilon
                    sdf_eps = _sdf_batch_multi(
                        target_points,
                        positions,
                        orient_eps,
                        radius_bottom,
                        radius_top,
                        heights,
                    )
                    grad_orient[:, :, i] = (sdf_eps - sdf_values) / epsilon

                rb_eps = radius_bottom + epsilon
                sdf_rb = _sdf_batch_multi(
                    target_points,
                    positions,
                    orientations,
                    rb_eps,
                    radius_top,
                    heights,
                )
                grad_rb = (sdf_rb - sdf_values) / epsilon

                rt_eps = radius_top + epsilon
                sdf_rt = _sdf_batch_multi(
                    target_points,
                    positions,
                    orientations,
                    radius_bottom,
                    rt_eps,
                    heights,
                )
                grad_rt = (sdf_rt - sdf_values) / epsilon

                h_eps = heights + epsilon
                sdf_h = _sdf_batch_multi(
                    target_points,
                    positions,
                    orientations,
                    radius_bottom,
                    radius_top,
                    h_eps,
                )
                grad_h = (sdf_h - sdf_values) / epsilon

                grad_pos_mean = np.mean(weight[:, :, None] * grad_pos, axis=1)
                grad_orient_mean = np.mean(weight[:, :, None] * grad_orient, axis=1)
                grad_rb_mean = np.mean(weight * grad_rb, axis=1)
                grad_rt_mean = np.mean(weight * grad_rt, axis=1)
                grad_h_mean = np.mean(weight * grad_h, axis=1)

                positions -= self.learning_rate * grad_pos_mean
                orientations -= self.learning_rate * grad_orient_mean
                radius_bottom -= self.learning_rate * grad_rb_mean
                radius_top -= self.learning_rate * grad_rt_mean
                heights -= self.learning_rate * grad_h_mean

                radius_bottom = np.maximum(radius_bottom, 0.1)
                radius_top = np.maximum(radius_top, 0.01)
                heights = np.maximum(heights, 0.1)

                if step % 10 == 0 or step == steps - 1:
                    _log(
                        f"    Optimization step {step}/{steps}: loss = {total_loss:.6f}"
                    )
                if progress_callback is not None:
                    progress_callback(1)

            for idx, sf in enumerate(primitives):
                sf.position = positions[idx].copy()
                sf.orientation = orientations[idx].copy()
                sf.radius_bottom = float(radius_bottom[idx])
                sf.radius_top = float(radius_top[idx])
                sf.height = float(heights[idx])

            return primitives

        for step in range(steps):
            total_loss = 0.0

            for sf in primitives:
                sdf_values = np.array([sf.sdf(p) for p in target_points])

                loss = np.mean(sdf_values**2)
                total_loss += loss

                grad_pos = np.zeros(3)
                grad_orient = np.zeros(2)
                grad_rb = 0.0
                grad_rt = 0.0
                grad_h = 0.0

                for p in target_points:
                    grads = sf.gradient(p)
                    sdf_val = sf.sdf(p)
                    grad_pos += 2 * sdf_val * grads["position"]
                    grad_orient += 2 * sdf_val * grads["orientation"]
                    grad_rb += 2 * sdf_val * grads["radius_bottom"]
                    grad_rt += 2 * sdf_val * grads["radius_top"]
                    grad_h += 2 * sdf_val * grads["height"]

                n = len(target_points)
                grad_pos /= n
                grad_orient /= n
                grad_rb /= n
                grad_rt /= n
                grad_h /= n

                sf.position -= self.learning_rate * grad_pos
                sf.orientation -= self.learning_rate * grad_orient
                sf.radius_bottom -= self.learning_rate * grad_rb
                sf.radius_top -= self.learning_rate * grad_rt
                sf.height -= self.learning_rate * grad_h

                sf.radius_bottom = max(sf.radius_bottom, 0.1)
                sf.radius_top = max(sf.radius_top, 0.01)
                sf.height = max(sf.height, 0.1)

            if step % 10 == 0 or step == steps - 1:
                _log(f"    Optimization step {step}/{steps}: loss = {total_loss:.6f}")
            if progress_callback is not None:
                progress_callback(1)

        return primitives

    def compute_residual_error(
        self, primitives: List[SuperFrustum], target_points: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute residual error (points not well-covered by primitives).

        Args:
            primitives: List of SuperFrustum primitives
            target_points: Nx3 array of target surface points

        Returns:
            Tuple of (total_error, per_point_errors)
        """
        if target_points.size == 0:
            raise ValueError("target_points is empty")
        if not primitives:
            raise ValueError("primitives is empty")

        target_points = np.asarray(target_points, dtype=np.float64)
        if target_points.ndim != 2 or target_points.shape[1] != 3:
            raise ValueError("target_points must have shape (N, 3)")

        min_sdf = None
        for sf in primitives:
            sdf_vals = np.abs(sf.sdf_batch(target_points))
            if min_sdf is None:
                min_sdf = sdf_vals
            else:
                min_sdf = np.minimum(min_sdf, sdf_vals)

        per_point_errors = min_sdf if min_sdf is not None else np.zeros(0)
        total_error = float(np.mean(per_point_errors)) if per_point_errors.size else 0.0
        return total_error, per_point_errors

    def add_primitive_at_error_region(
        self,
        primitives: List[SuperFrustum],
        target_points: np.ndarray,
        per_point_errors: np.ndarray,
    ) -> Optional[SuperFrustum]:
        """
        Add new primitive in highest-error region.

        Args:
            primitives: Existing primitives
            target_points: Nx3 array of target points
            per_point_errors: Error value for each point

        Returns:
            New SuperFrustum primitive, or None if can't add
        """
        if len(primitives) >= self.max_primitives:
            return None
        if per_point_errors.size == 0 or target_points.size == 0:
            return None

        # Find points with high error (top 25%)
        error_threshold = np.percentile(per_point_errors, 75)
        high_error_mask = per_point_errors > error_threshold
        high_error_points = target_points[high_error_mask]

        if len(high_error_points) == 0:
            return None

        # Initialize new primitive at high-error region
        center = high_error_points.mean(axis=0)

        # Estimate radius from point spread
        distances = np.linalg.norm(high_error_points - center, axis=1)
        radius = np.percentile(distances, 50)

        # Create new primitive
        new_sf = SuperFrustum(
            position=tuple(center),
            orientation=(0.0, 0.0),
            radius_bottom=max(radius, 0.1),
            radius_top=max(radius * 0.7, 0.1),
            height=max(radius * 2.0, 0.5),
        )

        return new_sf

    def fit(
        self,
        target_points: np.ndarray,
        initial_primitives: Optional[List[SuperFrustum]] = None,
        num_initial: int = 5,
        verbose: bool = True,
    ) -> List[SuperFrustum]:
        """
        Run ResFit algorithm to fit primitives to target points.

        Args:
            target_points: Nx3 array of target surface points
            initial_primitives: Optional initial primitives (if None, auto-initialize)
            num_initial: Number of initial primitives (if auto-initializing)
            verbose: Print progress

        Returns:
            List of fitted SuperFrustum primitives
        """
        if target_points.size == 0:
            raise ValueError("target_points is empty")

        if verbose:
            print("\n" + "=" * 70)
            print("RESIDUAL PRIMITIVE FITTING (ResFit)")
            print("=" * 70)

        # Initialize primitives
        if initial_primitives is None:
            if num_initial <= 0:
                raise ValueError("num_initial must be >= 1")
            if verbose:
                print(f"\nInitializing {num_initial} primitives from target points...")

            # Simple initialization: cluster target points
            # For now, use vertical slicing
            z_coords = target_points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()

            self.primitives = []
            for i in range(num_initial):
                z_start = z_min + i * (z_max - z_min) / num_initial
                z_end = z_min + (i + 1) * (z_max - z_min) / num_initial

                # Get points in this z-range
                mask = (z_coords >= z_start) & (z_coords < z_end)
                segment_points = target_points[mask]

                if len(segment_points) == 0:
                    continue

                # Compute properties
                center = segment_points.mean(axis=0)
                xy_extent = segment_points[:, :2] - center[:2]
                radius = np.percentile(np.linalg.norm(xy_extent, axis=1), 75)

                sf = SuperFrustum(
                    position=tuple(center),
                    orientation=(0.0, 0.0),
                    radius_bottom=max(radius, 0.1),
                    radius_top=max(radius * 0.8, 0.1),
                    height=z_end - z_start,
                )
                self.primitives.append(sf)
        else:
            self.primitives = initial_primitives

        if verbose:
            print(f"  ✓ Initialized {len(self.primitives)} primitives")

        # Iterative fitting
        self.errors = []

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Optimize primitives
            if verbose:
                print(f"  Optimizing {len(self.primitives)} primitives...")
            self.primitives = self.optimize_primitives(
                self.primitives, target_points, steps=self.optimization_steps
            )

            # Compute residual error
            total_error, per_point_errors = self.compute_residual_error(
                self.primitives, target_points
            )
            self.errors.append(total_error)

            if verbose:
                print(f"  Residual error: {total_error:.6f}")

            # Check convergence
            if total_error < self.error_threshold:
                if verbose:
                    print(
                        f"\n✓ Converged! Error {total_error:.6f} < {self.error_threshold}"
                    )
                break

            # Add new primitive in high-error region
            new_primitive = self.add_primitive_at_error_region(
                self.primitives, target_points, per_point_errors
            )

            if new_primitive is not None:
                self.primitives.append(new_primitive)
                if verbose:
                    print(
                        "  + Added primitive in high-error region (total:"
                        f" {len(self.primitives)})"
                    )
            else:
                if verbose:
                    print(
                        "  No new primitive added (max reached or no high-error regions)"
                    )

        if verbose:
            print("\n" + "=" * 70)
            print("RESFITTING COMPLETE")
            print("=" * 70)
            print(f"  Final primitives: {len(self.primitives)}")
            print(f"  Final error: {self.errors[-1]:.6f}")
            print(f"  Iterations: {len(self.errors)}")

        return self.primitives

    def get_history(self) -> Dict[str, Optional[object]]:
        """
        Get fitting history.

        Returns:
            Dictionary with fitting history
        """
        return {
            "num_primitives": len(self.primitives),
            "errors": self.errors,
            "iterations": len(self.errors),
            "final_error": self.errors[-1] if self.errors else None,
        }


ResiduaFitter = ResidualFitter
