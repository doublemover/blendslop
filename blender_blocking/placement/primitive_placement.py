"""
Blender script for 3D primitive placement and mesh joining.
Positions and scales primitives based on slice analysis and joins them using boolean unions.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import bpy
    import bmesh
    from mathutils import Vector

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None
    bmesh = None

    class Vector:
        """Minimal Vector fallback for non-Blender contexts."""

        def __init__(self, seq: Sequence[float]) -> None:
            self.x = float(seq[0])
            self.y = float(seq[1])
            self.z = float(seq[2])

        def __sub__(self, other: "Vector") -> "Vector":
            return Vector((self.x - other.x, self.y - other.y, self.z - other.z))

        def __add__(self, other: "Vector") -> "Vector":
            return Vector((self.x + other.x, self.y + other.y, self.z + other.z))


import json
import math
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.blender_ops.mesh_quality import MeshQualityReport, collect_mesh_quality
from utils.blender_version import resolve_boolean_solver


class SliceAnalyzer:
    """Analyzes 2D slices to determine primitive placement parameters."""

    def __init__(
        self,
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        num_slices: int = 10,
        vertical_profile: Optional[List[Tuple[float, float]]] = None,
        min_radius_ratio: Optional[float] = None,
        min_radius_abs: float = 0.01,
        z_overlap_ratio: float = 2.5,
    ) -> None:
        """
        Initialize slice analyzer.

        Args:
            bounds_min: Vector representing minimum bounds (x, y, z)
            bounds_max: Vector representing maximum bounds (x, y, z)
            num_slices: Number of slices to analyze along Z axis
            vertical_profile: Optional list of (height, radius) tuples from image analysis
                            where height is 0 (bottom) to 1 (top) and radius is 0-1 normalized
            min_radius_ratio: Minimum radius as a ratio of the maximum dimension
            min_radius_abs: Absolute minimum radius fallback (world units)
            z_overlap_ratio: Ratio controlling slice overlap along Z
        """
        if num_slices <= 0:
            raise ValueError("num_slices must be >= 1")

        self.bounds_min = Vector(bounds_min)
        self.bounds_max = Vector(bounds_max)
        self.num_slices = num_slices
        self.vertical_profile = vertical_profile
        self.z_overlap_ratio = z_overlap_ratio

        dims = self.bounds_max - self.bounds_min
        max_dim = max(dims.x, dims.y, dims.z)
        if min_radius_ratio is None:
            min_radius_ratio = (min_radius_abs / max_dim) if max_dim > 0 else 0.0
        self.min_radius_ratio = min_radius_ratio
        self.min_radius = self.min_radius_ratio * max_dim
        self.slice_thickness = (
            (dims.z / self.num_slices) if self.num_slices > 0 else 0.0
        )

    def analyze_slice(self, z_position: float) -> Dict[str, Any]:
        """
        Analyze a single horizontal slice at given z position.

        Args:
            z_position: Z coordinate of the slice

        Returns:
            dict with 'center', 'radius', and 'scale' for primitive placement
        """
        # Calculate relative position in the volume (0 to 1)
        z_range = self.bounds_max.z - self.bounds_min.z
        z_normalized = (
            (z_position - self.bounds_min.z) / z_range if z_range > 0 else 0.5
        )

        # Use vertical profile from image if available, otherwise fallback to uniform cylinder
        if self.vertical_profile:
            profile_factor = self._interpolate_profile(z_normalized)
        else:
            # Uniform cylinder fallback when no profile data available
            profile_factor = 0.8

        # Calculate center position for this slice
        center = Vector(
            (
                (self.bounds_min.x + self.bounds_max.x) / 2,
                (self.bounds_min.y + self.bounds_max.y) / 2,
                z_position,
            )
        )

        # Calculate radius based on profile
        max_radius = min(
            (self.bounds_max.x - self.bounds_min.x) / 2,
            (self.bounds_max.y - self.bounds_min.y) / 2,
        )
        # When using extracted profile, don't apply the 0.8 shrink factor
        # The profile already contains the exact measurements we want
        if self.vertical_profile:
            radius = max_radius * profile_factor
        else:
            radius = max_radius * profile_factor * 0.8

        # Scale factor for primitive
        # Use overlap for Z to ensure cylinders blend smoothly
        scale = Vector((radius, radius, self.slice_thickness * self.z_overlap_ratio))

        return {
            "center": center,
            "radius": radius,
            "scale": scale,
            "profile_factor": profile_factor,
        }

    def get_all_slice_data(self) -> List[Dict[str, Any]]:
        """
        Analyze all slices and return placement data.

        Returns:
            List of slice analysis results
        """
        slice_data = []
        if self.num_slices == 1:
            z_pos = self.bounds_min.z + (self.bounds_max.z - self.bounds_min.z) * 0.5
            slice_data.append(self.analyze_slice(z_pos))
        else:
            z_step = (self.bounds_max.z - self.bounds_min.z) / (self.num_slices - 1)
            for i in range(self.num_slices):
                z_pos = self.bounds_min.z + i * z_step
                slice_data.append(self.analyze_slice(z_pos))

        return slice_data

    def _interpolate_profile(self, z_normalized: float) -> float:
        """
        Interpolate radius from profile data at given normalized height.

        Args:
            z_normalized: Height position normalized to 0-1 range

        Returns:
            Interpolated radius factor (0-1)
        """
        if not self.vertical_profile:
            return 0.8  # Default fallback

        # Find the two profile points that bracket z_normalized
        heights = np.array([h for h, r in self.vertical_profile], dtype=float)
        radii = np.array([r for h, r in self.vertical_profile], dtype=float)

        if heights.size == 0:
            return 0.8

        return float(np.interp(z_normalized, heights, radii))


class PrimitivePlacer:
    """Places and manages 3D primitives based on slice analysis."""

    def __init__(self) -> None:
        """Initialize the placer with an empty object registry."""
        self.placed_objects: List[bpy.types.Object] = []

    def create_primitive(
        self,
        primitive_type: str = "CUBE",
        location: Sequence[float] = (0, 0, 0),
        scale: Sequence[float] = (1, 1, 1),
        **kwargs: Any,
    ) -> bpy.types.Object:
        """
        Create a primitive object.

        Args:
            primitive_type: Type of primitive ('CUBE', 'SPHERE', 'CYLINDER', 'CONE', 'SUPERFRUSTUM')
            location: World location for the primitive
            scale: Scale vector for the primitive
            **kwargs: Additional parameters for specific primitive types
                     For SUPERFRUSTUM: radius_top, radius_bottom, height

        Returns:
            Created Blender object
        """
        if primitive_type == "CUBE":
            bpy.ops.mesh.primitive_cube_add(location=location)
        elif primitive_type == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(location=location)
        elif primitive_type == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(location=location, vertices=32)
        elif primitive_type == "CONE":
            bpy.ops.mesh.primitive_cone_add(location=location)
        elif primitive_type == "SUPERFRUSTUM":
            # Import spawn_superfrustum
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from primitives.primitives import spawn_superfrustum

            # Extract SuperFrustum-specific parameters
            radius_bottom = kwargs.get("radius_bottom", 1.0)
            radius_top = kwargs.get("radius_top", 0.5)
            height = kwargs.get("height", 2.0)

            # Create SuperFrustum
            obj = spawn_superfrustum(
                radius_bottom=radius_bottom,
                radius_top=radius_top,
                height=height,
                location=location,
            )
            # Apply scale after creation
            obj.scale = scale
            self.placed_objects.append(obj)
            return obj
        else:
            raise ValueError(f"Unknown primitive type: {primitive_type}")

        obj = bpy.context.active_object
        obj.scale = scale
        self.placed_objects.append(obj)
        return obj

    def place_primitives_from_slices(
        self,
        slice_data: List[Dict[str, Any]],
        primitive_type: str = "CYLINDER",
        min_radius: float = 0.0,
        fast_ops: bool = False,
    ) -> List[bpy.types.Object]:
        """
        Place primitives based on slice analysis data.

        Args:
            slice_data: List of slice analysis results
            primitive_type: Type of primitive to place
            fast_ops: Suppress view layer updates during bulk creation (unsafe hack).
                      Use only for large primitive batches; never for modifiers/rendering.

        Returns:
            List of created objects
        """
        created_objects = []

        from integration.blender_ops.ops_fastpath import suppress_view_layer_updates

        with suppress_view_layer_updates(enabled=fast_ops):
            for i, data in enumerate(slice_data):
                if data["radius"] > min_radius:  # Only place if radius is significant
                    obj = self.create_primitive(
                        primitive_type=primitive_type,
                        location=data["center"],
                        scale=data["scale"],
                    )
                    obj.name = f"Primitive_{i:02d}"
                    created_objects.append(obj)

        return created_objects


@dataclass(frozen=True)
class JoinAttempt:
    """One measured mesh join attempt."""

    mode: str
    success: bool
    elapsed_s: float
    warnings: Tuple[str, ...]
    quality: Optional[MeshQualityReport]
    solver_requested: Optional[str] = None
    solver_resolved: Optional[str] = None
    object_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "mode": self.mode,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "warnings": list(self.warnings),
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "solver_requested": self.solver_requested,
            "solver_resolved": self.solver_resolved,
            "object_count": self.object_count,
        }


@dataclass(frozen=True)
class JoinResult:
    """Structured result for mesh joins and explicit fallback metadata."""

    object: Any
    attempts: Tuple[JoinAttempt, ...]
    selected_mode: Optional[str]
    degraded: bool
    fatal_error: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "object_name": getattr(self.object, "name", None),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_mode": self.selected_mode,
            "degraded": self.degraded,
            "fatal_error": self.fatal_error,
        }


class MeshJoiner:
    """Joins multiple mesh objects using boolean operations."""

    @staticmethod
    def join_result(
        objects: List[bpy.types.Object],
        target_name: str = "Joined_Mesh",
        mode: str = "auto",
        voxel_size: Optional[float] = None,
        solver: Optional[str] = None,
        allow_degraded_simple_join: bool = True,
    ) -> JoinResult:
        """Join mesh objects and return measured attempts plus fallback status."""
        if not objects:
            return JoinResult(
                object=None,
                attempts=(),
                selected_mode=None,
                degraded=False,
                fatal_error="No objects provided for joining",
            )

        if len(objects) == 1:
            objects[0].name = target_name
            quality = collect_mesh_quality(objects[0])
            attempt = JoinAttempt(
                mode="single",
                success=True,
                elapsed_s=0.0,
                warnings=(),
                quality=quality,
                object_count=1,
            )
            result = JoinResult(
                object=objects[0],
                attempts=(attempt,),
                selected_mode="single",
                degraded=False,
                fatal_error=None,
            )
            MeshJoiner._store_join_metadata(objects[0], result)
            return result

        attempts: List[JoinAttempt] = []
        fatal_errors: List[str] = []
        mode_order = MeshJoiner._join_mode_order(
            mode, len(objects), allow_degraded_simple_join
        )

        for attempt_mode in mode_order:
            attempt_objects = MeshJoiner._duplicate_objects_for_attempt(
                objects, f"JoinAttempt_{attempt_mode}"
            )
            start = time.perf_counter()
            warnings: List[str] = []
            joined_obj = None
            quality = None
            solver_resolved = None
            try:
                if attempt_mode == "boolean":
                    solver_resolved = resolve_boolean_solver(solver)
                    if solver_resolved != solver and solver not in {None, "auto"}:
                        warnings.append(
                            f"boolean_solver_fallback:{solver}->{solver_resolved}"
                        )
                    joined_obj = MeshJoiner.join_with_boolean_union(
                        attempt_objects,
                        target_name=target_name,
                        solver=solver_resolved,
                    )
                elif attempt_mode == "voxel":
                    joined_obj = MeshJoiner.join_with_voxel_remesh(
                        attempt_objects,
                        target_name=target_name,
                        voxel_size=voxel_size,
                    )
                elif attempt_mode == "simple":
                    if mode == "simple":
                        warnings.append(
                            "simple_join_requested_disconnected_shells_possible"
                        )
                    else:
                        warnings.append(
                            "simple_join_degraded_internal_overlaps_possible"
                        )
                    joined_obj = MeshJoiner.join_simple(
                        attempt_objects,
                        target_name=target_name,
                    )
                else:
                    raise ValueError(f"Unknown mesh join mode: {attempt_mode}")

                quality = collect_mesh_quality(joined_obj)
                if quality.vertices == 0 or quality.faces == 0:
                    warnings.append("joined_mesh_empty")
                    raise RuntimeError("Joined mesh is empty")

                elapsed_s = time.perf_counter() - start
                attempt = JoinAttempt(
                    mode=attempt_mode,
                    success=True,
                    elapsed_s=elapsed_s,
                    warnings=tuple(warnings),
                    quality=quality,
                    solver_requested=solver,
                    solver_resolved=solver_resolved,
                    object_count=len(objects),
                )
                attempts.append(attempt)
                degraded = attempt_mode == "simple" and mode != "simple"
                MeshJoiner._remove_objects(objects)
                result = JoinResult(
                    object=joined_obj,
                    attempts=tuple(attempts),
                    selected_mode=attempt_mode,
                    degraded=degraded,
                    fatal_error=None,
                )
                MeshJoiner._store_join_metadata(joined_obj, result)
                return result
            except Exception as exc:
                elapsed_s = time.perf_counter() - start
                warnings.append(str(exc))
                attempts.append(
                    JoinAttempt(
                        mode=attempt_mode,
                        success=False,
                        elapsed_s=elapsed_s,
                        warnings=tuple(warnings),
                        quality=quality,
                        solver_requested=solver,
                        solver_resolved=solver_resolved,
                        object_count=len(objects),
                    )
                )
                fatal_errors.append(f"{attempt_mode}: {exc}")
                cleanup_targets = list(attempt_objects)
                if joined_obj is not None:
                    cleanup_targets.append(joined_obj)
                MeshJoiner._remove_objects(cleanup_targets)

        return JoinResult(
            object=None,
            attempts=tuple(attempts),
            selected_mode=None,
            degraded=False,
            fatal_error="; ".join(fatal_errors) if fatal_errors else "join failed",
        )

    @staticmethod
    def join(
        objects: List[bpy.types.Object],
        target_name: str = "Joined_Mesh",
        mode: str = "auto",
        voxel_size: Optional[float] = None,
        solver: Optional[str] = None,
        allow_degraded_simple_join: bool = True,
    ) -> bpy.types.Object:
        """Join multiple objects and return the legacy joined object."""
        result = MeshJoiner.join_result(
            objects,
            target_name=target_name,
            mode=mode,
            voxel_size=voxel_size,
            solver=solver,
            allow_degraded_simple_join=allow_degraded_simple_join,
        )
        if result.object is None:
            raise RuntimeError(result.fatal_error or "Mesh join failed")
        if result.degraded:
            print("Warning: Mesh join used degraded simple join fallback.")
        return result.object

    @staticmethod
    def join_objects(
        objects: List[bpy.types.Object],
        target_name: str = "Joined_Mesh",
        mode: str = "auto",
        voxel_size: Optional[float] = None,
        solver: Optional[str] = None,
        allow_degraded_simple_join: bool = True,
    ) -> bpy.types.Object:
        """Compatibility wrapper for callers using a join_objects name."""
        return MeshJoiner.join(
            objects,
            target_name=target_name,
            mode=mode,
            voxel_size=voxel_size,
            solver=solver,
            allow_degraded_simple_join=allow_degraded_simple_join,
        )

    @staticmethod
    def _join_mode_order(
        mode: str, object_count: int, allow_degraded_simple_join: bool = True
    ) -> List[str]:
        if mode == "auto":
            first = "boolean" if object_count <= 8 else "voxel"
            second = "voxel" if first == "boolean" else "boolean"
            order = [first, second]
            if allow_degraded_simple_join:
                order.append("simple")
            return order
        if mode == "boolean":
            order = ["boolean", "voxel"]
            if allow_degraded_simple_join:
                order.append("simple")
            return order
        if mode == "voxel":
            order = ["voxel", "boolean"]
            if allow_degraded_simple_join:
                order.append("simple")
            return order
        if mode == "simple":
            return ["simple"]
        raise ValueError(f"Unknown mesh join mode: {mode}")

    @staticmethod
    def _duplicate_objects_for_attempt(
        objects: List[bpy.types.Object], prefix: str
    ) -> List[bpy.types.Object]:
        duplicates = []
        for index, obj in enumerate(objects):
            duplicate = obj.copy()
            duplicate.data = obj.data.copy()
            duplicate.name = f"{prefix}_{index:03d}"
            collection = (
                obj.users_collection[0] if obj.users_collection else bpy.context.collection
            )
            collection.objects.link(duplicate)
            duplicates.append(duplicate)
        return duplicates

    @staticmethod
    def _remove_objects(objects: List[bpy.types.Object]) -> None:
        seen = set()
        for obj in objects:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
                if bpy.data.objects.get(obj.name) is not None:
                    bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                continue

    @staticmethod
    def _store_join_metadata(obj: bpy.types.Object, result: JoinResult) -> None:
        obj["join_result_json"] = json.dumps(result.to_dict(), sort_keys=True)
        obj["join_selected_mode"] = result.selected_mode or ""
        obj["join_degraded"] = bool(result.degraded)
        if result.attempts and result.attempts[-1].quality is not None:
            obj["mesh_quality_json"] = json.dumps(
                result.attempts[-1].quality.to_dict(), sort_keys=True
            )

    @staticmethod
    def join_with_boolean_union(
        objects: List[bpy.types.Object],
        target_name: str = "Joined_Mesh",
        solver: Optional[str] = None,
    ) -> bpy.types.Object:
        """
        Join multiple objects using boolean union operations.

        Args:
            objects: List of mesh objects to join
            target_name: Name for the resulting joined mesh

        Returns:
            The final joined object
        """
        if not objects:
            raise ValueError("No objects provided for joining")

        if len(objects) == 1:
            objects[0].name = target_name
            quality = collect_mesh_quality(objects[0])
            objects[0]["mesh_quality_json"] = json.dumps(
                quality.to_dict(), sort_keys=True
            )
            objects[0]["join_degraded"] = False
            return objects[0]

        resolved_solver = resolve_boolean_solver(solver)
        work = list(objects)
        round_index = 0
        while len(work) > 1:
            next_round = []
            for pair_index in range(0, len(work), 2):
                base_obj = work[pair_index]
                if pair_index + 1 >= len(work):
                    next_round.append(base_obj)
                    continue
                obj = work[pair_index + 1]
                modifier = base_obj.modifiers.new(
                    name=f"Union_{round_index}_{pair_index // 2}",
                    type="BOOLEAN",
                )
                modifier.operation = "UNION"
                modifier.object = obj
                modifier.solver = resolved_solver

                bpy.context.view_layer.objects.active = base_obj
                bpy.ops.object.modifier_apply(modifier=modifier.name)
                bpy.data.objects.remove(obj, do_unlink=True)
                next_round.append(base_obj)
            work = next_round
            round_index += 1

        work[0].name = target_name
        quality = collect_mesh_quality(work[0])
        work[0]["mesh_quality_json"] = json.dumps(quality.to_dict(), sort_keys=True)
        work[0]["join_degraded"] = False
        return work[0]

    @staticmethod
    def join_with_voxel_remesh(
        objects: List[bpy.types.Object],
        target_name: str = "Joined_Mesh",
        voxel_size: Optional[float] = None,
    ) -> bpy.types.Object:
        """
        Join objects then apply voxel remesh for a unified mesh.

        Args:
            objects: List of mesh objects to join
            target_name: Name for the resulting joined mesh
            voxel_size: Optional voxel size (world units)

        Returns:
            Remeshed joined object
        """
        if not objects:
            raise ValueError("No objects provided for joining")

        joined = MeshJoiner.join_simple(objects, target_name=target_name)

        if voxel_size is None:
            min_coords = [float("inf"), float("inf"), float("inf")]
            max_coords = [float("-inf"), float("-inf"), float("-inf")]
            for vertex in joined.bound_box:
                world_coord = joined.matrix_world @ Vector(vertex)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_coord[i])
                    max_coords[i] = max(max_coords[i], world_coord[i])
            max_dim = max(max_coords[i] - min_coords[i] for i in range(3))
            voxel_size = max(max_dim / 64.0, 1e-4)

        modifier = joined.modifiers.new(name="VoxelRemesh", type="REMESH")
        modifier.mode = "VOXEL"
        modifier.voxel_size = voxel_size
        modifier.use_remove_disconnected = False

        bpy.context.view_layer.objects.active = joined
        bpy.ops.object.modifier_apply(modifier=modifier.name)

        quality = collect_mesh_quality(joined)
        joined["mesh_quality_json"] = json.dumps(quality.to_dict(), sort_keys=True)
        joined["join_degraded"] = False
        return joined

    @staticmethod
    def join_simple(
        objects: List[bpy.types.Object], target_name: str = "Joined_Mesh"
    ) -> bpy.types.Object:
        """
        Join objects using simple mesh joining (faster but less precise).

        Args:
            objects: List of mesh objects to join
            target_name: Name for the resulting joined mesh

        Returns:
            The joined object
        """
        if not objects:
            raise ValueError("No objects provided for joining")

        # Deselect all
        bpy.ops.object.select_all(action="DESELECT")

        # Select all objects to join
        for obj in objects:
            obj.select_set(True)

        # Set active object
        bpy.context.view_layer.objects.active = objects[0]

        # Join
        bpy.ops.object.join()

        # Rename
        joined_obj = bpy.context.active_object
        joined_obj.name = target_name
        quality = collect_mesh_quality(joined_obj)
        joined_obj["mesh_quality_json"] = json.dumps(quality.to_dict(), sort_keys=True)
        joined_obj["join_degraded"] = True

        return joined_obj


def clear_scene() -> None:
    """Remove all mesh objects from the scene."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()


def example_usage() -> bpy.types.Object:
    """Example demonstrating primitive placement and mesh joining."""

    # Clear existing meshes
    clear_scene()

    # Define volume bounds
    bounds_min = Vector((-2, -2, 0))
    bounds_max = Vector((2, 2, 6))

    # Analyze slices
    print("Analyzing slices...")
    analyzer = SliceAnalyzer(bounds_min, bounds_max, num_slices=12)
    slice_data = analyzer.get_all_slice_data()

    # Place primitives based on analysis
    print("Placing primitives...")
    placer = PrimitivePlacer()
    objects = placer.place_primitives_from_slices(slice_data, primitive_type="CYLINDER")

    print(f"Placed {len(objects)} primitives")

    # Join all primitives using boolean union
    print("Joining meshes with boolean union...")
    joiner = MeshJoiner()
    final_mesh = joiner.join(
        objects,
        target_name="Sculpt_Base",
        mode="boolean",
    )

    print(f"Created final mesh: {final_mesh.name}")
    print("Ready for sculpting!")

    return final_mesh


# Run example if executed in Blender
if __name__ == "__main__":
    result = example_usage()
