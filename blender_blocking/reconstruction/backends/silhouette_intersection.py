"""Backend-owned silhouette-intersection reconstruction."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..backend import BackendCapabilities, BaseBackend, BackendBudget
from ..types import Bounds2D, CandidateRequest, CandidateResult, ViewConstraint
from .blender_metrics import candidate_metrics_from_blender_object


def _constraint_for_view(
    request: CandidateRequest, view: str
) -> Optional[ViewConstraint]:
    for constraint in request.target.constraints:
        if constraint.view == view:
            return constraint
    return None


def _mask_for_constraint(constraint: Optional[ViewConstraint]) -> Optional[np.ndarray]:
    if constraint is None:
        return None
    return np.asarray(constraint.mask, dtype=bool)


def _normalize_bounds(bounds: Optional[Bounds2D]) -> Optional[Tuple[float, float, float, float]]:
    if bounds is None:
        return None
    return (bounds.x0, bounds.x1, bounds.y0, bounds.y1)


def _target_minmax(
    request: CandidateRequest,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if request.target.bounds is None:
        raise ValueError("silhouette_intersection requires target 3D bounds")
    return request.target.bounds.to_min_max()


class SilhouetteIntersectionBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="silhouette_intersection",
            capabilities=BackendCapabilities(
                requires_blender=True,
                supports_pure_python=False,
                supports_multi_view=True,
                supports_top_view=False,
                supports_constraints=True,
                outputs_mesh=True,
                editability_score=0.35,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        return BackendBudget(
            estimated_seconds=4.0,
            notes=("boolean intersections can be expensive or fail empty",),
        )

    def validate_config(self, config: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if float(config.get("extrude_distance", 0.0)) <= 0:
            errors.append("extrude_distance must be > 0")
        if str(config.get("contour_mode", "external")) not in {
            "external",
            "tree",
        }:
            errors.append("contour_mode must be external or tree")
        if str(config.get("boolean_solver", "auto")) not in {
            "auto",
            "FAST",
            "EXACT",
            "MANIFOLD",
        }:
            errors.append("boolean_solver must be auto, FAST, EXACT, or MANIFOLD")
        return errors

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        if not getattr(request.context, "blender_available", False):
            return self.unavailable(
                request, "silhouette_intersection requires Blender boolean APIs"
            )

        started = time.perf_counter()
        front_constraint = _constraint_for_view(request, "front")
        side_constraint = _constraint_for_view(request, "side")
        front_mask = _mask_for_constraint(front_constraint)
        side_mask = _mask_for_constraint(side_constraint)
        if front_mask is None or side_mask is None:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=("silhouette_intersection requires front and side masks",),
            )

        try:
            import bpy  # type: ignore
            import cv2  # type: ignore

            from integration.blender_ops.mesh_generator import (
                center_extrusion,
                clean_mesh_for_boolean,
                create_mesh_from_contours,
                extrude_profile,
                triangulate_object,
            )
            from integration.blender_ops.silhouette_boolean import (
                apply_boolean,
                apply_transforms,
                mesh_counts,
                split_contours,
            )
            from integration.blender_ops.scene_setup import add_camera, add_lighting, setup_scene
            from integration.shape_matching.contour_analyzer import find_contours
            from utils.blender_version import resolve_boolean_solver
            from utils.manifest import apply_object_tags

            setup_scene(clear_existing=True)
            bounds_min, bounds_max = _target_minmax(request)
            width = bounds_max[0] - bounds_min[0]
            depth = bounds_max[1] - bounds_min[1]
            height = bounds_max[2] - bounds_min[2]
            if width <= 0 or depth <= 0 or height <= 0:
                return CandidateResult(
                    candidate_id=request.candidate_id,
                    backend_name=self.name,
                    status="failed",
                    errors=("silhouette_intersection received degenerate target bounds",),
                )

            contour_mode = str(request.config.get("contour_mode", "external"))
            front_contours, front_hierarchy = find_contours(
                front_mask.astype(np.uint8) * 255,
                mode=contour_mode,
                return_hierarchy=True,
            )
            side_contours, side_hierarchy = find_contours(
                side_mask.astype(np.uint8) * 255,
                mode=contour_mode,
                return_hierarchy=True,
            )
            if not front_contours or not side_contours:
                return CandidateResult(
                    candidate_id=request.candidate_id,
                    backend_name=self.name,
                    status="failed",
                    errors=("silhouette_intersection found no usable contours",),
                )

            largest_only_cfg = request.config.get("largest_component_only", None)
            largest_only = bool(largest_only_cfg) if largest_only_cfg is not None else False
            solver_override = str(request.config.get("boolean_solver", "auto"))
            workflow = getattr(request.context, "workflow", None)
            if solver_override == "auto" and workflow is not None:
                solver_override = workflow.config.mesh_join.boolean_solver
            solver = resolve_boolean_solver(solver_override)
            extrude_distance = float(request.config.get("extrude_distance", 1.0))

            def _build_silhouette_object(
                contours: List[np.ndarray],
                hierarchy: Optional[np.ndarray],
                *,
                name_prefix: str,
                scale: Tuple[float, float, float],
                rotation: Tuple[float, float, float],
                source_size: Tuple[int, int],
                normalize_bounds: Optional[Tuple[float, float, float, float]],
            ) -> Optional[object]:
                outer, holes = split_contours(contours, hierarchy)
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
                        continue
                    triangulate_object(obj)
                    extrude_profile(obj, extrude_distance=extrude_distance)
                    center_extrusion(obj, extrude_distance=extrude_distance)
                    obj.scale = scale
                    obj.rotation_euler = rotation
                    apply_transforms(
                        obj,
                        bpy_module=bpy,
                        triangulate_object=triangulate_object,
                        clean_mesh_for_boolean=clean_mesh_for_boolean,
                    )

                    for hole_idx in holes.get(idx, []):
                        hole_obj = create_mesh_from_contours(
                            [contours[hole_idx]],
                            name=f"{name_prefix}_Hole_{hole_idx}",
                            source_size=source_size,
                            normalize_bounds=normalize_bounds,
                        )
                        if hole_obj is None:
                            continue
                        triangulate_object(hole_obj)
                        extrude_profile(hole_obj, extrude_distance=extrude_distance)
                        center_extrusion(hole_obj, extrude_distance=extrude_distance)
                        hole_obj.scale = scale
                        hole_obj.rotation_euler = rotation
                        apply_transforms(
                            hole_obj,
                            bpy_module=bpy,
                            triangulate_object=triangulate_object,
                            clean_mesh_for_boolean=clean_mesh_for_boolean,
                        )
                        apply_boolean(
                            obj,
                            hole_obj,
                            "DIFFERENCE",
                            solver,
                            bpy_module=bpy,
                        )
                    parts.append(obj)

                if not parts:
                    return None
                base = parts[0]
                for extra in parts[1:]:
                    apply_boolean(base, extra, "UNION", solver, bpy_module=bpy)
                return base

            front_obj = _build_silhouette_object(
                front_contours,
                front_hierarchy,
                name_prefix="Front_Silhouette",
                scale=(width / 2.0, height / 2.0, depth),
                rotation=(math.radians(-90.0), 0.0, 0.0),
                source_size=(front_mask.shape[1], front_mask.shape[0]),
                normalize_bounds=_normalize_bounds(
                    front_constraint.bbox if front_constraint else None
                ),
            )
            side_obj = _build_silhouette_object(
                side_contours,
                side_hierarchy,
                name_prefix="Side_Silhouette",
                scale=(depth / 2.0, height / 2.0, width),
                rotation=(math.radians(-90.0), 0.0, math.radians(90.0)),
                source_size=(side_mask.shape[1], side_mask.shape[0]),
                normalize_bounds=_normalize_bounds(
                    side_constraint.bbox if side_constraint else None
                ),
            )
            if front_obj is None or side_obj is None:
                return CandidateResult(
                    candidate_id=request.candidate_id,
                    backend_name=self.name,
                    status="failed",
                    errors=("silhouette_intersection failed to build extruded views",),
                )

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

            final_verts, final_faces = mesh_counts(base_obj)
            if final_verts == 0 or final_faces == 0:
                return CandidateResult(
                    candidate_id=request.candidate_id,
                    backend_name=self.name,
                    status="failed",
                    errors=(
                        "silhouette_intersection boolean produced an empty mesh",
                    ),
                )
            generation_context = getattr(request.context, "generation_context", None)
            if generation_context is not None:
                apply_object_tags(base_obj, role="final", context=generation_context)
            add_camera()
            add_lighting()
            obj = base_obj
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(str(exc),),
            )
        elapsed_s = time.perf_counter() - started
        metrics = candidate_metrics_from_blender_object(
            obj,
            editability_score=0.35,
            elapsed_s=elapsed_s,
            extras={
                "front_contours": len(front_contours),
                "side_contours": len(side_contours),
                "boolean_solver": solver,
                "contour_mode": contour_mode,
            },
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="success",
            metric_result=metrics,
            payload=obj,
        )
