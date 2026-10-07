"""Backend-owned profile-loft reconstruction."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from geometry.dual_profile import build_elliptical_profile_from_views
from geometry.profile_models import PixelScale
from geometry.slicing import sample_elliptical_slices

from ..backend import BackendCapabilities, BaseBackend, BackendBudget
from ..types import CandidateRequest, CandidateResult
from .blender_metrics import candidate_metrics_from_blender_object


def _mask_for_view(request: CandidateRequest, view: str) -> np.ndarray | None:
    for constraint in request.target.constraints:
        if constraint.view == view:
            from ..visibility import valid_evidence
            return np.asarray(constraint.mask, dtype=bool) & valid_evidence(constraint)
    return None


class ProfileLoftBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="profile_loft",
            capabilities=BackendCapabilities(
                requires_blender=True,
                supports_pure_python=False,
                supports_multi_view=True,
                supports_top_view=True,
                supports_uncertainty=True,
                outputs_mesh=True,
                editability_score=0.55,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        return BackendBudget(
            estimated_seconds=2.0,
            notes=("profile loft avoids legacy sequential boolean unions",),
        )

    def validate_config(self, config: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if int(config.get("num_slices", 0)) < 1:
            errors.append("num_slices must be >= 1")
        if float(config.get("unit_scale", 0.0)) <= 0:
            errors.append("unit_scale must be > 0")
        if int(config.get("num_samples", 0)) < 2:
            errors.append("num_samples must be >= 2")
        if int(config.get("radial_segments", 0)) < 3:
            errors.append("radial_segments must be >= 3")
        return errors

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        if not getattr(request.context, "blender_available", False):
            return self.unavailable(request, "profile_loft requires Blender mesh APIs")

        started = time.perf_counter()
        warnings: list[str] = []
        front_mask = _mask_for_view(request, "front")
        side_mask = _mask_for_view(request, "side")
        if front_mask is None and side_mask is None:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=("profile_loft requires at least one front or side mask",),
            )
        if front_mask is None or side_mask is None:
            warnings.append("single_view_profile_loft_used_circular_fallback")

        try:
            from integration.blender_ops.profile_loft_mesh import (
                create_loft_mesh_from_slices,
            )
            from integration.blender_ops.scene_setup import add_camera, add_lighting, setup_scene
            from utils.manifest import apply_object_tags

            setup_scene(clear_existing=True)
            scale = PixelScale(unit_per_px=float(request.config.get("unit_scale", 0.01)))
            profile = build_elliptical_profile_from_views(
                front_mask,
                side_mask,
                scale,
                num_samples=int(request.config.get("num_samples", 100)),
                z0=0.0,
                height_strategy=str(request.config.get("height_strategy", "front")),
                fallback_policy=str(request.config.get("fallback_policy", "circular")),
                min_radius_u=float(request.config.get("min_radius_u", 0.0)),
                sample_policy=str(request.config.get("sample_policy", "endpoints")),
                fill_strategy=str(request.config.get("fill_strategy", "interp_linear")),
                smoothing_window=int(request.config.get("smoothing_window", 3)),
                enable_offsets=bool(request.config.get("enable_offsets", False)),
            )
            from ..visibility import valid_evidence
            partial = any(not valid_evidence(c).all() for c in request.target.constraints
                          if c.view in {"front", "side"})
            if request.target.extras.get("view_calibration") or partial:
                from ..projection_contract import calibrated_profile
                profile = calibrated_profile(request)
            slices = sample_elliptical_slices(
                profile,
                num_slices=int(request.config.get("num_slices", 10)),
                sampling=str(request.config.get("sample_policy", "endpoints")),
            )
            if request.config.get("adaptive_sections", False):
                from ..adaptive_geometry import adaptive_slices
                slices = adaptive_slices(profile, int(request.config.get("num_slices", 10)),
                    int(request.config.get("max_sections", 64)), float(request.config.get("section_tolerance", .012)))
            obj = create_loft_mesh_from_slices(
                slices,
                name="Blockout_Mesh",
                radial_segments=int(request.config.get("radial_segments", 24)),
                adaptive_radial_segments=bool(
                    request.config.get("adaptive_radial_segments", False)
                ),
                cap_mode=str(request.config.get("cap_mode", "fan")),
                min_radius_u=float(request.config.get("min_radius_u", 0.0)),
                merge_threshold_u=float(request.config.get("merge_threshold_u", 0.0)),
                recalc_normals=bool(request.config.get("recalc_normals", True)),
                shade_smooth=bool(request.config.get("shade_smooth", True)),
                surface_mode=str(request.config.get("surface_mode", "smooth")),
                surface_subdivisions=request.config.get("surface_subdivisions", 4),
                weld_degenerate_rings=bool(
                    request.config.get("weld_degenerate_rings", True)
                ),
            )
            if obj is not None:
                if request.target.extras.get("view_calibration"):
                    obj.location.x, obj.location.y = request.target.bounds.center[:2]
                if request.config.get("contour_sections", False):
                    from ..adaptive_geometry import contour_section_mesh
                    from ..native_geometry import evaluated_arrays, GeometryArrays, NativeOwnedGeometry
                    from ..projected_metrics import projected_mesh_metrics
                    mesh = contour_section_mesh(request.target, slices, request.config.get("section_resolution", 64))
                    if mesh is not None and mesh.available:
                        baseline = evaluated_arrays(obj)
                        proposal = GeometryArrays.capture(mesh.vertices, mesh.faces)
                        def score(data):
                            rows = projected_mesh_metrics(request.target, data.vertices, data.faces).values()
                            from ..visibility import observed_area_score
                            return observed_area_score(rows)
                        from ..feature_evidence import mesh_empty_features
                        baseline_features = mesh_empty_features(request.target, baseline)
                        proposal_features = mesh_empty_features(request.target, proposal)
                        required_feature_gain = not baseline_features["passed"] and proposal_features["passed"]
                        if (proposal_features["passed"] and mesh.topology.get("watertight") and
                            (required_feature_gain or score(proposal) > score(baseline)+1e-6)):
                            obj.hide_render = True
                            obj["blendslop_export_exclude"] = True
                            owner = NativeOwnedGeometry(proposal, "ContourSectionLoft")
                            output = owner.attach()
                            obj.parent = output
                            obj = output
                from ..projection_contract import observed_holes
                if any(pixels >= 4 for pixels in observed_holes(request.target).values()):
                    from ..native_geometry import evaluated_arrays
                    from ..feature_evidence import mesh_empty_features
                    if not mesh_empty_features(request.target, evaluated_arrays(obj))["passed"]:
                        raise ValueError("profile proposal fills a required known-empty feature")
                generation_context = getattr(request.context, "generation_context", None)
                if generation_context is not None:
                    apply_object_tags(obj, role="final", context=generation_context)
                add_camera()
                add_lighting()
        except Exception as exc:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=(str(exc),),
            )
        elapsed_s = time.perf_counter() - started
        if obj is None:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="failed",
                errors=("profile loft mesh generation returned no object",),
                warnings=tuple(warnings),
            )
        metrics = candidate_metrics_from_blender_object(
            obj,
            editability_score=0.55,
            elapsed_s=elapsed_s,
            extras={
                "surface_mode": str(request.config.get("surface_mode", "smooth")),
                "slice_count": len(slices),
                "adaptive_sections": bool(request.config.get("adaptive_sections", False)),
                "front_mask_available": front_mask is not None,
                "side_mask_available": side_mask is not None,
            },
        )
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="success",
            metric_result=metrics,
            payload=obj,
            warnings=tuple(warnings),
        )
