"""Pure-Python ground truth generation for synthetic specs."""

from __future__ import annotations

from typing import Any
from typing import Mapping

from .analytic_sdf import analytic_metadata, require_numpy, sdf_sample_summary, sdf_occupancy, sdf_samples
from .degradations import apply_degradation, generate_adversarial_mask, mask_to_uint8
from .quality_targets import quality_targets_for
from .specs import ShapeFamily, SyntheticShapeSpec


def build_pure_artifacts(
    spec: SyntheticShapeSpec,
    volume_resolution: int = 64,
) -> dict[str, Any]:
    if spec.family == ShapeFamily.ANALYTIC_PRIMITIVE.value:
        samples = sdf_samples(spec, resolution=volume_resolution)
        occupancy = sdf_occupancy(samples["sdf"])
        metadata = analytic_metadata(spec, volume_resolution)
        metadata["sample_summary"] = sdf_sample_summary(samples["points"], samples["sdf"])
        metadata["geometry_reference"] = geometry_reference_metadata(samples)
        return {
            "volumes": {f"occupancy-r{volume_resolution}": occupancy},
            "sdf_samples": samples,
            "surface_samples": surface_samples_from_sdf(samples),
            "masks": {},
            "metadata": metadata,
            "quality_targets": quality_targets_for(spec),
        }

    if spec.family == ShapeFamily.ADVERSARIAL_SILHOUETTE.value:
        resolution = _resolution_from_spec(spec)
        mask, metadata = generate_adversarial_mask(str(spec.parameters["mask_kind"]), resolution, spec.seed)
        return {
            "volumes": {},
            "sdf_samples": {},
            "masks": {"clean/front": mask_to_uint8(mask)},
            "mask_arrays": {"clean/front": mask},
            "metadata": {**_metadata_base(spec, "rendered_silhouette"), **metadata},
            "quality_targets": quality_targets_for(spec),
        }

    if spec.family == ShapeFamily.CAPTURE_NOISE.value:
        resolution = _resolution_from_spec(spec)
        mask, metadata = generate_adversarial_mask("off_center_dark", resolution, spec.seed)
        clean = mask_to_uint8(mask)
        degraded, degradation_params = apply_degradation(clean, str(spec.parameters["degradation"]), seed=spec.seed)
        masks = {"clean/front": clean}
        if spec.parameters["degradation"] != "missing_top_view":
            masks["noisy/front"] = degraded
        return {
            "volumes": {},
            "sdf_samples": {},
            "masks": masks,
            "mask_arrays": {"clean/front": mask},
            "metadata": {
                **_metadata_base(spec, "rendered_silhouette"),
                **metadata,
                "degradation_parameters": degradation_params,
            },
            "quality_targets": quality_targets_for(spec),
        }

    if spec.family == ShapeFamily.PROFILE_LATHE.value:
        return {
            "volumes": {},
            "sdf_samples": {},
            "masks": {},
            "profile_json": {
                "profile_kind": spec.parameters.get("profile_kind"),
                "height": spec.parameters.get("height"),
                "profile": spec.parameters.get("profile"),
                "segments": spec.parameters.get("segments"),
            },
            "metadata": {
                **_metadata_base(spec, "profile_json"),
                "profile": spec.parameters.get("profile"),
                "segments": spec.parameters.get("segments"),
            },
            "quality_targets": quality_targets_for(spec),
        }

    return {
        "volumes": {},
        "sdf_samples": {},
        "masks": {},
        "metadata": {
            **_metadata_base(spec, "compound_blockout"),
            "parts": spec.parameters.get("parts", []),
            "known_limits": list(spec.expected_failure_modes),
        },
        "quality_targets": quality_targets_for(spec),
    }


def _resolution_from_spec(spec: SyntheticShapeSpec) -> tuple[int, int]:
    value = spec.parameters.get("resolution", [256, 256])
    return int(value[0]), int(value[1])  # type: ignore[index]


def _metadata_base(spec: SyntheticShapeSpec, ground_truth_level: str) -> dict[str, object]:
    return {
        "shape_id": spec.shape_id,
        "shape_family": spec.family,
        "seed": spec.seed,
        "generator_version": spec.generator_version,
        "ground_truth_level": ground_truth_level,
        "intended_challenges": list(spec.intended_challenges),
        "expected_failure_modes": list(spec.expected_failure_modes),
    }


def surface_samples_from_sdf(
    samples: dict[str, Any],
    *,
    band_width: float | None = None,
    max_points: int = 20000,
) -> Any:
    """Extract deterministic near-surface point samples from an SDF grid."""

    np = require_numpy()
    points = np.asarray(samples.get("points"), dtype=np.float32)
    sdf = np.asarray(samples.get("sdf"), dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3 or sdf.shape != points.shape[:3]:
        raise ValueError("SDF samples must contain points[*,*,*,3] and matching sdf")
    if band_width is None:
        band_width = _grid_spacing(points) * 0.75
    mask = np.abs(sdf) <= float(band_width)
    surface = points[mask]
    if surface.size == 0:
        idx = np.argsort(np.abs(sdf).reshape(-1))[: max(1, min(max_points, sdf.size))]
        surface = points.reshape((-1, 3))[idx]
    if len(surface) > max_points:
        step = max(1, len(surface) // int(max_points))
        surface = surface[::step][: int(max_points)]
    return surface.astype(np.float32, copy=False)


def geometry_reference_metadata(samples: dict[str, Any]) -> dict[str, object]:
    np = require_numpy()
    points = np.asarray(samples.get("points"), dtype=np.float32)
    sdf = np.asarray(samples.get("sdf"), dtype=np.float32)
    surface = surface_samples_from_sdf(samples, max_points=4096)
    occupancy = sdf <= 0.0
    return {
        "surface_sample_count": int(len(surface)),
        "surface_sampling": "near_sdf_zero_band",
        "surface_band_width": float(_grid_spacing(points) * 0.75),
        "occupancy_shape": [int(v) for v in occupancy.shape],
        "occupancy_count": int(np.count_nonzero(occupancy)),
        "occupancy_ratio": float(np.count_nonzero(occupancy) / max(1, occupancy.size)),
        "metric_payload_keys": [
            "geometry.true.chamfer_l1",
            "geometry.true.chamfer_l2",
            "geometry.true.fscore_tau",
            "geometry.true.volumetric_iou",
        ],
    }


def geometry_payload_from_candidate(
    reference_artifacts: dict[str, Any],
    *,
    candidate_surface_points: Any | None = None,
    candidate_occupancy: Any | None = None,
    recoverable_surface_points: Any | None = None,
    recoverable_occupancy: Any | None = None,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Build EvaluationBundle-ready geometry/recoverability metrics.

    The returned payload is intended for ``CandidateMetrics.extras``.  It keeps
    true synthetic geometry and the best silhouette-recoverable envelope in
    separate namespaces so downstream gates can distinguish genuine backend
    regressions from input ambiguity.
    """

    try:
        from evaluation.geometry import surface_distance_report, volumetric_iou
    except ImportError:  # pragma: no cover - package import fallback
        from blender_blocking.evaluation.geometry import surface_distance_report, volumetric_iou

    samples = reference_artifacts.get("sdf_samples", {})
    if not samples:
        raise ValueError("reference_artifacts must include SDF samples")
    reference_surface = reference_artifacts.get("surface_samples")
    if reference_surface is None:
        reference_surface = surface_samples_from_sdf(samples)
    volumes = reference_artifacts.get("volumes", {})
    reference_occupancy = _first_volume(volumes)

    payload: dict[str, Any] = {}
    true_payload: dict[str, Any] = {"source": "synthetic_ground_truth"}
    if candidate_surface_points is not None:
        true_payload.update(
            surface_distance_report(
                reference_surface,
                candidate_surface_points,
                tolerance=tolerance,
            ).to_dict()
        )
    if candidate_occupancy is not None and reference_occupancy is not None:
        true_payload["volumetric_iou"] = volumetric_iou(
            reference_occupancy,
            candidate_occupancy,
        )
    if len(true_payload) > 1:
        payload["geometry_true"] = true_payload

    recoverable_payload: dict[str, Any] = {"source": "recoverable_envelope"}
    if recoverable_surface_points is not None and candidate_surface_points is not None:
        recoverable_payload.update(
            surface_distance_report(
                recoverable_surface_points,
                candidate_surface_points,
                tolerance=tolerance,
            ).to_dict()
        )
    if recoverable_occupancy is not None and candidate_occupancy is not None:
        recoverable_payload["volumetric_iou"] = volumetric_iou(
            recoverable_occupancy,
            candidate_occupancy,
        )
    if len(recoverable_payload) > 1:
        gap_chamfer_l1 = _positive_difference(
            true_payload.get("chamfer_l1"),
            recoverable_payload.get("chamfer_l1"),
        )
        gap_chamfer_l2 = _positive_difference(
            true_payload.get("chamfer_l2"),
            recoverable_payload.get("chamfer_l2"),
        )
        gap_volume_iou = _positive_difference(
            recoverable_payload.get("volumetric_iou"),
            true_payload.get("volumetric_iou"),
        )
        payload["geometry_recoverable"] = recoverable_payload
        payload["recoverability"] = {
            "source": "synthetic_shape_factory",
            "true_geometry": payload.get("geometry_true", {}),
            "recoverable_geometry": recoverable_payload,
            "ambiguity_gap_chamfer_l1": gap_chamfer_l1,
            "ambiguity_gap_chamfer_l2": gap_chamfer_l2,
            "ambiguity_gap_volume_iou": gap_volume_iou,
            "metadata": {
                "tolerance": tolerance,
                "reference_surface_sample_count": int(len(reference_surface)),
            },
        }
    return payload


def recoverable_envelope_from_views(
    views: Mapping[str, Any],
    *,
    config: Any = None,
    resolution: int = 32,
    max_surface_points: int = 8192,
    profile_samples: int = 64,
    bounds_minmax: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Compute a silhouette-recoverable visual-hull envelope for synthetic rows.

    This is intentionally pure-Python and deterministic.  It turns the same
    reference silhouettes used by the e2e matrix into a typed target, carves a
    bounded visual hull, and returns the occupancy plus surface samples needed
    by ``geometry_payload_from_candidate``.
    """

    if not views:
        raise ValueError("views are required to build a recoverable envelope")
    if resolution < 2:
        raise ValueError("resolution must be >= 2")

    try:
        from blender_blocking.reconstruction.target_builder import build_target_from_images
        from blender_blocking.reconstruction.point_cloud import visual_hull_grid_from_target
        from blender_blocking.volume import surface_points
    except ImportError:  # pragma: no cover - direct script execution fallback
        from reconstruction.target_builder import build_target_from_images  # type: ignore
        from reconstruction.point_cloud import visual_hull_grid_from_target  # type: ignore
        from volume import surface_points  # type: ignore

    target_build = build_target_from_images(
        views,
        config=config,
        bounds_minmax=bounds_minmax,
        profile_samples=profile_samples,
    )
    grid = visual_hull_grid_from_target(
        target_build.target,
        resolution=resolution,
        chunk_size=None,
        use_vectorized=True,
        backend="dense",
        boundary_refine=True,
    )
    max_voxels = max(1, int(resolution) ** 3)
    occupancy = grid.to_dense(max_voxels=max_voxels).astype(bool, copy=False)
    recoverable_surface = surface_points(grid, max_voxels=max_voxels)
    if len(recoverable_surface) > max_surface_points:
        np = require_numpy()
        indices = (
            np.linspace(0, len(recoverable_surface) - 1, int(max_surface_points))
            .round()
            .astype(int)
        )
        recoverable_surface = recoverable_surface[indices]
    return {
        "surface_points": recoverable_surface,
        "occupancy": occupancy,
        "metadata": {
            "source": "reference_silhouette_visual_hull",
            "resolution": int(resolution),
            "surface_sample_count": int(len(recoverable_surface)),
            "occupancy_shape": [int(v) for v in occupancy.shape],
            "occupancy_count": int(occupancy.sum()),
            "target_views": list(target_build.target.views()),
            "target_warnings": list(target_build.warnings),
            "grid_stats": grid.stats().to_dict(),
        },
    }


def _first_volume(volumes: dict[str, Any]) -> Any | None:
    if not volumes:
        return None
    first_key = sorted(volumes)[0]
    return volumes[first_key]


def _positive_difference(left: Any, right: Any) -> float | None:
    try:
        if left is None or right is None:
            return None
        return max(0.0, float(left) - float(right))
    except (TypeError, ValueError):
        return None


def _grid_spacing(points: Any) -> float:
    np = require_numpy()
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 4 or points_array.shape[0] <= 1:
        return 1.0
    return float(abs(points_array[1, 0, 0, 0] - points_array[0, 0, 0, 0]))
