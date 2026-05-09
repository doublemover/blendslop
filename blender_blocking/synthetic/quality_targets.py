"""Expected quality targets for synthetic fixtures."""

from __future__ import annotations

from .curriculum import curriculum_expectation_payload_for_spec
from .materials import appearance_expectations_from_materials
from .specs import ShapeFamily, SyntheticShapeSpec


def quality_targets_for(spec: SyntheticShapeSpec) -> dict[str, object]:
    curriculum = curriculum_expectation_payload_for_spec(spec)
    if spec.family == ShapeFamily.ANALYTIC_PRIMITIVE.value:
        return _with_curriculum_targets(
            curriculum,
            _with_appearance_targets(
                spec,
                {
                    "shape_id": spec.shape_id,
                    "ground_truth_level": "analytic_exact",
                    "targets": {
                        "profile_loft": {
                            "front_area_iou_min": 0.88,
                            "side_area_iou_min": 0.84,
                        },
                        "silhouette_intersection": {
                            "front_area_iou_min": 0.94,
                            "side_area_iou_min": 0.92,
                            "non_manifold_edges_max": 0,
                        },
                        "visual_hull_voxel": {
                            "volume_iou_min": 0.82,
                            "known_limit": "view count controls exactness even with analytic ground truth",
                        },
                    },
                },
            ),
        )
    if spec.family == ShapeFamily.PROFILE_LATHE.value:
        return _with_curriculum_targets(
            curriculum,
            {
                "shape_id": spec.shape_id,
                "ground_truth_level": "profile_json",
                "targets": {
                    "profile_loft": {
                        "front_area_iou_min": 0.9,
                        "side_area_iou_min": 0.9,
                    },
                    "visual_hull_voxel": {"volume_iou_min": 0.76},
                    "primitive_fit": {
                        "known_limit": "multi-band profiles may not reduce to one radius"
                    },
                },
            },
        )
    if spec.family == ShapeFamily.FURNITURE.value:
        return _with_curriculum_targets(
            curriculum,
            {
                "shape_id": spec.shape_id,
                "ground_truth_level": "compound_blockout",
                "targets": {
                    "profile_loft": {
                        "front_area_iou_min": 0.72,
                        "side_area_iou_min": 0.68,
                        "known_limit": "topology will not preserve leg gaps",
                    },
                    "silhouette_intersection": {
                        "front_area_iou_min": 0.86,
                        "side_area_iou_min": 0.82,
                    },
                    "visual_hull_voxel": {
                        "volume_iou_min": 0.62,
                        "known_limit": "hidden support gaps may be filled",
                    },
                },
            },
        )
    if spec.family == ShapeFamily.VEHICLE_MECHANICAL.value:
        return _with_curriculum_targets(
            curriculum,
            {
                "shape_id": spec.shape_id,
                "ground_truth_level": "compound_blockout",
                "targets": {
                    "profile_loft": {
                        "front_area_iou_min": 0.65,
                        "known_limit": "holes and wheel gaps are expected losses",
                    },
                    "silhouette_intersection": {
                        "front_area_iou_min": 0.84,
                        "side_area_iou_min": 0.8,
                    },
                    "visual_hull_voxel": {
                        "volume_iou_min": 0.66,
                        "known_limit": "concavity fill is expected",
                    },
                },
            },
        )
    if spec.family == ShapeFamily.ADVERSARIAL_SILHOUETTE.value:
        return _with_curriculum_targets(
            curriculum,
            {
                "shape_id": spec.shape_id,
                "ground_truth_level": "rendered_silhouette",
                "targets": {
                    "segmentation": {
                        "mask_iou_min": 0.98,
                        "known_failure_modes": list(spec.expected_failure_modes),
                    },
                    "profile_loft": {
                        "expected_status": (
                            "ambiguous"
                            if spec.expected_failure_modes
                            else "research_candidate"
                        ),
                    },
                },
            },
        )
    if spec.family == ShapeFamily.CAPTURE_NOISE.value:
        return _with_curriculum_targets(
            curriculum,
            {
                "shape_id": spec.shape_id,
                "ground_truth_level": "rendered_silhouette",
                "targets": {
                    "segmentation": {
                        "mask_iou_min": 0.82,
                        "expected_status": "robustness_candidate",
                        "known_failure_modes": list(spec.expected_failure_modes),
                    },
                    "uncertainty": {"should_record_degradation_parameters": True},
                },
            },
        )
    return _with_curriculum_targets(
        curriculum,
        _with_appearance_targets(spec, {"shape_id": spec.shape_id, "targets": {}}),
    )


def _with_appearance_targets(
    spec: SyntheticShapeSpec,
    payload: dict[str, object],
) -> dict[str, object]:
    appearance = appearance_expectations_from_materials(spec.materials)
    if not appearance:
        return payload
    targets = payload.setdefault("targets", {})
    if not isinstance(targets, dict):
        return payload
    targets["appearance"] = appearance
    return payload


def _with_curriculum_targets(
    curriculum: dict[str, object],
    payload: dict[str, object],
) -> dict[str, object]:
    if not curriculum:
        return payload
    payload["curriculum"] = curriculum
    targets = payload.setdefault("targets", {})
    if isinstance(targets, dict):
        targets["curriculum"] = {
            "level": curriculum.get("level"),
            "expected_ambiguity": curriculum.get("expected_ambiguity"),
            "required_views": curriculum.get("required_views", []),
            "expected_best_backends": curriculum.get("expected_best_backends", []),
            "failure_labels": curriculum.get("failure_labels", []),
        }
    return payload
