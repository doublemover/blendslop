"""Expected quality targets for synthetic fixtures."""

from __future__ import annotations

from .specs import ShapeFamily, SyntheticShapeSpec


def quality_targets_for(spec: SyntheticShapeSpec) -> dict[str, object]:
    if spec.family == ShapeFamily.ANALYTIC_PRIMITIVE.value:
        return {
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
        }
    if spec.family == ShapeFamily.PROFILE_LATHE.value:
        return {
            "shape_id": spec.shape_id,
            "ground_truth_level": "profile_json",
            "targets": {
                "profile_loft": {"front_area_iou_min": 0.9, "side_area_iou_min": 0.9},
                "visual_hull_voxel": {"volume_iou_min": 0.76},
                "primitive_fit": {"known_limit": "multi-band profiles may not reduce to one radius"},
            },
        }
    if spec.family == ShapeFamily.FURNITURE.value:
        return {
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
        }
    if spec.family == ShapeFamily.VEHICLE_MECHANICAL.value:
        return {
            "shape_id": spec.shape_id,
            "ground_truth_level": "compound_blockout",
            "targets": {
                "profile_loft": {"front_area_iou_min": 0.65, "known_limit": "holes and wheel gaps are expected losses"},
                "silhouette_intersection": {"front_area_iou_min": 0.84, "side_area_iou_min": 0.8},
                "visual_hull_voxel": {"volume_iou_min": 0.66, "known_limit": "concavity fill is expected"},
            },
        }
    if spec.family == ShapeFamily.ADVERSARIAL_SILHOUETTE.value:
        return {
            "shape_id": spec.shape_id,
            "ground_truth_level": "rendered_silhouette",
            "targets": {
                "segmentation": {
                    "mask_iou_min": 0.98,
                    "known_failure_modes": list(spec.expected_failure_modes),
                },
                "profile_loft": {
                    "expected_status": "ambiguous" if spec.expected_failure_modes else "research_candidate",
                },
            },
        }
    if spec.family == ShapeFamily.CAPTURE_NOISE.value:
        return {
            "shape_id": spec.shape_id,
            "ground_truth_level": "rendered_silhouette",
            "targets": {
                "segmentation": {"mask_iou_min": 0.82, "expected_status": "robustness_candidate"},
                "uncertainty": {"should_record_degradation_parameters": True},
            },
        }
    return {"shape_id": spec.shape_id, "targets": {}}
