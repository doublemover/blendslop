"""Pure-Python ground truth generation for synthetic specs."""

from __future__ import annotations

from typing import Any

from .analytic_sdf import analytic_metadata, occupancy_grid, sdf_samples
from .degradations import apply_degradation, generate_adversarial_mask, mask_to_uint8
from .quality_targets import quality_targets_for
from .specs import ShapeFamily, SyntheticShapeSpec


def build_pure_artifacts(
    spec: SyntheticShapeSpec,
    volume_resolution: int = 64,
) -> dict[str, Any]:
    if spec.family == ShapeFamily.ANALYTIC_PRIMITIVE.value:
        occupancy = occupancy_grid(spec, resolution=volume_resolution)
        samples = sdf_samples(spec, resolution=volume_resolution)
        return {
            "volumes": {f"occupancy-r{volume_resolution}": occupancy},
            "sdf_samples": samples,
            "masks": {},
            "metadata": analytic_metadata(spec, volume_resolution),
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
            "metadata": {"ground_truth_level": "rendered_silhouette", **metadata},
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
                "ground_truth_level": "rendered_silhouette",
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
                "ground_truth_level": "profile_json",
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
            "ground_truth_level": "compound_blockout",
            "parts": spec.parameters.get("parts", []),
            "known_limits": list(spec.expected_failure_modes),
        },
        "quality_targets": quality_targets_for(spec),
    }


def _resolution_from_spec(spec: SyntheticShapeSpec) -> tuple[int, int]:
    value = spec.parameters.get("resolution", [256, 256])
    return int(value[0]), int(value[1])  # type: ignore[index]
