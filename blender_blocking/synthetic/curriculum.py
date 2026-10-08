"""Progressive adversarial curricula for synthetic reconstruction fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .specs import SyntheticShapeSpec


@dataclass(frozen=True)
class CurriculumExpectation:
    """Expected failure surface and scoring context for a synthetic fixture."""

    definition: str
    level: int
    expected_ambiguity: str
    required_views: tuple[str, ...] = ("front", "side", "top")
    expected_best_backends: tuple[str, ...] = ("ensemble",)
    expected_topology: Mapping[str, object] = field(default_factory=dict)
    failure_labels: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "definition": self.definition,
            "level": self.level,
            "expected_ambiguity": self.expected_ambiguity,
            "required_views": list(self.required_views),
            "expected_best_backends": list(self.expected_best_backends),
            "expected_topology": dict(self.expected_topology),
            "failure_labels": list(self.failure_labels),
            "notes": self.notes,
        }


CURRICULUM_SUITES: Mapping[str, tuple[str, ...]] = {
    "adversarial-level-1": (
        "box",
        "sphere",
        "vase",
        "table",
        "border_touching",
        "low_contrast",
    ),
    "adversarial-level-2": (
        "chair",
        "car",
        "thin_diagonal_struts",
        "holed_silhouette",
        "salt_and_pepper",
        "partial_occlusion",
    ),
    "adversarial-level-3": (
        "torus",
        "superquadric",
        "gear",
        "pipe_elbow",
        "inconsistent_front_side",
        "missing_top_view",
        "tilted_input",
    ),
}


_EXPECTATIONS: Mapping[str, CurriculumExpectation] = {
    "box": CurriculumExpectation(
        "box",
        1,
        "low",
        expected_best_backends=("profile_loft", "silhouette_intersection", "visual_hull_voxel"),
        expected_topology={"watertight": True, "connected_components_max": 1},
        failure_labels=("baseline_contract",),
        notes="Axis-aligned primitive should expose basic bbox, topology, and render-IoU regressions.",
    ),
    "sphere": CurriculumExpectation(
        "sphere",
        1,
        "medium",
        expected_best_backends=("primitive_fit_refine", "gaussian_ellipsoid_proxy", "visual_hull_voxel"),
        expected_topology={"watertight": True, "connected_components_max": 1},
        failure_labels=("smooth_surface", "visual_hull_faceting"),
    ),
    "vase": CurriculumExpectation(
        "vase",
        1,
        "medium",
        expected_best_backends=("profile_loft", "shape_program", "ensemble"),
        expected_topology={"watertight": True, "connected_components_max": 1},
        failure_labels=("profile_band_regression",),
    ),
    "table": CurriculumExpectation(
        "table",
        1,
        "medium",
        expected_best_backends=("shape_program", "primitive_fit_refine", "ensemble"),
        expected_topology={"connected_components_expected": 1, "thin_supports": True},
        failure_labels=("thin_support_loss", "editability_regression"),
    ),
    "border_touching": CurriculumExpectation(
        "border_touching",
        1,
        "low",
        expected_best_backends=("segmentation", "profile_loft"),
        expected_topology={"mesh_optional": True},
        failure_labels=("bbox_clipping", "canonicalization_shift"),
    ),
    "low_contrast": CurriculumExpectation(
        "low_contrast",
        1,
        "low",
        expected_best_backends=("segmentation", "ensemble"),
        expected_topology={"mesh_optional": True},
        failure_labels=("threshold_sensitivity", "confidence_low"),
    ),
    "chair": CurriculumExpectation(
        "chair",
        2,
        "high",
        expected_best_backends=("shape_program", "visual_hull_voxel", "ensemble"),
        expected_topology={"thin_supports": True, "connected_components_max": 1},
        failure_labels=("leg_gap_fill", "support_dropout", "editable_structure"),
    ),
    "car": CurriculumExpectation(
        "car",
        2,
        "high",
        expected_best_backends=("shape_program", "primitive_fit_refine", "ensemble"),
        expected_topology={"wheels_or_holes_expected": True},
        failure_labels=("wheel_gap_loss", "side_profile_ambiguity"),
    ),
    "thin_diagonal_struts": CurriculumExpectation(
        "thin_diagonal_struts",
        2,
        "high",
        expected_best_backends=("visual_hull_voxel", "differentiable_refine", "ensemble"),
        expected_topology={"thin_supports": True},
        failure_labels=("thin_structure_loss", "boundary_iou_regression"),
    ),
    "holed_silhouette": CurriculumExpectation(
        "holed_silhouette",
        2,
        "high",
        expected_best_backends=("visual_hull_voxel", "shape_program"),
        expected_topology={"holes_expected": True},
        failure_labels=("hole_fill", "topology_repair_overreach"),
    ),
    "salt_and_pepper": CurriculumExpectation(
        "salt_and_pepper",
        2,
        "medium",
        expected_best_backends=("segmentation", "ensemble"),
        expected_topology={"mesh_optional": True},
        failure_labels=("noise_overfit", "outlier_bbox_expansion"),
    ),
    "partial_occlusion": CurriculumExpectation(
        "partial_occlusion",
        2,
        "high",
        expected_best_backends=("uncertainty", "ensemble"),
        expected_topology={"mesh_optional": True},
        failure_labels=("occlusion_overfit", "required_view_failure"),
    ),
    "torus": CurriculumExpectation(
        "torus",
        3,
        "very_high",
        expected_best_backends=("primitive_fit_refine", "shape_program", "ensemble"),
        expected_topology={"holes_expected": True, "genus_min": 1},
        failure_labels=("hidden_hole_ambiguity", "primitive_family_selection"),
    ),
    "superquadric": CurriculumExpectation(
        "superquadric",
        3,
        "high",
        expected_best_backends=("primitive_fit_refine", "gaussian_ellipsoid_proxy"),
        expected_topology={"watertight": True},
        failure_labels=("primitive_calibration", "normal_consistency"),
    ),
    "gear": CurriculumExpectation(
        "gear",
        3,
        "very_high",
        expected_best_backends=("visual_hull_voxel", "shape_program", "ensemble"),
        expected_topology={"repeated_detail": True, "holes_expected": True},
        failure_labels=("tooth_dropout", "hole_fill", "boundary_detail_loss"),
    ),
    "pipe_elbow": CurriculumExpectation(
        "pipe_elbow",
        3,
        "very_high",
        expected_best_backends=("visual_hull_voxel", "shape_program", "ensemble"),
        expected_topology={"holes_expected": True, "non_axis_aligned_curvature": True},
        failure_labels=("concavity_ambiguity", "view_underconstraint"),
    ),
    "inconsistent_front_side": CurriculumExpectation(
        "inconsistent_front_side",
        3,
        "very_high",
        expected_best_backends=("target_diagnostics", "ensemble"),
        expected_topology={"mesh_optional": True},
        failure_labels=("input_inconsistency", "candidate_rejection"),
        notes="Should diagnose contradictory silhouettes instead of rewarding a plausible-looking compromise.",
    ),
    "missing_top_view": CurriculumExpectation(
        "missing_top_view",
        3,
        "very_high",
        required_views=("front", "side"),
        expected_best_backends=("active_view_recommendation", "uncertainty", "ensemble"),
        expected_topology={"mesh_optional": True},
        failure_labels=("missing_required_view", "active_view_needed"),
    ),
    "tilted_input": CurriculumExpectation(
        "tilted_input",
        3,
        "very_high",
        expected_best_backends=("calibration", "target_diagnostics"),
        expected_topology={"mesh_optional": True},
        failure_labels=("camera_calibration_error", "axis_role_suspect"),
    ),
}


def curriculum_suite_names() -> tuple[str, ...]:
    """Return suite names that are part of the progressive curriculum."""

    return tuple(sorted(CURRICULUM_SUITES))


def expectation_for_definition(definition: str) -> CurriculumExpectation | None:
    """Return curriculum expectation metadata for a definition name."""

    return _EXPECTATIONS.get(str(definition))


def expectation_for_spec(spec: SyntheticShapeSpec) -> CurriculumExpectation | None:
    """Infer and return curriculum metadata for a synthetic spec."""

    return expectation_for_definition(definition_name_from_spec(spec))


def curriculum_expectation_payload_for_spec(spec: SyntheticShapeSpec) -> dict[str, object]:
    """Return JSON-ready expectation metadata for a spec, or an empty dict."""

    expectation = expectation_for_spec(spec)
    return {} if expectation is None else expectation.to_dict()


def curriculum_cases_for_suite(suite: str) -> tuple[CurriculumExpectation, ...]:
    """Return all curriculum cases for a curriculum suite."""

    names = CURRICULUM_SUITES.get(str(suite), ())
    return tuple(
        expectation
        for name in names
        if (expectation := expectation_for_definition(name)) is not None
    )


def curriculum_summary(suites: Sequence[str] | None = None) -> dict[str, object]:
    """Summarize curriculum levels and failure labels for reporting."""

    selected = tuple(suites) if suites is not None else curriculum_suite_names()
    cases = []
    labels: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for suite in selected:
        for expectation in curriculum_cases_for_suite(suite):
            cases.append(expectation.to_dict())
            level = str(expectation.level)
            level_counts[level] = level_counts.get(level, 0) + 1
            for label in expectation.failure_labels:
                labels[label] = labels.get(label, 0) + 1
    return {
        "schema_version": "synthetic_curriculum_summary_v1",
        "suites": list(selected),
        "case_count": len(cases),
        "level_counts": level_counts,
        "failure_label_counts": dict(sorted(labels.items())),
        "cases": cases,
    }


def definition_name_from_spec(spec: SyntheticShapeSpec) -> str:
    """Infer the registry definition name from a synthetic spec."""

    parameters = spec.parameters
    for key in (
        "primitive",
        "profile_kind",
        "blockout_kind",
        "mask_kind",
        "degradation",
        "material_fixture",
    ):
        value = parameters.get(key)
        if value is not None:
            if key == "material_fixture":
                return f"material_{value}"
            return str(value)
    return str(spec.shape_id)
