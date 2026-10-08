"""Pure checks for the synthetic shape factory contracts."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import BlockingConfig
from evaluation.appearance import report_from_mapping
from synthetic.artifact_writer import validate_manifest, validate_manifest_tree, write_artifact_set
from synthetic.curriculum import (
    curriculum_cases_for_suite,
    curriculum_summary,
    expectation_for_spec,
)
from synthetic.ground_truth import (
    build_pure_artifacts,
    geometry_payload_from_candidate,
    recoverable_envelope_from_views,
)
from synthetic.materials import (
    MATERIAL_FIXTURE_KINDS,
    appearance_payload_from_materials,
)
from synthetic.quality_targets import quality_targets_for
from synthetic.registry import get_definition, list_suites, specs_for_suite
from synthetic.specs import SyntheticShapeSpec
from blender_blocking.e2e.matrix import _build_pure_mask_case


def _rgb_rect() -> np.ndarray:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:25, 10:23, :] = 0
    return image


class SyntheticFactoryTests(unittest.TestCase):
    def test_spec_roundtrip(self) -> None:
        spec = get_definition("box").create(12)
        restored = SyntheticShapeSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, restored)

    def test_deterministic_suite_specs(self) -> None:
        first = [spec.to_dict() for spec in specs_for_suite("smoke", seed=8)]
        second = [spec.to_dict() for spec in specs_for_suite("smoke", seed=8)]
        self.assertEqual(first, second)

    def test_analytic_occupancy_and_manifest(self) -> None:
        spec = get_definition("sphere").create(3)
        artifacts = build_pure_artifacts(spec, volume_resolution=12)
        self.assertIn("occupancy-r12", artifacts["volumes"])
        self.assertIn("surface_samples", artifacts)
        self.assertIn("geometry_reference", artifacts["metadata"])
        self.assertIn("deterministic_signature", artifacts["metadata"])
        self.assertIn("sample_summary", artifacts["metadata"])
        with tempfile.TemporaryDirectory() as tmp:
            artifact_set = write_artifact_set(spec, artifacts, Path(tmp))
            result = validate_manifest_tree(artifact_set.manifest_path)
        self.assertTrue(result["ok"])

    def test_analytic_ground_truth_builds_geometry_metric_payload(self) -> None:
        spec = get_definition("sphere").create(3)
        artifacts = build_pure_artifacts(spec, volume_resolution=10)
        occupancy = artifacts["volumes"]["occupancy-r10"]
        surface = artifacts["surface_samples"]

        payload = geometry_payload_from_candidate(
            artifacts,
            candidate_surface_points=surface,
            candidate_occupancy=occupancy,
            recoverable_surface_points=surface,
            recoverable_occupancy=occupancy,
            tolerance=0.05,
        )

        self.assertIn("geometry_true", payload)
        self.assertIn("geometry_recoverable", payload)
        self.assertIn("recoverability", payload)
        self.assertAlmostEqual(payload["geometry_true"]["volumetric_iou"], 1.0)
        self.assertAlmostEqual(
            payload["recoverability"]["ambiguity_gap_chamfer_l2"],
            0.0,
        )
        self.assertAlmostEqual(
            payload["recoverability"]["ambiguity_gap_volume_iou"],
            0.0,
        )
        self.assertGreaterEqual(payload["geometry_true"]["fscore_tau"], 0.99)

    def test_recoverable_envelope_builds_from_reference_views(self) -> None:
        envelope = recoverable_envelope_from_views(
            {
                "front": _rgb_rect(),
                "side": _rgb_rect(),
                "top": _rgb_rect(),
            },
            config=BlockingConfig(),
            resolution=8,
            max_surface_points=256,
            profile_samples=4,
        )

        self.assertIn("surface_points", envelope)
        self.assertIn("occupancy", envelope)
        self.assertGreater(len(envelope["surface_points"]), 0)
        self.assertEqual(envelope["occupancy"].shape, (8, 8, 8))
        self.assertEqual(
            envelope["metadata"]["source"],
            "reference_silhouette_visual_hull",
        )
        self.assertEqual(
            sorted(envelope["metadata"]["target_views"]),
            ["front", "side", "top"],
        )

    def test_generated_artifact_policy_records_skips(self) -> None:
        spec = get_definition("single_outlier_pixel").create(4)
        artifacts = build_pure_artifacts(spec)
        with tempfile.TemporaryDirectory() as tmp:
            artifact_set = write_artifact_set(
                spec,
                artifacts,
                Path(tmp),
                generation_policy={
                    "commit_small_fixtures_only": True,
                    "keep_heavy_artifacts": False,
                },
            )
            result = validate_manifest(artifact_set.manifest_path)
        self.assertTrue(result["ok"])
        self.assertEqual(artifact_set.mask_paths, {})
        self.assertIn("mask/clean/front", result["ignored"])

    def test_adversarial_mask_records_effect(self) -> None:
        spec = get_definition("single_outlier_pixel").create(4)
        artifacts = build_pure_artifacts(spec)
        self.assertIn("clean/front", artifacts["masks"])
        self.assertIn("expected_effect", artifacts["metadata"])
        self.assertEqual(artifacts["metadata"]["shape_id"], spec.shape_id)

    def test_pure_mask_matrix_case_expands_backend_views_under_temp_root(self) -> None:
        spec = get_definition("single_outlier_pixel").create(4)
        with tempfile.TemporaryDirectory() as tmp:
            case = _build_pure_mask_case(spec=spec, output_root=Path(tmp))

        self.assertEqual(sorted(case["views"]), ["front", "side", "top"])
        self.assertEqual(sorted(case["reference_paths"]), ["front", "side", "top"])
        for path in case["reference_paths"].values():
            self.assertTrue(Path(path).is_relative_to(Path(tmp)))
        self.assertIn("expected_effect", case["metadata"])

    def test_suite_registry_contains_required_entries(self) -> None:
        self.assertIn("adversarial-level-1", list_suites())
        self.assertIn("adversarial-level-2", list_suites())
        self.assertIn("adversarial-level-3", list_suites())
        self.assertIn("adversarial-silhouettes", list_suites())
        self.assertIn("blender-smoke", list_suites())
        self.assertIn("capture-noise", list_suites())
        self.assertIn("degradation-stress", list_suites())
        self.assertIn("deterministic-micro", list_suites())
        self.assertIn("material-appearance", list_suites())
        self.assertIn("silhouette-edge-cases", list_suites())

    def test_adversarial_curriculum_declares_progressive_expectations(self) -> None:
        level_one = curriculum_cases_for_suite("adversarial-level-1")
        level_three = curriculum_cases_for_suite("adversarial-level-3")
        summary = curriculum_summary(("adversarial-level-1", "adversarial-level-3"))

        self.assertGreater(len(level_one), 0)
        self.assertTrue(all(item.level == 1 for item in level_one))
        self.assertTrue(all(item.level == 3 for item in level_three))
        self.assertIn("missing_top_view", [item.definition for item in level_three])
        self.assertEqual(summary["level_counts"]["1"], len(level_one))
        self.assertIn("active_view_needed", summary["failure_label_counts"])

    def test_curriculum_expectations_are_exposed_in_quality_targets(self) -> None:
        spec = get_definition("missing_top_view").create(19)
        expectation = expectation_for_spec(spec)
        targets = quality_targets_for(spec)

        self.assertIsNotNone(expectation)
        self.assertEqual(targets["curriculum"]["level"], 3)  # type: ignore[index]
        curriculum_targets = targets["targets"]["curriculum"]  # type: ignore[index]
        self.assertEqual(curriculum_targets["required_views"], ["front", "side"])
        self.assertIn("active_view_recommendation", curriculum_targets["expected_best_backends"])

    def test_new_degradation_families_are_registered(self) -> None:
        mask_spec = get_definition("checkerboard_breakup").create(5)
        noise_spec = get_definition("salt_and_pepper").create(5)
        self.assertEqual(mask_spec.parameters["mask_kind"], "checkerboard_breakup")
        self.assertEqual(noise_spec.parameters["degradation"], "salt_and_pepper")
        self.assertIn("silhouette_ambiguity", mask_spec.expected_failure_modes)
        self.assertIn("outlier_bbox_expansion", noise_spec.expected_failure_modes)

    def test_deterministic_fixture_specs_have_target_entries(self) -> None:
        fixture_root = Path(__file__).parent / "synthetic" / "fixtures"
        fixture_files = {
            "adversarial_checkerboard_breakup_spec.json": "checkerboard_breakup_fixture_seed_0000",
            "capture_salt_and_pepper_spec.json": "salt_and_pepper_fixture_seed_0000",
            "analytic_sphere_spec.json": "sphere_fixture_seed_0000",
        }
        expected_targets_text = (fixture_root / "expected_targets.json").read_text(encoding="utf-8")
        for filename, shape_id in fixture_files.items():
            self.assertTrue((fixture_root / filename).exists())
            self.assertIn(shape_id, expected_targets_text)

    def test_material_appearance_suite_is_deterministic_and_metric_shaped(self) -> None:
        specs = specs_for_suite("material-appearance", seed=6)
        repeat = specs_for_suite("material-appearance", seed=6)

        self.assertEqual(len(specs), len(MATERIAL_FIXTURE_KINDS))
        self.assertEqual([spec.to_dict() for spec in specs], [spec.to_dict() for spec in repeat])
        for spec in specs:
            self.assertIn("material_fixture", spec.parameters)
            self.assertIn("appearance_validation", spec.parameters)
            report = report_from_mapping(appearance_payload_from_materials(spec.materials))
            self.assertTrue(report.required)
            self.assertIsNotNone(report.material_slot_count)

    def test_material_texture_only_fixture_declares_hallucination_risk(self) -> None:
        spec = get_definition("material_texture_only_high_frequency").create(12)
        report = report_from_mapping(appearance_payload_from_materials(spec.materials))
        targets = quality_targets_for(spec)["targets"]["appearance"]  # type: ignore[index]
        artifacts = build_pure_artifacts(spec, volume_resolution=8)

        self.assertTrue(report.image_space_hallucination_warning)
        self.assertIn("texture_only_hallucination", spec.expected_failure_modes)
        self.assertTrue(targets["image_space_hallucination_expected"])  # type: ignore[index]
        self.assertIn("appearance_reference", artifacts["metadata"])
        self.assertEqual(artifacts["metadata"]["material_fixture_kind"], "texture_only_high_frequency")

    def test_material_uv_distortion_fixture_is_strictly_invalid(self) -> None:
        spec = get_definition("material_uv_checker_distortion").create(14)
        report = report_from_mapping(appearance_payload_from_materials(spec.materials))
        targets = quality_targets_for(spec)["targets"]["appearance"]  # type: ignore[index]

        self.assertFalse(report.uv_valid)
        self.assertIn("uv_invalid", report.errors)
        self.assertTrue(targets["strict_uv"])  # type: ignore[index]
        self.assertFalse(targets["uv_valid_expected"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
