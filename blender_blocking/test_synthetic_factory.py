"""Pure checks for the synthetic shape factory contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthetic.artifact_writer import validate_manifest, validate_manifest_tree, write_artifact_set
from synthetic.ground_truth import build_pure_artifacts, geometry_payload_from_candidate
from synthetic.registry import get_definition, list_suites, specs_for_suite
from synthetic.specs import SyntheticShapeSpec


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
        self.assertGreaterEqual(payload["geometry_true"]["fscore_tau"], 0.99)

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

    def test_suite_registry_contains_required_entries(self) -> None:
        self.assertIn("adversarial-silhouettes", list_suites())
        self.assertIn("blender-smoke", list_suites())
        self.assertIn("capture-noise", list_suites())
        self.assertIn("degradation-stress", list_suites())
        self.assertIn("deterministic-micro", list_suites())
        self.assertIn("silhouette-edge-cases", list_suites())

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


if __name__ == "__main__":
    unittest.main()
