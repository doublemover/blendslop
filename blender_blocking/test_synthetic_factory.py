"""Pure checks for the synthetic shape factory contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthetic.artifact_writer import validate_manifest_tree, write_artifact_set
from synthetic.ground_truth import build_pure_artifacts
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
        with tempfile.TemporaryDirectory() as tmp:
            artifact_set = write_artifact_set(spec, artifacts, Path(tmp))
            result = validate_manifest_tree(artifact_set.manifest_path)
        self.assertTrue(result["ok"])

    def test_adversarial_mask_records_effect(self) -> None:
        spec = get_definition("single_outlier_pixel").create(4)
        artifacts = build_pure_artifacts(spec)
        self.assertIn("clean/front", artifacts["masks"])
        self.assertIn("expected_effect", artifacts["metadata"])

    def test_suite_registry_contains_required_entries(self) -> None:
        self.assertIn("adversarial-silhouettes", list_suites())
        self.assertIn("blender-smoke", list_suites())


if __name__ == "__main__":
    unittest.main()
