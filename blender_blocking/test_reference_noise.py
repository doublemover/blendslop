"""Candidate-independent reference-noise contract checks; no native execution."""
from copy import deepcopy
import json
import unittest
import numpy as np
from unittest.mock import patch

from evaluation.reference_noise import (reference_noise_workload, validate_reference_workload,
    reference_noise_observation, canonical_digest, oriented_surface_identity)
from reconstruction.native_geometry import GeometryArrays
from synthetic.quality_references import coverage_fixture_contract


class TestReferenceNoise(unittest.TestCase):
    def test_contract_is_deterministic_candidate_blind_and_has_no_tolerance(self):
        frozen = reference_noise_workload()
        self.assertEqual(canonical_digest(frozen), canonical_digest(reference_noise_workload()))
        self.assertEqual(set(frozen["families"]), {"sphere", "rounded_triangle_dot"})
        self.assertIsNone(frozen["family_acceptance_contract"])
        self.assertIn("forbidden", frozen["candidate_access"])
        self.assertEqual(frozen["metric"]["sample_count_per_direction"], 4096)
        self.assertNotIn("2.5", json.dumps(frozen))
        self.assertNotIn("0.003", json.dumps(frozen))
        self.assertEqual(frozen["renders"], "unrun")

    def test_reference_parameter_mutation_is_rejected_before_measurement(self):
        retained = coverage_fixture_contract()
        frozen = reference_noise_workload()
        validate_reference_workload(retained, frozen)
        retained["cases"][0]["parameters"]["radius"] = .81
        with self.assertRaisesRegex(ValueError, "authored reference parameters differ"):
            validate_reference_workload(retained, frozen)
        with self.assertRaisesRegex(ValueError, "authored coverage-reference"):
            validate_reference_workload({"protocol": "candidate"}, frozen)

    def test_workload_mutation_does_not_change_subsequent_freeze(self):
        first = reference_noise_workload()
        first["families"]["sphere"]["authored_parameters"]["radius"] = 2.
        self.assertEqual(reference_noise_workload()["families"]["sphere"]["authored_parameters"]["radius"], .8)

    def test_only_complete_declared_reference_levels_are_measured(self):
        frozen = reference_noise_workload()
        with self.assertRaisesRegex(ValueError, "all independently frozen"):
            reference_noise_observation("sphere", {"baseline": None, "candidate": None}, frozen)

    def test_native_vertex_and_face_permutation_preserves_exact_oriented_identity(self):
        source = GeometryArrays.capture([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [[0,1,2],[0,2,3]])
        order = np.array([2,0,3,1])
        inverse = np.argsort(order)
        other = GeometryArrays.capture(source.vertices[order], inverse[source.faces[::-1, [1,2,0]]])
        self.assertNotEqual(source.content_hash, other.content_hash)
        self.assertEqual(oriented_surface_identity(source), oriented_surface_identity(other))
        for vertices, faces in [(source.vertices, source.faces[:, ::-1]),
                                (source.vertices, [[0,1,3],[1,2,3]]),
                                (source.vertices + 1e-12, source.faces)]:
            altered = GeometryArrays.capture(vertices, faces)
            self.assertNotEqual(oriented_surface_identity(source), oriented_surface_identity(altered))

    def test_reference_observations_never_grant_family_acceptance(self):
        class Array:
            def __init__(self, name):
                self.content_hash = name
        levels = {name: Array(name) for name in ("baseline", "dense", "finer")}
        frozen = reference_noise_workload()
        untouched = deepcopy(frozen)
        with patch("blender_blocking.evaluation.canonical_artifacts.raw_surface_observation", return_value={"metric_status": "measured"}) as compare:
            row = reference_noise_observation("sphere", levels, frozen)
        self.assertEqual(compare.call_count, 2)
        self.assertEqual(compare.call_args_list[0].args, (levels["baseline"], levels["dense"]))
        self.assertEqual(compare.call_args_list[1].args, (levels["dense"], levels["finer"]))
        self.assertTrue(row["noise_floor_measured"])
        self.assertFalse(row["noise_qualified"])
        self.assertFalse(row["candidate_accessed"])
        self.assertIsNone(row["family_acceptance_contract"])
        self.assertEqual(frozen, untouched)


if __name__ == "__main__":
    unittest.main()
