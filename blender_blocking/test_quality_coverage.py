"""Strict coverage aggregation and frozen authored feature contracts."""
import unittest
from synthetic.quality_contracts import quality_workload
from synthetic.quality_references import coverage_acceptance, coverage_fixture_contract


class QualityCoverageTests(unittest.TestCase):
    def passing_receipts(self):
        return {row["name"]: {
            "silhouette": {"status": "passed", "views": {v: {"passed": True, "thresholds": {"min_boundary_iou": .8, "max_signed_distance_loss": .1}} for v in quality_workload()["views"]}},
            "surface": {"status": "passed", "surface_passed": True, "noise_qualified": True}, "topology": {"status": "passed", "boundary_qualified": True, "screen": {"valid_solid": True}},
            "editability": {"status": "passed", "passed": True}, "features": {"status": "passed"}}
            for row in quality_workload()["cases"]}

    def test_required_surface_failure_cannot_be_hidden_by_mean(self):
        receipts = self.passing_receipts()
        self.assertEqual(coverage_acceptance(receipts)["status"], "passed")
        receipts["thin_plate"]["surface"] = {"status": "failed", "aggregate_score": .999}
        self.assertEqual(coverage_acceptance(receipts)["status"], "blocked")

    def test_missing_cases_views_boundary_and_editability_are_blockers(self):
        for mutation in (lambda r: r.pop("sphere"),
                         lambda r: r["sphere"]["silhouette"]["views"].pop("side"),
                         lambda r: r["sphere"]["topology"].pop("boundary_qualified"),
                         lambda r: r["sphere"].pop("editability")):
            receipts = self.passing_receipts()
            mutation(receipts)
            self.assertEqual(coverage_acceptance(receipts)["status"], "blocked")

    def test_required_depth_cavity_and_part_features_are_independent(self):
        for name in ("thin_plate", "concave_arch", "asymmetric_multipart_solid"):
            receipts = self.passing_receipts()
            receipts[name]["features"]["status"] = "failed"
            self.assertEqual(coverage_acceptance(receipts)["status"], "blocked")

    def test_extended_fixture_preserves_frozen_parameters_and_holdouts(self):
        original, extended = quality_workload(), coverage_fixture_contract()
        self.assertEqual(original["cases"], extended["cases"])
        self.assertEqual(original["held_out_before_tuning"], extended["held_out_before_tuning"])
        self.assertEqual(len(extended["negative_controls"]), 3)
        self.assertEqual(extended["feature_contracts"]["thin_plate"]["expected_depth_world"], .08)


    def test_pass_labels_cannot_override_failed_or_unqualified_raw_metrics(self):
        for metric, key in (("surface", "surface_passed"), ("surface", "noise_qualified"),
                            ("editability", "passed")):
            receipts = self.passing_receipts()
            receipts["sphere"][metric][key] = False
            self.assertEqual(coverage_acceptance(receipts)["status"], "blocked")
        receipts = self.passing_receipts()
        receipts["sphere"]["silhouette"]["views"]["side"]["thresholds"]["min_boundary_iou"] = None
        self.assertEqual(coverage_acceptance(receipts)["status"], "blocked")
