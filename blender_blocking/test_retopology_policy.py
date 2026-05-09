"""Tests for editable-mesh retopology policy decisions."""

from __future__ import annotations

import unittest

import numpy as np

from metrics.retopology_policy import (
    RetopologyPolicy,
    retopology_decision_from_mesh,
    retopology_policy_from_config,
)
from metrics.topology import mesh_topology_report
from reconstruction.backends.visual_hull import _record_retopology_policy


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        np.asarray(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            dtype=np.int64,
        ),
    )


class RetopologyPolicyTests(unittest.TestCase):
    def test_closed_editable_mesh_is_preserved(self) -> None:
        vertices, faces = _tetrahedron()

        decision = retopology_decision_from_mesh(
            vertices,
            faces,
            editability_score=0.9,
            min_required_iou=0.9,
            signed_distance_loss=0.02,
            config={"retopology_max_signed_distance_loss": 0.2},
        )

        self.assertEqual(decision.action, "preserve")
        self.assertTrue(decision.accepted_for_editing)
        self.assertEqual(decision.recommended_postprocess, "none")
        self.assertEqual(decision.metrics["topology_score"], 1.0)

    def test_open_boundary_mesh_requires_retopology(self) -> None:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=float,
        )
        faces = np.asarray(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=np.int64,
        )

        decision = retopology_decision_from_mesh(
            vertices,
            faces,
            editability_score=0.8,
            min_required_iou=0.8,
        )

        self.assertEqual(decision.action, "retopology_required")
        self.assertFalse(decision.accepted_for_editing)
        self.assertIn("boundary_edges", decision.reason)
        self.assertFalse(decision.repair_plan["safe_automatic"])

    def test_low_editability_prefers_primitive_proxy(self) -> None:
        vertices, faces = _tetrahedron()

        decision = retopology_decision_from_mesh(
            vertices,
            faces,
            editability_score=0.1,
            min_required_iou=0.9,
        )

        self.assertEqual(decision.action, "primitive_proxy_required")
        self.assertEqual(decision.recommended_backend, "shape_program_or_primitive_fit")
        self.assertIn("editable_parametric_proxy", decision.config_overrides["target_style"])

    def test_bad_silhouette_fit_rejects_before_retopology(self) -> None:
        vertices, faces = _tetrahedron()

        decision = retopology_decision_from_mesh(
            vertices,
            faces,
            editability_score=0.9,
            min_required_iou=0.92,
            signed_distance_loss=0.5,
            policy=RetopologyPolicy(max_signed_distance_loss=0.2),
        )

        self.assertEqual(decision.action, "reject_or_recapture")
        self.assertEqual(decision.recommended_backend, "calibration_or_view_recapture")
        self.assertIn("signed_distance_loss", decision.reason)

    def test_safe_repair_is_recommended_for_low_risk_invalid_faces(self) -> None:
        vertices, faces = _tetrahedron()
        faces = np.vstack([faces, np.asarray([[0, 0, 1]], dtype=np.int64)])

        decision = retopology_decision_from_mesh(
            vertices,
            faces,
            editability_score=0.9,
            min_required_iou=0.9,
        )

        self.assertEqual(decision.action, "safe_repair_recommended")
        self.assertEqual(decision.recommended_postprocess, "topology_repair")
        self.assertTrue(decision.repair_plan["safe_automatic"])

    def test_policy_from_config_parses_thresholds(self) -> None:
        policy = retopology_policy_from_config(
            {
                "retopology_min_topology_score": 0.95,
                "retopology_max_boundary_edges": 4,
                "retopology_min_editability_score": 0.7,
                "retopology_max_signed_distance_loss": 0.12,
                "retopology_prefer_primitives_below_editability": 0.5,
                "retopology_require_watertight": "false",
            }
        )

        self.assertEqual(policy.min_topology_score, 0.95)
        self.assertEqual(policy.max_boundary_edges, 4)
        self.assertEqual(policy.min_editability_score, 0.7)
        self.assertEqual(policy.max_signed_distance_loss, 0.12)
        self.assertEqual(policy.prefer_primitives_below_editability, 0.5)
        self.assertFalse(policy.require_watertight)

    def test_visual_hull_records_retopology_policy_metrics(self) -> None:
        vertices, faces = _tetrahedron()
        topology = mesh_topology_report(vertices, faces).to_dict()
        metrics = {"topology": topology}
        warnings: list[str] = []

        decision = _record_retopology_policy(
            mesh_metrics=metrics,
            per_view_metrics={
                "front": {"area_iou": 0.91, "signed_distance_loss": 0.03},
                "side": {"area_iou": 0.88, "signed_distance_loss": 0.05},
            },
            editability_score=0.9,
            config={"retopology_max_signed_distance_loss": 0.2},
            warnings=warnings,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(metrics["retopology_policy"]["action"], "preserve")
        self.assertEqual(metrics["retopology_policy"]["metrics"]["min_required_iou"], 0.88)
        self.assertEqual(
            metrics["retopology_policy"]["metrics"]["signed_distance_loss"],
            0.04,
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
