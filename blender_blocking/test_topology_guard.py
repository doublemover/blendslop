"""Tests for guarded mesh topology-change acceptance."""

from __future__ import annotations

import unittest

import numpy as np

from metrics.topology import mesh_topology_report
from metrics.topology_guard import (
    MeshChangeGuardPolicy,
    evaluate_mesh_change,
    guard_policy_from_config,
)
from reconstruction.backends.visual_hull import _postprocess_mesh
from volume import MeshExtractionResult


class TopologyGuardTests(unittest.TestCase):
    def test_guard_rejects_boundary_regression(self) -> None:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        before = mesh_topology_report(
            vertices,
            ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
        )
        after = mesh_topology_report(vertices, ((0, 1, 2),))

        decision = evaluate_mesh_change(
            before=before,
            after=after,
            before_vertex_count=4,
            after_vertex_count=4,
            before_face_count=4,
            after_face_count=1,
            policy=MeshChangeGuardPolicy(),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("boundary_edges", decision.reason)

    def test_guard_policy_from_config_parses_thresholds(self) -> None:
        policy = guard_policy_from_config(
            {
                "guard_min_topology_delta": 0.1,
                "guard_max_face_count_ratio": 1.5,
                "guard_require_improvement": True,
            }
        )

        self.assertEqual(policy.min_topology_delta, 0.1)
        self.assertEqual(policy.max_face_count_ratio, 1.5)
        self.assertTrue(policy.require_improvement)

    def test_visual_hull_smooth_respects_guard_rejection(self) -> None:
        mesh = MeshExtractionResult(
            status="ok",
            method="fixture",
            requested_method="fixture",
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 1.0],
                ],
                dtype=float,
            ),
            faces=np.array(
                [
                    [0, 1, 4],
                    [1, 2, 4],
                    [2, 3, 4],
                    [3, 0, 4],
                ],
                dtype=np.int64,
            ),
        )

        returned, status = _postprocess_mesh(
            mesh,
            "smooth_guarded",
            config={
                "guard_require_improvement": True,
                "smooth_iterations": 1,
            },
        )

        self.assertIs(returned, mesh)
        self.assertEqual(status["status"], "skipped")
        self.assertFalse(status["guard"]["accepted"])
        self.assertIn("improvement required", status["message"])


if __name__ == "__main__":
    unittest.main()
