"""Pure tests for differentiable rendering backend selection and behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reconstruction import differentiable_render as diff_render
from reconstruction.types import CandidateRequest, ReconstructionTarget


class TestDifferentiableRender(unittest.TestCase):
    def make_target(self) -> ReconstructionTarget:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.1, 0.2],
                [-0.4, -0.1, 0.3],
                [0.1, 0.4, -0.2],
            ],
            dtype=np.float64,
        )
        return ReconstructionTarget(extras={"surface_points": points})

    def test_optional_nvdiffrast_skip_policy(self) -> None:
        request = CandidateRequest(
            candidate_id="skip-nvdiffrast",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={"backend": "nvdiffrast", "optional_dependency_policy": "skip"},
        )

        with patch.object(diff_render, "NvdiffrastBackend", _MissingNvdiffrastBackend):
            result = diff_render.run_refinement_candidate(request)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("nvdiffrast", result.warnings[0])
        self.assertFalse(result.succeeded)

    def test_optional_nvdiffrast_fail_policy(self) -> None:
        request = CandidateRequest(
            candidate_id="fail-nvdiffrast",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={"backend": "nvdiffrast", "optional_dependency_policy": "fail"},
        )

        with patch.object(diff_render, "NvdiffrastBackend", _MissingNvdiffrastBackend):
            result = diff_render.run_refinement_candidate(request)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("nvdiffrast", result.warnings[0])
        self.assertFalse(result.succeeded)

    def test_nvdiffrast_available_backend_routes_through_renderer(self) -> None:
        request = CandidateRequest(
            candidate_id="fake-nvdiffrast",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={
                "backend": "nvdiffrast",
                "primitive_count": 2,
                "target_point_count": 4,
                "optional_dependency_policy": "fail",
            },
        )

        with patch.object(diff_render, "NvdiffrastBackend", _FakeNvdiffrastBackend):
            result = diff_render.run_refinement_candidate(request)
        self.assertEqual(result.status, "success")
        self.assertEqual(
            result.metric_result.extras["render_metadata"]["backend"],
            "nvdiffrast",
        )
        self.assertTrue(result.succeeded)

    def test_cpu_soft_silhouette_backend_emits_artifacts_and_metrics(self) -> None:
        base_request = CandidateRequest(
            candidate_id="cpu-soft-artifacts",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={
                "backend": "cpu_soft_silhouette",
                "primitive_count": 2,
                "target_point_count": 4,
                "silhouette_l2_weight": 1.0,
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            request = CandidateRequest(
                candidate_id=base_request.candidate_id,
                backend_name=base_request.backend_name,
                target=base_request.target,
                config=base_request.config,
                artifact_root=Path(tmp),
            )
            result = diff_render.run_refinement_candidate(request)
            self.assertEqual(result.status, "success")
            self.assertTrue(result.succeeded)
            self.assertEqual(result.metric_result.area_iou_min, 1.0)
            self.assertEqual(result.metric_result.area_iou_mean, 1.0)
            self.assertGreater(result.metric_result.complexity_penalty, 0.0)
            self.assertIn("primitive_json", result.artifacts)
            self.assertIn("mesh_obj", result.artifacts)
            self.assertIn("objective_history", result.artifacts)
            self.assertIn("refinement_history", result.artifacts)
            self.assertTrue(result.primitive_path is not None)
            self.assertTrue(result.mesh_path is not None)
            self.assertTrue(result.primitive_path.exists())
            self.assertTrue(result.mesh_path.exists())
            self.assertEqual(result.metric_result.per_view, {})
            self.assertEqual(result.artifacts["primitive_json"], result.primitive_path)
            self.assertEqual(result.artifacts["mesh_obj"], result.mesh_path)


class _MissingNvdiffrastBackend:
    available = False
    unavailable_reason = "nvdiffrast: missing for test"
    dependency_report = "nvdiffrast: missing for test"


class _FakeNvdiffrastBackend:
    name = "nvdiffrast"
    available = True
    unavailable_reason = None
    dependency_report = "dependencies satisfied"

    def render(self, scene, cameras):
        _ = scene
        return diff_render.RenderBatch(
            silhouettes={
                camera.name: np.zeros(
                    (int(camera.image_size[1]), int(camera.image_size[0])),
                    dtype=np.float64,
                )
                for camera in cameras
            },
            metadata={
                "backend": "nvdiffrast",
                "fake": True,
                "camera_count": len(cameras),
            },
        )

    def loss(self, render_batch, target, weights, view_weights=None):
        return diff_render.evaluate_render_loss(
            render_batch,
            target,
            weights,
            view_weights=view_weights,
        )

    def backward(self, loss):
        _ = loss
        return diff_render.GradientBatch(gradients={}, epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
