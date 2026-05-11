"""Pure tests for differentiable rendering backend selection and behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reconstruction import differentiable as diff_render
from reconstruction.types import (
    Bounds3D,
    CandidateBudget,
    CandidateRequest,
    OrthographicCameraSpec,
    ReconstructionTarget,
    ViewConstraint,
)
from primitives.soft_silhouette import soft_mask_metrics
from reconstruction.differentiable.candidate_adapter import _boundary_sdf_improvement_summary
from reconstruction.differentiable.status import differentiable_candidate_status


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

    def make_silhouette_target(self) -> ReconstructionTarget:
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[4:12, 5:11] = 1.0
        return ReconstructionTarget(
            constraints=(
                ViewConstraint(
                    view="front",
                    mask=mask,
                    camera=OrthographicCameraSpec(
                        view_name="front",
                        axis="front",
                        resolution=(16, 16),
                    ),
                ),
            ),
            bounds=Bounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            extras={"surface_points": self.make_target().extras["surface_points"]},
        )

    def test_optional_nvdiffrast_skip_policy(self) -> None:
        request = CandidateRequest(
            candidate_id="skip-nvdiffrast",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={"backend": "nvdiffrast", "optional_dependency_policy": "skip"},
        )

        with patch.object(
            diff_render.candidate_adapter,
            "NvdiffrastBackend",
            _MissingNvdiffrastBackend,
        ):
            result = diff_render.run_refinement_candidate(request)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("nvdiffrast", result.warnings[0])
        self.assertIn("optional_dependencies", result.metric_result.extras)
        self.assertEqual(result.errors, ())
        self.assertFalse(result.succeeded)

    def test_optional_nvdiffrast_fail_policy(self) -> None:
        request = CandidateRequest(
            candidate_id="fail-nvdiffrast",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={"backend": "nvdiffrast", "optional_dependency_policy": "fail"},
        )

        with patch.object(
            diff_render.candidate_adapter,
            "NvdiffrastBackend",
            _MissingNvdiffrastBackend,
        ):
            result = diff_render.run_refinement_candidate(request)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("nvdiffrast", result.warnings[0])
        self.assertIn("nvdiffrast", result.errors[0])
        self.assertEqual(
            result.metric_result.extras["optional_dependency_policy"]["result_status"],
            "failed",
        )
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

        with patch.object(
            diff_render.candidate_adapter,
            "NvdiffrastBackend",
            _FakeNvdiffrastBackend,
        ):
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
            init_diag = result.metric_result.extras["initialization_diagnostics"]
            first_primitive = result.payload["primitives"][0]
            self.assertTrue(init_diag["bounds_proxy"]["enabled"])
            np.testing.assert_allclose(
                first_primitive.center,
                [0.05, 0.15, 0.05],
                atol=1e-8,
            )
            np.testing.assert_allclose(
                first_primitive.radii,
                [0.45, 0.25, 0.25],
                atol=1e-8,
            )

    def test_soft_renderer_per_view_losses_fill_required_candidate_metrics(self) -> None:
        metrics = diff_render._candidate_per_view_metrics(
            {
                "front": {
                    "area_iou_loss": 0.2,
                    "soft_iou_loss": 0.3,
                    "soft_l2": 0.04,
                },
                "side": {
                    "area_iou_loss": 1.0,
                    "soft_iou_loss": 0.95,
                    "soft_l2": 0.8,
                },
                "top": {
                    "area_iou_loss": 0.1,
                    "soft_iou_loss": 0.2,
                    "soft_l2": 0.01,
                    "boundary_iou": 0.91,
                    "signed_distance_loss": 0.07,
                },
            }
        )

        self.assertAlmostEqual(metrics["front"]["area_iou"], 0.8)
        self.assertAlmostEqual(metrics["front"]["boundary_iou"], 0.7)
        self.assertAlmostEqual(metrics["front"]["signed_distance_loss"], 0.04)
        self.assertTrue(metrics["front"]["passed"])
        self.assertEqual(metrics["front"]["pass"], metrics["front"]["passed"])
        self.assertAlmostEqual(metrics["side"]["area_iou"], 0.0)
        self.assertFalse(metrics["side"]["passed"])
        self.assertIn("soft silhouette", metrics["side"]["reason"])
        self.assertAlmostEqual(metrics["top"]["boundary_iou"], 0.91)
        self.assertAlmostEqual(metrics["top"]["signed_distance_loss"], 0.07)

    def test_loss_weight_aliases_cover_boundary_and_signed_distance_terms(self) -> None:
        parsed, errors, warnings = diff_render._normalize_differentiable_config(
            {
                "loss_weights": {
                    "silhouette": 2.0,
                    "boundary": 0.75,
                    "sdf": 0.5,
                },
                "area_weight": 0.4,
                "depth_l2_weight": 0.1,
            }
        )

        self.assertEqual(errors, ())
        self.assertEqual(warnings, ())
        weights = parsed["loss_weights"]
        self.assertAlmostEqual(weights.silhouette_l2, 2.0)
        self.assertAlmostEqual(weights.boundary_iou, 0.75)
        self.assertAlmostEqual(weights.signed_distance, 0.5)
        self.assertAlmostEqual(weights.area_iou, 0.4)
        self.assertAlmostEqual(weights.depth_l2, 0.1)
        self.assertEqual(
            parsed["loss_weights_dict"]["signed_distance"],
            weights.signed_distance,
        )
        self.assertTrue(parsed["optimize_boundary_sdf_first"])

    def test_evaluate_render_loss_scores_boundary_and_sdf_terms(self) -> None:
        target = np.zeros((20, 20), dtype=np.float64)
        target[5:15, 5:15] = 1.0
        shifted = np.zeros_like(target)
        shifted[5:15, 7:17] = 1.0
        batch = diff_render.RenderBatch(silhouettes={"front": shifted})
        reconstruction_target = diff_render.ReconstructionTarget(
            silhouettes={"front": target}
        )

        loss = diff_render.evaluate_render_loss(
            batch,
            reconstruction_target,
            weights=diff_render.LossWeights(
                silhouette_l2=0.0,
                soft_iou=0.0,
                area_iou=0.0,
                boundary_iou=1.0,
                signed_distance=2.0,
            ),
        )

        self.assertGreater(loss.per_view["front"]["boundary_iou_loss"], 0.0)
        self.assertGreater(loss.per_view["front"]["signed_distance_loss"], 0.0)
        self.assertAlmostEqual(
            loss.terms["boundary_iou"],
            loss.per_view["front"]["boundary_iou_loss"],
        )
        self.assertAlmostEqual(
            loss.terms["signed_distance"],
            loss.per_view["front"]["signed_distance_loss"],
        )
        self.assertAlmostEqual(
            loss.terms["worst_boundary_iou_loss"],
            loss.per_view["front"]["boundary_iou_loss"],
        )
        self.assertAlmostEqual(
            loss.terms["worst_signed_distance_loss"],
            loss.per_view["front"]["signed_distance_loss"],
        )
        self.assertAlmostEqual(
            loss.total,
            loss.terms["boundary_iou"] + 2.0 * loss.terms["signed_distance"],
        )

    def test_boundary_sdf_summary_counts_worst_view_improvement(self) -> None:
        initial = SimpleNamespace(
            terms={
                "boundary_iou": 0.25,
                "signed_distance": 0.20,
                "worst_boundary_iou_loss": 0.90,
                "worst_signed_distance_loss": 0.70,
            }
        )
        final = SimpleNamespace(
            terms={
                "boundary_iou": 0.25,
                "signed_distance": 0.20,
                "worst_boundary_iou_loss": 0.55,
                "worst_signed_distance_loss": 0.40,
            }
        )

        summary = _boundary_sdf_improvement_summary(initial, final)

        self.assertTrue(summary["boundary_or_sdf_improved"])
        self.assertAlmostEqual(summary["worst_boundary_loss_improvement"], 0.35)
        self.assertAlmostEqual(
            summary["worst_signed_distance_loss_improvement"],
            0.30,
        )

    def test_soft_mask_metrics_hardens_low_opacity_overlap_for_diagnostics(self) -> None:
        target = np.zeros((20, 20), dtype=np.float64)
        target[5:15, 5:15] = 1.0
        predicted = np.zeros_like(target)
        predicted[5:15, 5:15] = 0.25

        metrics = soft_mask_metrics(predicted, target)

        self.assertEqual(metrics["area_iou_loss"], 0.0)
        self.assertLess(metrics["area_iou_loss"], 1.0)
        self.assertLess(metrics["pred_hard_threshold"], 0.5)
        self.assertGreater(metrics["boundary_iou"], 0.0)
        self.assertGreaterEqual(metrics["pred_bbox_x0"], 0.0)

    def test_objective_regression_can_fail_strict_candidate(self) -> None:
        request = CandidateRequest(
            candidate_id="cpu-soft-regression",
            backend_name="differentiable_refine",
            target=self.make_target(),
            config={
                "backend": "cpu_soft_silhouette",
                "primitive_count": 2,
                "target_point_count": 4,
                "require_objective_improvement": True,
            },
        )
        losses = [
            diff_render.LossResult(total=1.0, terms={"area_iou": 0.0}, per_view={}),
            diff_render.LossResult(total=2.0, terms={"area_iou": 1.0}, per_view={}),
        ]

        with patch.object(
            diff_render.candidate_adapter,
            "evaluate_render_loss",
            side_effect=losses,
        ):
            result = diff_render.run_refinement_candidate(request)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.succeeded)
        self.assertIn("objective did not improve", "\n".join(result.errors))

    def test_boundary_sdf_gate_degrades_average_only_improvement(self) -> None:
        status, degraded, errors, warnings = differentiable_candidate_status(
            config={},
            optimized_primitives_present=True,
            objective_improvement=0.1,
            boundary_or_sdf_improved=False,
            failed_required_views=0,
            warnings=(),
        )

        self.assertEqual(status, "degraded")
        self.assertTrue(degraded)
        self.assertEqual(errors, ())
        self.assertIn("boundary or signed-distance", "\n".join(warnings))

    def test_boundary_sdf_gate_can_fail_strict_average_only_improvement(self) -> None:
        status, degraded, errors, warnings = differentiable_candidate_status(
            config={"require_boundary_sdf_improvement": True},
            optimized_primitives_present=True,
            objective_improvement=0.1,
            boundary_or_sdf_improved=False,
            failed_required_views=0,
            warnings=(),
        )

        self.assertEqual(status, "failed")
        self.assertFalse(degraded)
        self.assertIn("boundary or signed-distance", "\n".join(errors))
        self.assertIn("boundary or signed-distance", "\n".join(warnings))

    def test_required_view_failure_can_fail_strict_candidate(self) -> None:
        status, degraded, errors, warnings = differentiable_candidate_status(
            config={"fail_on_required_view_failure": True},
            optimized_primitives_present=True,
            objective_improvement=0.1,
            boundary_or_sdf_improved=True,
            failed_required_views=1,
            warnings=(),
        )

        self.assertEqual(status, "failed")
        self.assertFalse(degraded)
        self.assertIn("required soft-silhouette", "\n".join(errors))
        self.assertIn("required soft-silhouette", "\n".join(warnings))

    def test_cpu_optimizer_honors_request_runtime_budget(self) -> None:
        request = CandidateRequest(
            candidate_id="cpu-soft-budgeted",
            backend_name="differentiable_refine",
            target=self.make_silhouette_target(),
            config={
                "backend": "cpu_soft_silhouette",
                "primitive_count": 1,
                "target_point_count": 4,
                "optimization_steps": 8,
                "max_runtime_s": 5.0,
                "max_objective_evaluations": 128,
            },
            budget=CandidateBudget(timeout_s=0.001),
        )

        result = diff_render.run_refinement_candidate(request)
        optimization = result.metric_result.extras["optimization"]

        self.assertTrue(optimization["enabled"])
        self.assertEqual(optimization["objective_policy"], "boundary_sdf_first")
        self.assertLessEqual(optimization["config"]["max_elapsed_s"], 0.001)
        self.assertIn(
            optimization["reason"],
            {"elapsed_time_budget", "objective_evaluation_budget", "max_iterations"},
        )


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
