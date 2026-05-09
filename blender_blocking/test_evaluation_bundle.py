"""Pure tests for SOTA-style evaluation bundle metric groups."""

from __future__ import annotations

import unittest

from evaluation.autopsy import autopsy_pack_from_bundle
from evaluation import bundle_from_candidate
from evaluation.appearance import (
    report_from_mapping as appearance_report_from_mapping,
    reports_from_payload as appearance_reports_from_payload,
)
from evaluation.export_qa import ExportQAReport, reports_from_payload
from evaluation.reporting import compact_console_summary, markdown_report
from evaluation.schemas import EvaluationBundle
from reconstruction.types import CandidateMetrics, CandidateResult


class EvaluationBundleTests(unittest.TestCase):
    def test_evaluation_bundle_round_trips_through_json_payload(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-roundtrip",
            backend_name="visual_hull_voxel",
            status="degraded",
            degraded=True,
            metric_result=CandidateMetrics(
                area_iou_min=0.72,
                area_iou_mean=0.8,
                boundary_iou_mean=0.66,
                topology_score=0.88,
                editability_score=0.54,
                elapsed_s=1.25,
                per_view={
                    "front": {
                        "area_iou": 0.72,
                        "boundary_iou": 0.66,
                        "signed_distance_loss": 0.08,
                        "required": True,
                        "passed": True,
                    }
                },
                extras={
                    "topology": {"watertight": True, "connected_components": 1},
                    "recoverability": {
                        "true_geometry": {"chamfer_l2": 0.08},
                        "recoverable_geometry": {"chamfer_l2": 0.04},
                    },
                },
            ),
            warnings=("dependency degraded",),
        )

        bundle = bundle_from_candidate(
            result=result,
            repo="abc123",
            run_id="run-roundtrip",
            suite="synthetic-smoke",
        )
        restored = EvaluationBundle.from_dict(bundle.to_dict())

        self.assertEqual(restored.to_dict(), bundle.to_dict())
        self.assertEqual(restored.metric_index()["silhouette.min_view_iou"].value, 0.72)
        self.assertEqual(restored.degradation_state["degraded"], True)
        self.assertEqual(restored.warnings, ("dependency degraded",))

    def test_export_qa_payload_becomes_first_class_metric_group(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-a",
            backend_name="shape_program",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.9,
                area_iou_mean=0.92,
                boundary_iou_mean=0.8,
                editability_score=0.85,
                extras={
                    "export_qa": {
                        "target": "blend",
                        "status": "pass",
                        "reimport_status": "pass",
                        "object_count": 3,
                        "vertex_count": 120,
                        "face_count": 64,
                        "material_count": 2,
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "pass")
        self.assertIn("export.qa_score", metrics)
        self.assertTrue(metrics["export.status_ok"].value)
        self.assertTrue(metrics["export.reimport_ok"].value)
        self.assertIn("export.object_count", metrics)
        self.assertEqual(metrics["export.target_count"].value, 1)
        self.assertGreater(float(metrics["export.qa_score"].value), 0.9)
        self.assertIn("export=", compact_console_summary(bundle))
        self.assertIn("| Export |", markdown_report([bundle]))

    def test_export_qa_low_score_classifies_delivery_failure(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-b",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                extras={
                    "asset_export": {
                        "target": "obj",
                        "status": "fail",
                        "reimport_status": "fail",
                        "errors": ["roundtrip_missing_mesh"],
                    }
                }
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(bundle.status, "fail")
        self.assertIn("export_qa_low", codes)

    def test_export_qa_payload_accepts_multiple_target_shapes(self) -> None:
        reports = reports_from_payload(
            {
                "targets": {
                    "blend": {"status": "pass", "reimport_status": "pass"},
                    "gltf": {"status": "pass", "reimport_status": "warn"},
                }
            }
        )

        self.assertEqual(len(reports), 2)
        self.assertIsInstance(reports[0], ExportQAReport)
        self.assertEqual({report.target for report in reports}, {"blend", "gltf"})

    def test_appearance_payload_becomes_first_class_metric_group(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-textured",
            backend_name="shape_program",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.9,
                area_iou_mean=0.93,
                boundary_iou_mean=0.82,
                editability_score=0.86,
                extras={
                    "appearance": {
                        "source": "blend_export",
                        "required": True,
                        "uv": {
                            "has_uv_map": True,
                            "uv_valid": True,
                            "uv_island_count": 4,
                            "uv_overlap_ratio": 0.01,
                            "uv_out_of_bounds_ratio": 0.0,
                            "texel_density_cv": 0.22,
                        },
                        "texture": {
                            "texture_width": 1024,
                            "texture_height": 1024,
                            "texture_file_count": 2,
                            "texture_memory_mb": 8.0,
                            "reprojection_metrics": {
                                "psnr": 31.0,
                                "ssim": 0.91,
                                "lpips": 0.12,
                            },
                        },
                        "materials": {
                            "material_slot_count": 3,
                            "named_material_ratio": 1.0,
                            "pbr_channel_coverage": {
                                "base_color": True,
                                "roughness": True,
                                "metallic": True,
                                "normal": True,
                            },
                        },
                        "appearance_attribution": {
                            "boundary_geometry_fidelity": 0.86,
                            "texture_only_detail_score": 0.18,
                            "geometry_detail_score": 0.82,
                        },
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "pass")
        self.assertTrue(metrics["appearance.uv_valid"].value)
        self.assertEqual(metrics["appearance.texture_resolution"].value, 1024 * 1024)
        self.assertEqual(metrics["appearance.reprojection_psnr"].value, 31.0)
        self.assertEqual(metrics["appearance.pbr_channel_coverage_ratio"].value, 1.0)
        self.assertIn("uv=True", compact_console_summary(bundle))
        self.assertIn("| UV | PBR | Texture-only |", markdown_report([bundle]))

    def test_invalid_appearance_cannot_hide_behind_image_metrics(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-texture-cheat",
            backend_name="differentiable_refine",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.91,
                area_iou_mean=0.94,
                boundary_iou_mean=0.8,
                editability_score=0.74,
                extras={
                    "appearance": {
                        "required": True,
                        "strict_uv": True,
                        "has_uv_map": True,
                        "uv_valid": False,
                        "uv_overlap_ratio": 0.35,
                        "texture_memory_mb": 512.0,
                        "max_texture_memory_mb": 128.0,
                        "materials": {
                            "pbr_channel_coverage": {
                                "base_color": True,
                                "roughness": True,
                                "metallic": True,
                                "normal": True,
                            }
                        },
                        "appearance_attribution": {
                            "texture_only_detail_score": 0.92,
                            "geometry_detail_score": 0.30,
                            "boundary_geometry_fidelity": 0.42,
                        },
                    },
                    "novel_view": {
                        "psnr": 34.0,
                        "ssim": 0.94,
                        "lpips": 0.08,
                        "image_count": 2,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(bundle.status, "fail")
        self.assertEqual(metrics["appearance.uv_valid"].status, "fail")
        self.assertEqual(metrics["appearance.texture_memory_mb"].status, "fail")
        self.assertTrue(metrics["appearance.image_space_hallucination_warning"].value)
        self.assertIn("appearance_uv_invalid", codes)
        self.assertIn("appearance_texture_hides_geometry", codes)
        self.assertNotIn("novel_view_psnr_low", codes)

    def test_appearance_report_payload_accepts_target_collections(self) -> None:
        reports = appearance_reports_from_payload(
            {
                "targets": {
                    "blend": {
                        "has_uv_map": True,
                        "uv_valid": True,
                        "materials": {
                            "pbr_channel_coverage": {
                                "base_color": True,
                                "roughness": True,
                            }
                        },
                    },
                    "glb": {
                        "has_uv_map": False,
                        "required": True,
                        "warnings": ["glb_missing_uv"],
                    },
                }
            }
        )
        direct = appearance_report_from_mapping({"texture_width": 64, "texture_height": 32})

        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0].has_uv_map)
        self.assertIn("required_uv_map_missing", reports[1].errors)
        self.assertEqual(direct.texture_resolution, 2048)

    def test_success_without_required_silhouette_metrics_fails_bundle(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-c",
            backend_name="gaussian_ellipsoid_proxy",
            status="success",
            metric_result=CandidateMetrics(),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "fail")
        self.assertEqual(metrics["silhouette.min_view_iou"].status, "fail")
        self.assertIn("silhouette_required_metrics_missing", codes)

    def test_required_per_view_boundary_and_sdf_are_first_class(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-d",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                topology_score=0.9,
                editability_score=0.8,
                per_view={
                    "front": {
                        "area_iou": 0.82,
                        "boundary_iou": 0.62,
                        "signed_distance_loss": 0.12,
                        "required": True,
                        "passed": True,
                    },
                    "side": {
                        "area_iou": 0.78,
                        "boundary_iou": 0.55,
                        "signed_distance_loss": 0.16,
                        "required": True,
                        "passed": True,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()

        self.assertEqual(bundle.status, "pass")
        self.assertEqual(metrics["silhouette.required_view_count"].value, 2)
        self.assertEqual(metrics["silhouette.missing_required_metric_count"].value, 0)
        self.assertAlmostEqual(metrics["silhouette.min_boundary_iou"].value, 0.55)
        self.assertAlmostEqual(
            metrics["silhouette.mean_signed_distance_loss"].value,
            0.14,
        )

    def test_zero_iou_required_views_are_not_reported_as_missing_metrics(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-zero-view",
            backend_name="differentiable_refine",
            status="degraded",
            degraded=True,
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.0,
                        "boundary_iou": 0.05,
                        "signed_distance_loss": 0.9,
                        "required": True,
                        "passed": False,
                        "reason": "soft silhouette did not satisfy required view gate",
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertIn("silhouette_required_views_failed", codes)
        self.assertNotIn("silhouette_required_metrics_missing", codes)
        self.assertEqual(bundle.status, "fail")

    def test_classified_failures_influence_bundle_status(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-boundary",
            backend_name="gaussian_ellipsoid_proxy",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.92,
                        "boundary_iou": 0.30,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                    "side": {
                        "area_iou": 0.91,
                        "boundary_iou": 0.32,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        codes = {failure.code for failure in bundle.failures}

        self.assertIn("silhouette_boundary_blobby", codes)
        self.assertEqual(bundle.status, "fail")

    def test_recoverability_payload_splits_true_and_envelope_geometry(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-e",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.8,
                area_iou_mean=0.84,
                boundary_iou_mean=0.7,
                extras={
                    "recoverability": {
                        "source": "synthetic-test",
                        "true_geometry": {
                            "chamfer_l2": 0.08,
                            "fscore_tau": 0.42,
                            "volumetric_iou": 0.5,
                        },
                        "recoverable_geometry": {
                            "chamfer_l2": 0.02,
                            "fscore_tau": 0.82,
                            "volumetric_iou": 0.76,
                        },
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(metrics["geometry.true.fscore_tau"].value, 0.42)
        self.assertEqual(metrics["geometry.recoverable.fscore_tau"].value, 0.82)
        self.assertAlmostEqual(metrics["geometry.ambiguity_gap_chamfer_l2"].value, 0.06)
        self.assertIn("geometry_true_recoverable_gap_large", codes)
        autopsy = autopsy_pack_from_bundle(bundle).to_dict()
        self.assertTrue(
            any(
                action["action_id"] == "add_active_view"
                for action in autopsy["suggested_actions"]
            )
        )
        self.assertGreater(
            len(autopsy["active_view_plan"]["requests"]),
            0,
        )
        self.assertEqual(
            autopsy["active_view_plan"]["requests"][0]["view_id"],
            "front_side_45",
        )

    def test_visual_hull_diagnostics_are_first_class_failures(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-vh-diag",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.91,
                        "boundary_iou": 0.7,
                        "signed_distance_loss": 0.05,
                        "required": True,
                        "passed": True,
                    },
                    "top": {
                        "area_iou": 0.12,
                        "boundary_iou": 0.04,
                        "signed_distance_loss": 0.8,
                        "required": True,
                        "passed": False,
                    },
                },
                extras={
                    "visual_hull_view_diagnostics": {
                        "axis_or_transform_suspect": True,
                        "catastrophic_view_failure": True,
                        "failed_views": ["top"],
                        "top_like_failures": ["top"],
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        metrics = bundle.metric_index()
        codes = {failure.code for failure in bundle.failures}

        self.assertEqual(
            metrics["diagnostics.visual_hull.catastrophic_view_failure"].value,
            True,
        )
        self.assertIn("visual_hull_axis_or_transform_suspect", codes)
        self.assertIn("visual_hull_catastrophic_view_failure", codes)
        self.assertEqual(bundle.status, "fail")
        autopsy = autopsy_pack_from_bundle(bundle).to_dict()
        action_ids = {action["action_id"] for action in autopsy["suggested_actions"]}
        calibration_plan = autopsy["calibration_plan"]
        probe_ids = {probe["probe_id"] for probe in calibration_plan["probes"]}

        self.assertIn("run_calibration_sweep", action_ids)
        self.assertEqual(calibration_plan["recommended_track"], "visual-hull-transform")
        self.assertTrue(calibration_plan["trigger_metrics"]["axis_or_transform_suspect"])
        self.assertIn("axis_role_permutation_sweep", probe_ids)
        self.assertIn("per_view_offset_sweep", probe_ids)

    def test_topology_autopsy_includes_safe_repair_plan(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-topology",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                topology_score=0.55,
                extras={
                    "topology": {
                        "connected_components": 2,
                        "boundary_edges": 4,
                        "non_manifold_edges": 3,
                        "degenerate_faces": 1,
                        "loose_vertices": 2,
                        "watertight": False,
                    }
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        autopsy = autopsy_pack_from_bundle(bundle).to_dict()
        action_ids = {action["action_id"] for action in autopsy["suggested_actions"]}
        repair_plan = autopsy["topology_repair_plan"]
        repair_ops = {step["operation"] for step in repair_plan["steps"]}

        self.assertIn("safe_topology_repair", action_ids)
        self.assertEqual(repair_plan["status"], "repair_recommended")
        self.assertIn("drop_invalid_or_degenerate_faces", repair_ops)
        self.assertIn("drop_loose_vertices", repair_ops)
        self.assertIn("drop_or_label_small_components", repair_ops)
        self.assertIn("split_or_remove_non_manifold_faces", repair_ops)
        self.assertIn("hole_fill_or_remesh_required", repair_ops)

    def test_boundary_autopsy_includes_refinement_probe_plan(self) -> None:
        result = CandidateResult(
            candidate_id="candidate-boundary",
            backend_name="differentiable_refine",
            status="success",
            metric_result=CandidateMetrics(
                area_iou_min=0.88,
                area_iou_mean=0.90,
                boundary_iou_mean=0.48,
                per_view={
                    "front": {
                        "area_iou": 0.90,
                        "boundary_iou": 0.44,
                        "signed_distance_loss": 0.16,
                        "required": True,
                        "passed": True,
                    },
                    "side": {
                        "area_iou": 0.88,
                        "boundary_iou": 0.52,
                        "signed_distance_loss": 0.12,
                        "required": True,
                        "passed": True,
                    },
                },
            ),
        )

        bundle = bundle_from_candidate(result=result, repo="test")
        autopsy = autopsy_pack_from_bundle(bundle).to_dict()
        action_ids = {action["action_id"] for action in autopsy["suggested_actions"]}
        plan = autopsy["boundary_refinement_plan"]
        probe_ids = {probe["probe_id"] for probe in plan["probes"]}

        self.assertIn("silhouette_boundary_blobby", {failure.code for failure in bundle.failures})
        self.assertIn("boundary_first_refinement", action_ids)
        self.assertEqual(plan["recommended_track"], "content-adaptive-patches")
        self.assertIn("mask_threshold_sweep", probe_ids)
        self.assertIn("content_adaptive_patch_sweep", probe_ids)
        self.assertIn("differentiable_boundary_weight_sweep", probe_ids)


if __name__ == "__main__":
    unittest.main()
