"""Tests for closed-loop adaptive refinement orchestration."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from blender_blocking.refinement_lab.adaptive import proposals_from_bundle
from blender_blocking.refinement_lab.adaptive_loop import (
    AdaptiveLoopOptions,
    _child_variants_from_results,
    run_adaptive_loop,
)
from blender_blocking.refinement_lab.contracts import ExperimentPlan, ExperimentResult
from blender_blocking.refinement_lab.runner import RunOptions


class _FakeRunner:
    def __init__(self, harness: "_FakeHarness", plan: ExperimentPlan) -> None:
        self.harness = harness
        self.plan = plan

    def run(self) -> tuple[bool, list[ExperimentResult]]:
        generation = len(self.harness.plans)
        self.harness.plans.append(self.plan)
        variant = self.plan.variants[0]
        if generation == 0:
            return True, [
                ExperimentResult(
                    run_id=self.plan.run_id,
                    case_id=self.plan.cases[0].case_id,
                    variant_id=variant.variant_id,
                    mode=variant.mode,
                    status="pass",
                    exit_code=0,
                    started_utc="s",
                    finished_utc="f",
                    elapsed_s=1.0,
                    backend_result={
                        "status": "success",
                        "degraded": False,
                        "metric_result": {
                            "area_iou_min": 0.42,
                            "area_iou_mean": 0.58,
                            "boundary_iou_mean": 0.18,
                            "signed_distance_loss_mean": 0.2,
                            "topology_score": 0.4,
                            "editability_score": 0.3,
                        },
                    },
                    metrics={
                        "average_iou": 0.58,
                        "front_iou": 0.58,
                        "side_iou": 0.42,
                        "top_iou": 0.74,
                    },
                )
            ]
        return True, [
            ExperimentResult(
                run_id=self.plan.run_id,
                case_id=self.plan.cases[0].case_id,
                variant_id=variant.variant_id,
                mode=variant.mode,
                status="pass",
                exit_code=0,
                started_utc="s",
                finished_utc="f",
                elapsed_s=1.0,
                backend_result={
                    "status": "success",
                    "metric_result": {
                        "area_iou_min": 0.93,
                        "area_iou_mean": 0.95,
                        "boundary_iou_mean": 0.92,
                        "signed_distance_loss_mean": 0.01,
                        "topology_score": 0.97,
                        "editability_score": 0.88,
                        "geometry_fscore_tau": 0.9,
                        "geometry_surface_coverage": 0.9,
                    },
                },
                metrics={
                    "average_iou": 0.95,
                    "front_iou": 0.95,
                    "side_iou": 0.93,
                    "top_iou": 0.97,
                },
            )
        ]


class _FakeHarness:
    def __init__(self) -> None:
        self.plans: list[ExperimentPlan] = []

    def factory(
        self,
        plan: ExperimentPlan,
        _options: RunOptions,
        _base_config: object,
    ) -> _FakeRunner:
        return _FakeRunner(self, plan)


class AdaptiveLoopTests(unittest.TestCase):
    def test_adaptive_planner_proposes_appearance_asset_audit(self) -> None:
        bundle = {
            "status": "fail",
            "metric_groups": [
                {
                    "name": "appearance",
                    "status": "fail",
                    "metrics": [
                        {"name": "appearance.uv_valid", "value": False},
                        {
                            "name": "appearance.pbr_channel_coverage_ratio",
                            "value": 0.25,
                        },
                        {
                            "name": "appearance.attribution_texture_only_detail_score",
                            "value": 0.9,
                        },
                        {
                            "name": "appearance.attribution_geometry_detail_score",
                            "value": 0.35,
                        },
                    ],
                }
            ],
            "failures": [
                {
                    "code": "appearance_texture_hides_geometry",
                    "severity": "warn",
                    "subsystem": "appearance",
                }
            ],
        }

        proposals = proposals_from_bundle(bundle)
        by_title = {proposal.title: proposal for proposal in proposals}

        self.assertIn("Texture, UV, and material audit", by_title)
        proposal = by_title["Texture, UV, and material audit"]
        self.assertIn("--evaluate-texture-materials", proposal.cli_args)
        self.assertIn("--uv-strict", proposal.cli_args)
        self.assertIn("appearance", proposal.tags)

    def test_loop_runs_child_generation_from_ranked_adaptive_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = _FakeHarness()
            summary = run_adaptive_loop(
                suite="default-vase",
                track="profile-loft-refinement",
                search="coordinate",
                objective="quality_win",
                output_root=Path(tmp) / "loop",
                max_runs=1,
                options=AdaptiveLoopOptions(
                    generations=2,
                    parent_top_k=1,
                    children_per_parent=2,
                    run_options=RunOptions(
                        html_report=False,
                        write_overlays=False,
                        append_global_index=False,
                        write_lineage=False,
                    ),
                ),
                runner_factory=harness.factory,
            )

            self.assertEqual(len(harness.plans), 2)
            self.assertEqual(len(summary.generations), 2)
            self.assertEqual(summary.generations[0].child_variant_count, 2)
            self.assertEqual(summary.generations[1].child_variant_count, 0)
            self.assertEqual(summary.stopped_reason, "no_child_variants")
            self.assertTrue(summary.summary_path.exists())
            child = harness.plans[1].variants[0]
            self.assertEqual(child.parent_variant_id, harness.plans[0].variants[0].variant_id)
            self.assertIn("adaptive-loop", child.tags)
            self.assertTrue(child.variant_id.startswith("g01_"))

            payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "refinement_adaptive_loop_v1")
            self.assertEqual(payload["generation_count"], 2)
            self.assertTrue(
                (Path(tmp) / "loop" / "g00" / "adaptive-loop-variants.json").exists()
            )

    def test_loop_stops_when_no_promotable_parent_exists(self) -> None:
        class BlockedRunner(_FakeRunner):
            def run(self) -> tuple[bool, list[ExperimentResult]]:
                self.harness.plans.append(self.plan)
                variant = self.plan.variants[0]
                return True, [
                    ExperimentResult(
                        run_id=self.plan.run_id,
                        case_id=self.plan.cases[0].case_id,
                        variant_id=variant.variant_id,
                        mode=variant.mode,
                        status="pass",
                        exit_code=0,
                        started_utc="s",
                        finished_utc="f",
                        elapsed_s=1.0,
                        backend_result={
                            "status": "success",
                            "metric_result": {
                                "area_iou_min": 0.99,
                                "area_iou_mean": 0.99,
                            },
                        },
                        metrics={
                            "validation_mode": "backend-status",
                            "backend": {
                                "area_iou_min": 0.99,
                                "area_iou_mean": 0.99,
                            },
                        },
                    )
                ]

        class BlockedHarness(_FakeHarness):
            def factory(
                self,
                plan: ExperimentPlan,
                _options: RunOptions,
                _base_config: object,
            ) -> BlockedRunner:
                return BlockedRunner(self, plan)

        with tempfile.TemporaryDirectory() as tmp:
            harness = BlockedHarness()
            summary = run_adaptive_loop(
                suite="default-vase",
                track="profile-loft-refinement",
                search="coordinate",
                objective="reliability_first",
                output_root=Path(tmp) / "loop",
                max_runs=1,
                options=AdaptiveLoopOptions(
                    generations=2,
                    parent_top_k=1,
                    children_per_parent=2,
                    run_options=RunOptions(
                        html_report=False,
                        write_overlays=False,
                        append_global_index=False,
                        write_lineage=False,
                    ),
                ),
                runner_factory=harness.factory,
            )

            self.assertEqual(len(harness.plans), 1)
            self.assertEqual(summary.stopped_reason, "no_promotable_parents")
            self.assertEqual(summary.generations[0].selected_parent_ids, ())
            self.assertEqual(summary.generations[0].child_variant_count, 0)
            self.assertEqual(summary.generations[0].parent_health["promotable_count"], 0)
            self.assertEqual(
                summary.generations[0].parent_health["promotion_state_counts"][
                    "metric_only_candidate"
                ],
                1,
            )

            proposal_payload = json.loads(
                summary.generations[0].proposal_path.read_text(encoding="utf-8")
            )
            self.assertEqual(proposal_payload["parent_health"]["promotable_count"], 0)
            self.assertEqual(
                proposal_payload["parent_health"]["top_render_winners"][0][
                    "promotion_state"
                ],
                "metric_only_candidate",
            )

    def test_child_variant_dedupe_preserves_contributing_parents(self) -> None:
        def result(parent_id: str) -> ExperimentResult:
            return ExperimentResult(
                run_id="run",
                case_id="case",
                variant_id=parent_id,
                mode="ensemble",
                status="pass",
                exit_code=0,
                started_utc="s",
                finished_utc="f",
                elapsed_s=1.0,
                backend_result={
                    "status": "success",
                    "metric_result": {
                        "area_iou_min": 0.42,
                        "area_iou_mean": 0.58,
                        "boundary_iou_mean": 0.18,
                        "signed_distance_loss_mean": 0.2,
                        "topology_score": 0.4,
                        "editability_score": 0.3,
                    },
                },
                metrics={
                    "average_iou": 0.58,
                    "front_iou": 0.58,
                    "side_iou": 0.42,
                    "top_iou": 0.74,
                },
            )

        proposals, variants = _child_variants_from_results(
            (result("parent-a"), result("parent-b")),
            generation=0,
            children_per_parent=2,
        )

        self.assertEqual(len(proposals), 4)
        self.assertEqual(len(variants), 2)
        contributing = variants[0].parameters[
            "adaptive_loop_contributing_parent_results"
        ]
        self.assertIn("parent-a", contributing)
        self.assertIn("parent-b", contributing)

    def test_loop_writes_summary_when_runner_raises(self) -> None:
        class RaisingHarness(_FakeHarness):
            def factory(
                self,
                plan: ExperimentPlan,
                _options: RunOptions,
                _base_config: object,
            ) -> object:
                self.plans.append(plan)

                class RaisingRunner:
                    def run(self) -> tuple[bool, list[ExperimentResult]]:
                        raise RuntimeError("synthetic family cannot render")

                return RaisingRunner()

        with tempfile.TemporaryDirectory() as tmp:
            harness = RaisingHarness()
            summary = run_adaptive_loop(
                suite="default-vase",
                track="profile-loft-refinement",
                search="coordinate",
                objective="reliability_first",
                output_root=Path(tmp) / "loop",
                max_runs=1,
                options=AdaptiveLoopOptions(
                    generations=2,
                    parent_top_k=1,
                    children_per_parent=2,
                    run_options=RunOptions(
                        html_report=False,
                        write_overlays=False,
                        append_global_index=False,
                        write_lineage=False,
                    ),
                ),
                runner_factory=harness.factory,
            )

            self.assertEqual(summary.stopped_reason, "runner_exception")
            self.assertEqual(len(summary.generations), 1)
            self.assertFalse(summary.generations[0].ok)
            self.assertEqual(summary.generations[0].failure_code, "runner_exception")
            self.assertTrue(summary.summary_path.exists())
            payload = json.loads(summary.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stopped_reason"], "runner_exception")
            self.assertIn("error_path", payload["generations"][0])


if __name__ == "__main__":
    unittest.main()
