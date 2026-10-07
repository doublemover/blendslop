from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from blender_blocking.moonshots import MoonshotRequest, run_experiment
from blender_blocking.refinement_lab.contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentResult,
    ExperimentVariant,
    utc_now,
)
from blender_blocking.refinement_lab.runner import BaseRunner, RunOptions


def _signals() -> dict[str, dict[str, object]]:
    return {
        "surface": {"available": True, "constraint_count": 3, "surface_point_count": 128},
        "profile": {
            "available": True,
            "view_count": 3,
            "band_samples": 24,
            "interval_count": 30,
            "hole_count": 1,
            "mean_width": 0.7,
            "max_width": 1.0,
            "complexity": 0.35,
        },
        "constraints": {
            "available": True,
            "constraint_count": 3,
            "constraint_views": {"front": 1, "side": 1, "top": 1},
        },
        "uncertainty": {
            "available": True,
            "overall_confidence_mean": 0.8,
            "overall_boundary_uncertainty_mean": 0.18,
            "consistency": 0.72,
            "view_details": {"top": {"boundary_uncertainty_mean": 0.28}},
        },
        "topology": {"available": True, "score": 0.68, "complexity": 0.35},
    }


def _candidate() -> dict[str, object]:
    return {
        "case_id": "case",
        "variant_id": "variant",
        "mode": "primitive_fit_refine",
        "status": "pass",
        "metrics": {
            "min_view_iou": 0.58,
            "average_iou": 0.72,
            "views": {
                "front": {"area_iou": 0.72, "boundary_iou": 0.45, "signed_distance_loss": 0.012},
                "side": {"area_iou": 0.69, "boundary_iou": 0.42, "signed_distance_loss": 0.018},
                "top": {"area_iou": 0.58, "boundary_iou": 0.31, "signed_distance_loss": 0.025},
            },
            "topology": {
                "topology_score": 0.62,
                "boundary_edges": 6,
                "non_manifold_edges": 2,
                "faces": 1200,
            },
            "editability_score": 0.48,
        },
        "backend_result": {
            "status": "success",
            "backend_name": "primitive_fit_refine",
            "metric_result": {
                "area_iou_min": 0.58,
                "boundary_iou_mean": 0.39,
                "topology_score": 0.62,
                "editability_score": 0.48,
                "extras": {
                    "primitive_count": 5,
                    "signal_summary": _signals(),
                    "primitives": {"count": 5},
                    "topology": {
                        "topology_score": 0.62,
                        "boundary_edges": 6,
                        "non_manifold_edges": 2,
                        "faces": 1200,
                    },
                },
            },
        },
    }


class MoonshotSidecarTests(unittest.TestCase):
    def _request(self, experiment_id: str, *, artifact_root: Path | None = None) -> MoonshotRequest:
        return MoonshotRequest(
            experiment_id=experiment_id,
            candidate=_candidate(),
            config={"target_signals": _signals()},
            artifact_root=artifact_root.as_posix() if artifact_root else None,
            allow_research_execution=True,
        )

    def test_shape_grammar_search_is_deterministic(self) -> None:
        first = run_experiment(self._request("shape_grammar_search")).to_dict()
        second = run_experiment(self._request("shape_grammar_search")).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "ran", first.get("errors") or first.get("degradation"))
        self.assertGreater(first["metrics"]["candidate_count"], 0)
        evidence = first["degradation"]["evidence"]
        self.assertTrue(evidence["selected_compile_plan"])
        self.assertTrue(evidence["family_hints"])
        self.assertIn("selected_fingerprint", evidence)
        self.assertGreaterEqual(first["metrics"]["valid_candidate_count"], 1.0)

    def test_active_view_planning_emits_ranked_requests(self) -> None:
        result = run_experiment(self._request("active_view_planning"))
        payload = result.to_dict()
        self.assertEqual(payload["status"], "ran")
        evidence = payload["degradation"]["evidence"]
        self.assertGreaterEqual(len(evidence["requests"]), 1)
        self.assertGreaterEqual(len(evidence["sequence_plan"]), 1)
        self.assertTrue(evidence["view_pair_pressure"])
        self.assertIn("ambiguity_index", evidence)
        self.assertGreater(payload["metrics"]["best_expected_metric_delta"], 0.0)
        self.assertGreater(payload["metrics"]["cumulative_expected_metric_delta"], 0.0)

    def test_sdf_and_retopology_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdf = run_experiment(self._request("implicit_sdf_proxy", artifact_root=root))
            topo = run_experiment(self._request("editable_retopology", artifact_root=root))
            self.assertEqual(sdf.status, "ran")
            self.assertEqual(topo.status, "ran")
            self.assertTrue(Path(sdf.artifacts["bundle"]).exists())
            self.assertTrue(Path(sdf.artifacts["extraction_plan"]).exists())
            self.assertTrue(Path(topo.artifacts["bundle"]).exists())
            self.assertTrue(sdf.degradation["evidence"]["extraction_plan"])
            self.assertTrue(topo.degradation["evidence"]["phase_plan"])
            self.assertTrue(topo.degradation["evidence"]["acceptance_gates"])
            self.assertGreater(topo.metrics["defect_count"], 0.0)

    def test_human_constraint_learning_and_dependency_statuses(self) -> None:
        human = run_experiment(self._request("human_constraint_learning"))
        self.assertEqual(human.status, "ran")
        self.assertGreaterEqual(human.metrics["constraint_count"], 1.0)
        self.assertGreaterEqual(human.metrics["training_example_count"], 1.0)
        self.assertTrue(human.degradation["evidence"]["ranking_priors"])
        self.assertTrue(human.degradation["evidence"]["training_examples"])
        unsupported = run_experiment(
            MoonshotRequest(
                experiment_id="differentiable_primitives",
                candidate={"backend_result": {"metric_result": {"extras": {}}}},
            )
        )
        self.assertEqual(unsupported.status, "unsupported")
        diff = run_experiment(self._request("differentiable_primitives"))
        self.assertEqual(diff.status, "ran")
        self.assertTrue(diff.degradation["evidence"]["finite_difference_schedule"])
        self.assertTrue(diff.degradation["evidence"]["objective_terms"])

    def test_portfolio_optimizer_ranks_prior_sidecar_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = run_experiment(self._request("active_view_planning", artifact_root=root))
            grammar = run_experiment(self._request("shape_grammar_search", artifact_root=root))
            sdf = run_experiment(self._request("implicit_sdf_proxy", artifact_root=root))
            topo = run_experiment(self._request("editable_retopology", artifact_root=root))
            human = run_experiment(self._request("human_constraint_learning", artifact_root=root))
            diff = run_experiment(self._request("differentiable_primitives", artifact_root=root))
            first = run_experiment(
                MoonshotRequest(
                    experiment_id="moonshot_portfolio_optimizer",
                    candidate=_candidate(),
                    config={
                        "target_signals": _signals(),
                        "prior_moonshot_results": (
                            active.to_dict(),
                            grammar.to_dict(),
                            sdf.to_dict(),
                            topo.to_dict(),
                            human.to_dict(),
                            diff.to_dict(),
                        ),
                        "portfolio_max_actions": 6,
                        "portfolio_budget": 6.0,
                    },
                    artifact_root=root.as_posix(),
                )
            )
            second = run_experiment(
                MoonshotRequest(
                    experiment_id="moonshot_portfolio_optimizer",
                    candidate=_candidate(),
                    config={
                        "target_signals": _signals(),
                        "prior_moonshot_results": (
                            active.to_dict(),
                            grammar.to_dict(),
                            sdf.to_dict(),
                            topo.to_dict(),
                            human.to_dict(),
                            diff.to_dict(),
                        ),
                        "portfolio_max_actions": 6,
                        "portfolio_budget": 6.0,
                    },
                    artifact_root=root.as_posix(),
                )
            )
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertEqual(first.status, "ran")
            self.assertGreater(first.metrics["selected_action_count"], 0.0)
            evidence = first.degradation["evidence"]
            self.assertTrue(evidence["selected_actions"])
            selected_sources = {
                action["source"]
                for action in evidence["selected_actions"]
                if isinstance(action, dict)
            }
            self.assertGreaterEqual(len(selected_sources), 3)
            selected_kinds = [
                action["kind"]
                for action in evidence["selected_actions"]
                if isinstance(action, dict)
            ]
            if "inject_learned_constraints" in selected_kinds and "compile_selected_shape_program" in selected_kinds:
                self.assertLess(
                    selected_kinds.index("inject_learned_constraints"),
                    selected_kinds.index("compile_selected_shape_program"),
                )
            self.assertTrue(evidence["available_actions"])
            self.assertTrue(evidence["rejected_actions"])
            self.assertTrue(evidence["gates"])
            self.assertTrue(evidence["execution_plan"])
            self.assertTrue(evidence["risk_register"])
            self.assertTrue(evidence["ranking_priors"])
            self.assertGreaterEqual(first.metrics["dependency_edge_count"], 1.0)
            self.assertGreaterEqual(first.metrics["execution_stage_count"], 2.0)
            self.assertTrue(Path(first.artifacts["bundle"]).exists())
            self.assertEqual(Path(first.artifacts["bundle"]).name, "bundle.json")

    def test_portfolio_optimizer_uses_short_bundle_name_for_long_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            long_variant = "shape_program-" + ("very_long_variant_segment_" * 6)
            root = Path(tmp) / "moonshots" / long_variant
            active = run_experiment(self._request("active_view_planning", artifact_root=root))
            portfolio = run_experiment(
                MoonshotRequest(
                    experiment_id="moonshot_portfolio_optimizer",
                    candidate=_candidate(),
                    config={
                        "target_signals": _signals(),
                        "prior_moonshot_results": (active.to_dict(),),
                    },
                    artifact_root=root.as_posix(),
                    allow_research_execution=True,
                )
            )

            self.assertEqual(portfolio.status, "ran")
            self.assertEqual(portfolio.errors, ())
            self.assertTrue(Path(portfolio.artifacts["bundle"]).exists())
            self.assertLessEqual(len(str(Path(portfolio.artifacts["bundle"]).resolve())), 258)


class RefinementMoonshotIntegrationTests(unittest.TestCase):
    def test_runner_attaches_moonshot_evidence_without_changing_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = ExperimentCase(
                case_id="synthetic-box",
                suite="synthetic",
                source="synthetic",
                synthetic_shape_id="box",
                synthetic_definition="box",
                metadata={"spec": {"parameters": {"primitive": "box", "width": 1.0}}},
            )
            variant = ExperimentVariant(
                variant_id="shape-baseline",
                label="shape baseline",
                mode="shape_program",
                validation_mode="backend-status",
            )
            plan = ExperimentPlan(
                plan_id="plan",
                suite="synthetic",
                track="shape-program-editability",
                search="coordinate",
                objective="profile_editable",
                output_root=root,
                run_id="run",
                cases=(case,),
                variants=(variant,),
            )
            runner = BaseRunner(
                plan=plan,
                options=RunOptions(
                    write_bounds_debug=False,
                    write_autopsy=False,
                    moonshot_sidecars=True,
                    moonshot_experiments=(
                        "active_view_planning",
                        "human_constraint_learning",
                        "moonshot_portfolio_optimizer",
                    ),
                ),
            )
            now = utc_now()
            result = ExperimentResult(
                run_id="run",
                case_id=case.case_id,
                variant_id=variant.variant_id,
                mode=variant.mode,
                status="pass",
                exit_code=0,
                started_utc=now,
                finished_utc=now,
                elapsed_s=0.1,
                metrics={
                    "validation_mode": "backend-status",
                    "topology": {"score": 0.7},
                    "editability_score": 0.5,
                },
                backend_result={
                    "status": "research_only",
                    "metric_result": {
                        "extras": {
                            "signal_summary": _signals(),
                            "primitive_count": 2,
                        }
                    },
                },
            )
            processed = runner._postprocess_result(case, variant, result)
            self.assertEqual(processed.status, "pass")
            self.assertIn("moonshots", processed.metrics)
            self.assertEqual(processed.metrics["moonshots"]["ran_count"], 3)
            self.assertTrue(processed.metrics["moonshots"]["portfolio_actions"])
            self.assertIn("moonshot_summary", processed.artifacts)
            self.assertTrue(processed.artifacts["moonshot_summary"].exists())
            self.assertIn("moonshot_evidence", processed.backend_result)


if __name__ == "__main__":
    unittest.main()
