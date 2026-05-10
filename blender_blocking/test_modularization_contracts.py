from __future__ import annotations

import unittest


class ModularizationContractTests(unittest.TestCase):
    def test_benchmark_contract_dataclasses(self) -> None:
        from benchmarks.contracts import BenchmarkCase, BenchResult

        result = BenchResult(
            name="sample",
            iterations=2,
            elapsed_s=0.5,
            per_iter_ms=250.0,
            meta={"rate": 4.0},
        )
        case = BenchmarkCase(
            name="sample-case",
            benches=("sample",),
            description="sample benchmark case",
        )

        self.assertEqual(result.meta["rate"], 4.0)
        self.assertEqual(case.benches, ("sample",))

    def test_adaptive_contracts_are_real_owners(self) -> None:
        from refinement_lab.adaptive.contracts import RefinementProposal
        from refinement_lab.adaptive.selection import merge_proposals
        from refinement_lab.adaptive.signals import ambiguity_signal

        proposal = RefinementProposal(
            proposal_id="boundary-first",
            title="Boundary first",
            hypothesis="Boundary metric is lagging.",
            priority=5,
        )
        variant = proposal.to_variant(parent_variant_id="baseline")

        self.assertEqual(proposal.to_dict()["proposal_id"], "boundary-first")
        self.assertEqual(variant.parent_variant_id, "baseline")
        self.assertTrue(ambiguity_signal({"geometry.ambiguity_gap": 0.2}, ()))
        self.assertEqual(merge_proposals((proposal,))[0], proposal)

    def test_resfit_contract_modules_import(self) -> None:
        from placement.resfit.artifacts import resfit_history_records
        from placement.resfit.config import ResFitPipelineConfig
        from placement.resfit.status import resfit_candidate_status

        config = ResFitPipelineConfig()

        self.assertEqual(config.validate(), ())
        self.assertEqual(resfit_history_records(()), [])
        self.assertTrue(callable(resfit_candidate_status))

    def test_reconstruction_backend_helpers_import(self) -> None:
        from reconstruction.backends.visual_hull.config import visual_hull_run_config
        from reconstruction.backends.visual_hull.status import visual_hull_result_status
        from reconstruction.differentiable.history import build_refinement_history_payloads
        from reconstruction.differentiable.optimization import run_differentiable_optimization
        from reconstruction.point_cloud.contracts import PointCloudArtifactStatus

        config = visual_hull_run_config({"resolution": 12, "backend": "chunked"})
        status = PointCloudArtifactStatus(source="synthetic", point_count=3)

        self.assertEqual(config.resolution, 12)
        self.assertEqual(config.requested_backend, "chunked")
        self.assertEqual(status.to_dict()["point_count"], 3)
        self.assertTrue(callable(visual_hull_result_status))
        self.assertTrue(callable(build_refinement_history_payloads))
        self.assertTrue(callable(run_differentiable_optimization))


if __name__ == "__main__":
    unittest.main()
