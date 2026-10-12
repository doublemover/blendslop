from __future__ import annotations

import unittest


class ModularizationContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
