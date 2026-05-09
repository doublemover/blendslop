"""Pure tests for content-adaptive patch refinement helpers."""

from __future__ import annotations

import unittest

import numpy as np

from refinement_lab.content_adaptive_patches import (
    fuse_patch_predictions,
    score_map_from_signals,
    select_adaptive_patches,
)
from refinement_lab.adaptive_planner import proposals_from_bundle


class ContentAdaptivePatchTests(unittest.TestCase):
    def test_score_map_and_nms_select_high_residual_regions(self) -> None:
        residual = np.zeros((48, 48), dtype=float)
        residual[8:14, 9:15] = 1.0
        residual[34:40, 33:39] = 0.8
        uncertainty = np.zeros_like(residual)
        uncertainty[34:40, 33:39] = 1.0

        score = score_map_from_signals(residual=residual, uncertainty=uncertainty)
        patches = select_adaptive_patches(
            score,
            patch_sizes=(12,),
            max_patches=2,
            margin_fraction=0.25,
            max_iou=0.1,
        )

        self.assertEqual(len(patches), 2)
        centers = [patch.box.center for patch in patches]
        self.assertTrue(any(center[0] < 20 and center[1] < 20 for center in centers))
        self.assertTrue(any(center[0] > 25 and center[1] > 25 for center in centers))
        self.assertGreater(patches[0].crop_box.area, patches[0].box.area)

    def test_patch_fusion_aligns_and_feathers_predictions(self) -> None:
        global_prediction = np.zeros((16, 16), dtype=float)
        score = np.zeros((16, 16), dtype=float)
        score[6:10, 6:10] = 1.0
        patch = select_adaptive_patches(score, patch_sizes=(8,), max_patches=1)[0]
        patch_prediction = (
            np.ones((patch.crop_box.height, patch.crop_box.width), dtype=float) * 2.0
        )

        fused = fuse_patch_predictions(
            global_prediction,
            (patch,),
            {patch.patch_id: patch_prediction},
            align_mean=False,
        )

        self.assertEqual(fused.prediction.shape, global_prediction.shape)
        self.assertGreater(float(np.max(fused.contribution_weight)), 0.0)
        self.assertGreater(float(np.max(fused.prediction)), 0.0)
        self.assertEqual(fused.metadata["used_patch_count"], 1)

    def test_patch_fusion_can_affine_align_local_predictions(self) -> None:
        yy, xx = np.mgrid[0:20, 0:20]
        global_prediction = (yy + xx).astype(float)
        score = np.zeros((20, 20), dtype=float)
        score[7:13, 7:13] = 1.0
        patch = select_adaptive_patches(score, patch_sizes=(10,), max_patches=1)[0]
        crop = global_prediction[
            patch.crop_box.y0 : patch.crop_box.y1,
            patch.crop_box.x0 : patch.crop_box.x1,
        ]
        local_prediction = (crop - 3.0) / 2.0

        fused = fuse_patch_predictions(
            global_prediction,
            (patch,),
            {patch.patch_id: local_prediction},
            alignment_mode="affine",
            edge_weight_map=score,
            edge_weight_strength=1.0,
        )

        alignment = fused.metadata["alignments"][0]
        self.assertEqual(alignment["mode"], "affine")
        self.assertAlmostEqual(alignment["scale"], 2.0)
        self.assertAlmostEqual(alignment["shift"], 3.0)
        self.assertLess(
            float(np.max(np.abs(fused.prediction - global_prediction))),
            1e-8,
        )
        self.assertGreater(float(np.max(fused.contribution_weight)), 1.0)

    def test_adaptive_planner_proposes_content_patch_pass_for_detail_failures(
        self,
    ) -> None:
        bundle = {
            "status": "pass",
            "failures": [{"code": "surface_detail_underfit"}],
            "metric_groups": [
                {
                    "name": "silhouette",
                    "metrics": [
                        {"name": "silhouette.min_view_iou", "value": 0.82},
                        {"name": "silhouette.mean_boundary_iou", "value": 0.42},
                        {"name": "silhouette.mean_signed_distance_loss", "value": 0.12},
                    ],
                },
                {
                    "name": "geometry",
                    "metrics": [
                        {"name": "geometry.fscore_tau", "value": 0.4},
                        {"name": "geometry.surface_coverage", "value": 0.55},
                    ],
                },
            ],
        }

        proposals = proposals_from_bundle(bundle)
        titles = {proposal.title for proposal in proposals}

        self.assertIn("Content-adaptive patch detail pass", titles)
        patch = next(
            proposal
            for proposal in proposals
            if proposal.title == "Content-adaptive patch detail pass"
        )
        self.assertIn("content-adaptive-patches", patch.tags)
        self.assertIn("--shape-residual-policy", patch.cli_args)


if __name__ == "__main__":
    unittest.main()
