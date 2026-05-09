"""Pure tests for content-adaptive patch refinement helpers."""

from __future__ import annotations

import unittest

import numpy as np

from refinement_lab.content_adaptive_patches import (
    fuse_patch_predictions,
    score_map_from_signals,
    select_adaptive_patches,
)


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
        patch_prediction = np.ones((patch.crop_box.height, patch.crop_box.width), dtype=float) * 2.0

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


if __name__ == "__main__":
    unittest.main()
