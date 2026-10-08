"""Pure tests for calibrated metric baseline selection."""

from __future__ import annotations

import unittest

from evaluation.baselines import (
    BaselineQuery,
    BaselineSlice,
    MetricDistribution,
    QualityBaseline,
    quality_baseline_from_mapping,
    select_baseline_slice,
    thresholds_for_query,
)


class EvaluationBaselineTests(unittest.TestCase):
    def test_selects_exact_slice_and_derives_thresholds_with_provenance(self) -> None:
        baseline = _baseline(
            _slice(
                suite="synthetic-smoke",
                mode="visual_hull_voxel",
                shape_family="vehicle",
                mask_noise_profile="anti_aliased",
                resolution=64,
                dependency_profile="skimage",
                metric="silhouette.min_view_iou",
                values=(0.82, 0.88, 0.92, 0.94),
                higher_is_better=True,
            )
        )

        thresholds, match = thresholds_for_query(
            baseline,
            BaselineQuery(
                suite="synthetic-smoke",
                mode="visual_hull_voxel",
                shape_family="vehicle",
                mask_noise_profile="anti_aliased",
                resolution=64,
                dependency_profile="skimage",
            ),
        )

        self.assertTrue(match.exact)
        self.assertEqual(
            match.fallback_level,
            "mode+suite+shape_family+mask_noise_profile+resolution+dependency_profile",
        )
        self.assertEqual(match.mismatched_fields, {})
        self.assertEqual(len(thresholds), 1)
        self.assertEqual(thresholds[0].metric, "silhouette.min_view_iou")
        self.assertIn("derived_baseline:", thresholds[0].source)
        self.assertIsNotNone(thresholds[0].min_value)

    def test_fallback_records_mismatched_dimensions(self) -> None:
        exact_noise = _slice(
            suite="synthetic-smoke",
            mode="visual_hull_voxel",
            shape_family="vehicle",
            mask_noise_profile="clean_binary",
            resolution=32,
            metric="silhouette.min_boundary_iou",
            values=(0.55, 0.62, 0.71),
            higher_is_better=True,
        )
        mode_suite = _slice(
            suite="synthetic-smoke",
            mode="visual_hull_voxel",
            shape_family="",
            mask_noise_profile="",
            resolution=64,
            metric="silhouette.min_boundary_iou",
            values=(0.45, 0.52, 0.60),
            higher_is_better=True,
        )
        baseline = _baseline(exact_noise, mode_suite)

        match = select_baseline_slice(
            baseline,
            {
                "suite": "synthetic-smoke",
                "mode": "visual_hull_voxel",
                "shape_family": "vehicle",
                "mask_noise_profile": "sketch_gaps",
                "resolution": 64,
            },
        )

        self.assertFalse(match.exact)
        self.assertEqual(match.fallback_level, "mode+suite+shape_family")
        self.assertIs(match.slice, exact_noise)
        self.assertIn("mask_noise_profile", match.mismatched_fields)
        self.assertIn("resolution", match.mismatched_fields)
        self.assertIn("baseline selected via", match.warnings[0])

    def test_mapping_round_trip_preserves_threshold_selectability(self) -> None:
        baseline = _baseline(
            _slice(
                suite="profile-smoke",
                mode="profile_loft",
                shape_family="vessel",
                mask_noise_profile="clean_binary",
                metric="novel_view.mse",
                values=(0.02, 0.03, 0.05),
                higher_is_better=False,
            )
        )
        restored = quality_baseline_from_mapping(baseline.to_dict())
        thresholds, match = thresholds_for_query(
            restored,
            {
                "suite": "profile-smoke",
                "mode": "profile_loft",
                "shape_family": "vessel",
                "mask_noise_profile": "clean_binary",
            },
        )

        self.assertEqual(match.slice.mode, "profile_loft")
        self.assertEqual(thresholds[0].metric, "novel_view.mse")
        self.assertIsNotNone(thresholds[0].max_value)
        self.assertIsNone(thresholds[0].min_value)

    def test_no_match_returns_empty_thresholds_and_warning(self) -> None:
        thresholds, match = thresholds_for_query(
            _baseline(),
            BaselineQuery(suite="synthetic-smoke", mode="ensemble"),
        )

        self.assertEqual(thresholds, ())
        self.assertIsNone(match.slice)
        self.assertEqual(match.fallback_level, "none")
        self.assertIn("no compatible baseline", match.warnings[0])


def _baseline(*slices: BaselineSlice) -> QualityBaseline:
    return QualityBaseline(
        schema_version="quality-baseline-v1",
        created_at_utc="2026-05-09T00:00:00Z",
        repo_revision="abc123",
        environment_hash="local",
        slices=tuple(slices),
    )


def _slice(
    *,
    suite: str,
    mode: str,
    shape_family: str,
    mask_noise_profile: str,
    metric: str,
    values: tuple[float, ...],
    higher_is_better: bool,
    resolution: int = 0,
    dependency_profile: str = "",
) -> BaselineSlice:
    distribution = _distribution(metric, values, higher_is_better)
    return BaselineSlice(
        suite=suite,
        mode=mode,
        shape_family=shape_family,
        view_count=3,
        mask_noise_profile=mask_noise_profile,
        resolution=resolution,
        dependency_profile=dependency_profile,
        distributions=(distribution,),
    )


def _distribution(
    metric: str,
    values: tuple[float, ...],
    higher_is_better: bool,
) -> MetricDistribution:
    sorted_values = sorted(values)
    mean = sum(sorted_values) / len(sorted_values)
    return MetricDistribution(
        metric=metric,
        sample_count=len(sorted_values),
        min_value=sorted_values[0],
        max_value=sorted_values[-1],
        mean=mean,
        median=sorted_values[len(sorted_values) // 2],
        p05=sorted_values[0],
        p25=sorted_values[0],
        p75=sorted_values[-1],
        p95=sorted_values[-1],
        stddev=0.01,
        higher_is_better=higher_is_better,
    )


if __name__ == "__main__":
    unittest.main()
