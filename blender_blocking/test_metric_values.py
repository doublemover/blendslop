"""Tests for shared metric value helpers."""

from __future__ import annotations

import math
import unittest

try:
    from metrics.values import (
        finite_float_or_none,
        float_or,
        get_metric_path,
        optional_float,
    )
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.metrics.values import (
        finite_float_or_none,
        float_or,
        get_metric_path,
        optional_float,
    )


class MetricValuesTests(unittest.TestCase):
    def test_float_helpers_handle_missing_bad_and_non_finite_values(self) -> None:
        self.assertEqual(optional_float("1.5"), 1.5)
        self.assertIsNone(optional_float(object()))
        self.assertEqual(float_or("bad", default=2.0), 2.0)
        self.assertIsNone(finite_float_or_none(math.inf))
        self.assertIsNone(finite_float_or_none(math.nan))
        self.assertEqual(finite_float_or_none("3.25"), 3.25)

    def test_get_metric_path_reads_flat_or_nested_metrics(self) -> None:
        metrics = {
            "metrics.render.min_view_iou": 0.4,
            "metrics": {"render": {"mean_view_iou": 0.7}},
        }

        self.assertEqual(get_metric_path(metrics, "metrics.render.min_view_iou"), 0.4)
        self.assertEqual(get_metric_path(metrics, "metrics.render.mean_view_iou"), 0.7)
        self.assertEqual(get_metric_path(metrics, "metrics.missing", "n/a"), "n/a")


if __name__ == "__main__":
    unittest.main()
