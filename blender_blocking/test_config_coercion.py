"""Tests for shared config coercion helpers."""

from __future__ import annotations

import math
import unittest

try:
    from config_models.coercion import (
        coerce_float,
        coerce_int,
        coerce_optional_float,
        coerce_optional_int,
    )
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.config_models.coercion import (
        coerce_float,
        coerce_int,
        coerce_optional_float,
        coerce_optional_int,
    )


class ConfigCoercionTests(unittest.TestCase):
    def test_coerce_int_applies_bounds_and_defaults(self) -> None:
        errors: list[str] = []

        self.assertEqual(coerce_int("5", "samples", errors, min_value=1), 5)
        self.assertEqual(
            coerce_int(0, "samples", errors, default=8, min_value=1, default_on_bounds=True),
            8,
        )
        self.assertEqual(coerce_optional_int(None, "limit", errors, default=4), 4)

        self.assertTrue(any("samples must be >=" in error for error in errors))

    def test_coerce_float_rejects_bool_and_non_finite_values(self) -> None:
        errors: list[str] = []

        self.assertEqual(coerce_float("0.5", "weight", errors, min_value=0.0), 0.5)
        self.assertIsNone(coerce_float(True, "weight", errors))
        self.assertIsNone(coerce_float(math.inf, "weight", errors))
        self.assertEqual(
            coerce_optional_float(
                -1.0,
                "weight",
                errors,
                default=0.25,
                min_value=0.0,
                default_on_bounds=True,
            ),
            0.25,
        )

        self.assertTrue(any("got True" in error for error in errors))
        self.assertTrue(any("must be finite" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
