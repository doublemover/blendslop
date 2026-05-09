"""Pure tests for optional dependency probes."""

from __future__ import annotations

import unittest

from utils.optional_deps import (
    clear_dependency_cache,
    dependency_report,
    probe_dependency,
)


class OptionalDependencyTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_dependency_cache()

    def test_missing_dependency_report_is_json_safe(self) -> None:
        dep = probe_dependency("__blendslop_missing_optional__", cache=False)
        payload = dep.to_dict()

        self.assertFalse(dep.available)
        self.assertFalse(dep.skip_reason == "")
        self.assertEqual(payload["module_name"], "__blendslop_missing_optional__")
        self.assertFalse(payload["available"])
        self.assertEqual(payload["error_type"], "ModuleNotFoundError")
        self.assertIsNone(payload["module_version"])
        self.assertIsNone(payload["module_file"])

    def test_dependency_report_includes_versions_for_available_modules(self) -> None:
        report = dependency_report(("json",))

        self.assertIn("json", report)
        self.assertTrue(report["json"]["available"])
        self.assertEqual(report["json"]["module_name"], "json")
        self.assertIsNotNone(report["json"]["module_file"])


if __name__ == "__main__":
    unittest.main()
