"""Pure tests for optional dependency probes."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import utils.optional_deps as optional_deps
from utils.dependency_installer import (
    LPIPS_VERSION,
    PYTORCH_CPU_INDEX,
    SYMPY_VERSION,
    TORCH_CPU_VERSION,
    TORCHVISION_CPU_VERSION,
    blender_research_cpu_plan,
)
from utils.optional_deps import (
    clear_dependency_cache,
    dependency_report,
    optional_policy_decision,
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
        self.assertEqual(payload["attempts"][0]["module_name"], "__blendslop_missing_optional__")

    def test_dependency_report_includes_versions_for_available_modules(self) -> None:
        report = dependency_report(("json",))

        self.assertIn("json", report)
        self.assertTrue(report["json"]["available"])
        self.assertEqual(report["json"]["module_name"], "json")
        self.assertIsNotNone(report["json"]["module_file"])

    def test_known_dependency_metadata_explains_nvidia_only_nvdiffrast(self) -> None:
        dep = probe_dependency("nvdiffrast", cache=False)
        payload = dep.to_dict()

        self.assertEqual(payload["module_name"], "nvdiffrast")
        self.assertEqual(payload["gpu_vendor"], "NVIDIA")
        self.assertEqual(payload["accelerator"], "CUDA/OpenGL")
        self.assertFalse(payload["supports_rocm"])
        self.assertIn("NVlabs", payload["install_hint"])
        self.assertIn("ROCm", payload["rocm_alternative"])

    def test_openvdb_probe_checks_pyopenvdb_alias_first(self) -> None:
        fake = types.ModuleType("pyopenvdb")
        fake.__version__ = "test-openvdb"
        previous_pyopenvdb = sys.modules.get("pyopenvdb")
        previous_openvdb = sys.modules.get("openvdb")
        sys.modules["pyopenvdb"] = fake
        sys.modules.pop("openvdb", None)
        try:
            dep = probe_dependency("openvdb", cache=False)
        finally:
            if previous_pyopenvdb is None:
                sys.modules.pop("pyopenvdb", None)
            else:
                sys.modules["pyopenvdb"] = previous_pyopenvdb
            if previous_openvdb is None:
                sys.modules.pop("openvdb", None)
            else:
                sys.modules["openvdb"] = previous_openvdb

        payload = dep.to_dict()
        self.assertTrue(dep.available)
        self.assertEqual(payload["module_name"], "openvdb")
        self.assertEqual(payload["import_name"], "pyopenvdb")
        self.assertEqual(payload["resolved_module_name"], "pyopenvdb")
        self.assertEqual(payload["attempts"][0]["module_name"], "pyopenvdb")

    def test_openvdb_missing_report_includes_all_import_attempts(self) -> None:
        with patch.object(
            optional_deps.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("missing for test"),
        ):
            dep = probe_dependency("openvdb", cache=False)

        payload = dep.to_dict()
        self.assertFalse(dep.available)
        self.assertEqual(
            [attempt["module_name"] for attempt in payload["attempts"]],
            ["pyopenvdb", "openvdb"],
        )
        self.assertIn("NPZ", " ".join(payload["notes"]))

    def test_optional_policy_decision_maps_skip_and_fail(self) -> None:
        dep = probe_dependency("__blendslop_missing_optional__", cache=False)
        skip = optional_policy_decision(dep, policy="skip", feature="unit")
        fail = optional_policy_decision(dep, policy="fail", feature="unit")

        self.assertEqual(skip["result_status"], "skipped")
        self.assertEqual(skip["policy"], "skip")
        self.assertEqual(fail["result_status"], "failed")
        self.assertEqual(fail["policy"], "fail")
        self.assertIn("unit", fail["message"])

    def test_torch_dll_load_failure_reports_binary_runtime_diagnostic(self) -> None:
        with patch.object(
            optional_deps.importlib,
            "import_module",
            side_effect=OSError(
                "[WinError 1114] A dynamic link library initialization routine failed"
            ),
        ):
            dep = probe_dependency("torch", cache=False)

        payload = dep.to_dict()
        diagnostic = payload["details"]["diagnostic"]
        self.assertFalse(dep.available)
        self.assertEqual(diagnostic["category"], "binary_runtime_load_failure")
        self.assertIn("CPU", diagnostic["remediation"])

    def test_research_cpu_install_plan_pins_known_good_blender_stack(self) -> None:
        plan = blender_research_cpu_plan(python_executable="python-blender")
        commands = [" ".join(step.args) for step in plan.steps]

        self.assertEqual(plan.name, "blender-research-cpu")
        self.assertEqual(len(plan.steps), 3)
        self.assertIn(PYTORCH_CPU_INDEX, commands[0])
        self.assertIn(f"torch=={TORCH_CPU_VERSION}+cpu", commands[0])
        self.assertIn(
            f"torchvision=={TORCHVISION_CPU_VERSION}+cpu",
            commands[0],
        )
        self.assertIn(f"sympy=={SYMPY_VERSION}", commands[1])
        self.assertIn(f"lpips=={LPIPS_VERSION}", commands[2])
        self.assertIn("--no-deps", commands[2])

    def test_windows_blender_native_conflict_blocks_crash_prone_coimport(self) -> None:
        previous_bpy = sys.modules.get("bpy")
        previous_open3d = sys.modules.get("open3d")
        sys.modules["bpy"] = types.ModuleType("bpy")
        sys.modules["open3d"] = types.ModuleType("open3d")
        try:
            with patch.object(optional_deps.os, "name", "nt"):
                dep = probe_dependency("torch", cache=False)
        finally:
            if previous_bpy is None:
                sys.modules.pop("bpy", None)
            else:
                sys.modules["bpy"] = previous_bpy
            if previous_open3d is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = previous_open3d

        payload = dep.to_dict()
        diagnostic = payload["details"]["diagnostic"]
        self.assertFalse(dep.available)
        self.assertEqual(payload["error_type"], "NativeRuntimeConflict")
        self.assertEqual(diagnostic["category"], "native_runtime_conflict")
        self.assertEqual(diagnostic["loaded_module"], "open3d")


if __name__ == "__main__":
    unittest.main()
