"""Regression checks for Blender launcher flags versus runner arguments."""

import unittest
from unittest.mock import patch

import test_runner


class RunnerArgumentTests(unittest.TestCase):
    def runner_args(self, argv, *, blender):
        with patch.object(test_runner.sys, "argv", argv), patch.object(
            test_runner, "check_blender_available", return_value=(blender, None)
        ):
            return test_runner._runner_argv()

    def test_documented_blender_full_command_has_no_script_arguments(self):
        argv = ["blender", "--background", "--python", "test_runner.py"]
        self.assertEqual(self.runner_args(argv, blender=True), [])

    def test_headless_ci_launcher_options_are_not_runner_options(self):
        argv = ["blender", "--background", "--factory-startup",
                "--python-exit-code", "1", "--python", "test_runner.py"]
        args = test_runner._parse_args(self.runner_args(argv, blender=True))
        self.assertFalse(args.quick)
        self.assertIsNone(args.phase)

    def test_blender_separator_preserves_quick_arguments(self):
        argv = ["blender", "--background", "--python", "test_runner.py",
                "--", "--quick"]
        args = test_runner._parse_args(self.runner_args(argv, blender=True))
        self.assertTrue(args.quick)

    def test_blender_separator_preserves_phase_arguments(self):
        argv = ["blender", "--python", "test_runner.py", "--", "--phase", "pure"]
        self.assertEqual(self.runner_args(argv, blender=True), ["--phase", "pure"])

    def test_plain_python_preserves_direct_arguments(self):
        argv = ["test_runner.py", "--phase", "pure"]
        args = test_runner._parse_args(self.runner_args(argv, blender=False))
        self.assertEqual(args.phase, ["pure"])

    def test_plain_python_separator_is_supported(self):
        argv = ["test_runner.py", "--", "--verbose"]
        args = test_runner._parse_args(self.runner_args(argv, blender=False))
        self.assertTrue(args.verbose)


if __name__ == "__main__":
    unittest.main()
