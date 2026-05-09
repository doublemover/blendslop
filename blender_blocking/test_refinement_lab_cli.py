"""Tests for refinement lab CLI helpers."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from refinement_lab import cli
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.refinement_lab import cli


class RefinementLabCliTests(unittest.TestCase):
    def test_list_commands(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["list-suites"]), 0)
            self.assertEqual(cli.main(["list-tracks"]), 0)

    def test_plan_command_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            with redirect_stdout(io.StringIO()):
                code = cli.main([
                    "plan",
                    "--suite",
                    "default-vase",
                    "--track",
                    "visual-hull-transform",
                    "--search",
                    "coordinate",
                    "--max-runs",
                    "2",
                    "--out",
                    str(out),
                ])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())

    def test_module_entrypoint_help_runs_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "blender_blocking.refinement_lab.cli",
                "--help",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("list-tracks", completed.stdout)


if __name__ == "__main__":
    unittest.main()
