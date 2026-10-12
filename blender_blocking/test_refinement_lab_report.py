"""Tests for refinement HTML report generation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from refinement_lab.artifact_report import generate_report
from refinement_lab.contracts import ExperimentResult


class RefinementLabReportTests(unittest.TestCase):
    def test_minimal_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = ExperimentResult(
                run_id="r",
                case_id="case-smoke",
                variant_id="variant-smoke",
                mode="profile_loft",
                status="pass",
                exit_code=0,
                started_utc="s",
                finished_utc="f",
                elapsed_s=1.0,
                metrics={"average_iou": 0.9, "front_iou": 0.9, "side_iou": 0.8, "top_iou": 0.7},
            )
            path = generate_report(run_root=root, results=[result])
            text = path.read_text(encoding="utf-8")
        self.assertIn("<td>case-smoke</td>", text)
        self.assertIn("<td>variant-smoke</td>", text)


if __name__ == "__main__":
    unittest.main()
