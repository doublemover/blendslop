"""Tests for refinement result indexing."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from refinement_lab.contracts import ExperimentResult
from refinement_lab.result_index import ResultIndex, load_index


class RefinementLabIndexTests(unittest.TestCase):
    def test_append_load_and_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = ResultIndex(root)
            result = ExperimentResult(
                run_id="r",
                case_id="c",
                variant_id="v",
                mode="profile_loft",
                status="pass",
                exit_code=0,
                started_utc="s",
                finished_utc="f",
                elapsed_s=1.0,
                backend_result={"status": "success"},
                metrics={"average_iou": 0.9, "front_iou": 0.9, "side_iou": 0.8, "top_iou": 0.7},
            )
            index.append(result)
            loaded = index.load()
            self.assertEqual(len(loaded), 1)
            index.write_leaderboards(loaded)
            self.assertTrue((root / "leaderboard.json").exists())
            self.assertTrue((root / "leaderboard.md").exists())
            leaderboard = json.loads((root / "leaderboard.json").read_text(encoding="utf-8"))
            self.assertEqual(leaderboard["rows"][0]["promotion_tier"], "promotable")
            self.assertTrue(leaderboard["fastest_acceptable"]["promotable"])

    def test_malformed_trailing_line_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            path.write_text("{bad\n", encoding="utf-8")
            results, malformed = load_index(path)
        self.assertEqual(results, [])
        self.assertEqual(malformed, 1)

    def test_leaderboard_reports_duplicate_quality_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = ResultIndex(root, objective="quality_win")
            base_metrics = {
                "average_iou": 0.92,
                "front_iou": 0.92,
                "side_iou": 0.9,
                "top_iou": 0.88,
                "topology_score": 1.0,
                "editability_score": 0.8,
            }
            results = [
                ExperimentResult(
                    run_id="r",
                    case_id="c",
                    variant_id="v-a",
                    mode="profile_loft",
                    status="pass",
                    exit_code=0,
                    started_utc="s",
                    finished_utc="f",
                    elapsed_s=1.0,
                    backend_result={"status": "success"},
                    metrics=base_metrics,
                ),
                ExperimentResult(
                    run_id="r",
                    case_id="c",
                    variant_id="v-b",
                    mode="profile_loft",
                    status="pass",
                    exit_code=0,
                    started_utc="s",
                    finished_utc="f",
                    elapsed_s=2.0,
                    backend_result={"status": "success"},
                    metrics=base_metrics,
                ),
            ]

            index.write_leaderboards(results)
            leaderboard = json.loads(
                (root / "leaderboard.json").read_text(encoding="utf-8")
            )

            self.assertEqual(leaderboard["duplicate_quality_group_count"], 1)
            self.assertEqual(leaderboard["duplicate_quality_groups"][0]["count"], 2)
            self.assertTrue(all(row["duplicate_quality"] for row in leaderboard["rows"]))
            self.assertEqual(
                {row["duplicate_quality_count"] for row in leaderboard["rows"]},
                {2},
            )
            self.assertIn(
                "Duplicate Quality Groups",
                (root / "leaderboard.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
