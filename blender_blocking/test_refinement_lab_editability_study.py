"""Tests for Blender editability study packs."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from refinement_lab.contracts import ExperimentResult
from refinement_lab.editability_study import (
    build_editability_study_pack,
    score_review_row,
    summarize_review_rows,
    write_editability_study_pack,
)


class RefinementLabEditabilityStudyTests(unittest.TestCase):
    def test_study_pack_writes_json_markdown_and_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            result_json = run_root / "results" / "case" / "result.json"
            result_json.parent.mkdir(parents=True)
            result_json.write_text("{}", encoding="utf-8")
            result = _result(
                run_root=run_root,
                variant_id="variant-a",
                status="pass",
                result_json=result_json,
            )

            pack = build_editability_study_pack(
                [result],
                run_root=run_root,
                top_k=5,
            )
            paths = write_editability_study_pack(pack, run_root / "study")

            self.assertEqual(len(pack.items), 1)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["review_template"].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "editability_study_pack_v1")
            self.assertEqual(payload["items"][0]["variant_id"], "variant-a")
            template_rows = paths["review_template"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(template_rows), 1)
            template = json.loads(template_rows[0])
            self.assertEqual(template["schema_version"], "editability_review_v1")
            self.assertIn("semantic_parts", template["scores"])
            self.assertIn("Blender Editability Study", paths["markdown"].read_text(encoding="utf-8"))

    def test_study_pack_filters_failed_candidates_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            passing = _result(run_root=run_root, variant_id="pass", status="pass")
            failing = _result(run_root=run_root, variant_id="fail", status="fail")

            default_pack = build_editability_study_pack(
                [passing, failing],
                run_root=run_root,
            )
            inclusive_pack = build_editability_study_pack(
                [passing, failing],
                run_root=run_root,
                include_failed=True,
            )

            self.assertEqual([item.variant_id for item in default_pack.items], ["pass"])
            self.assertEqual({item.variant_id for item in inclusive_pack.items}, {"pass", "fail"})

    def test_review_row_scoring_and_summary(self) -> None:
        row = {
            "item_id": "item-a",
            "variant_id": "variant-a",
            "reviewer": "artist",
            "scores": {
                "semantic_parts": 5,
                "topology_cleanliness": 4,
                "mesh_density": 4,
                "primitive_controls": 5,
                "modifier_stack": 3,
                "scale_orientation": 5,
                "silhouette_fidelity": 4,
                "export_reliability": 5,
            },
        }

        score = score_review_row(row)
        summary = summarize_review_rows([row])

        self.assertGreater(score["score"], 0.8)
        self.assertEqual(score["label"], "sculptable")
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["top_item"]["variant_id"], "variant-a")


def _result(
    *,
    run_root: Path,
    variant_id: str,
    status: str = "pass",
    result_json: Path | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        run_id="run-a",
        case_id="case-a",
        variant_id=variant_id,
        mode="shape_program",
        status=status,
        exit_code=0 if status == "pass" else 1,
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:00:01Z",
        elapsed_s=1.0,
        command=("blender", "--background"),
        result_json=result_json,
        render_paths={"front": run_root / "renders" / f"{variant_id}_front.png"},
        reference_paths={"front": run_root / "refs" / "front.png"},
        backend_result={
            "selected": {
                "candidate_id": variant_id,
                "backend_name": "shape_program",
                "status": "success" if status == "pass" else "failed",
                "mesh_path": str(run_root / "mesh" / f"{variant_id}.obj"),
                "metric_result": {
                    "extras": {
                        "topology": {"topology_score": 0.91, "watertight": True},
                        "editability": {"primitive_score": 0.95},
                        "export_qa": {"target": "obj", "status": "pass"},
                    }
                },
            }
        },
        metrics={
            "average_iou": 0.9,
            "front_iou": 0.9,
            "topology_score": 0.91,
            "editability_score": 0.88,
        },
        artifacts={"mesh_obj": run_root / "mesh" / f"{variant_id}.obj"},
    )


if __name__ == "__main__":
    unittest.main()
