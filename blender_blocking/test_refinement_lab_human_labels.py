"""Tests for human labels."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from refinement_lab.human_labels import HumanLabel, append_label, labels_by_variant, load_labels


class RefinementLabHumanLabelTests(unittest.TestCase):
    def test_label_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            append_label(path, HumanLabel(run_id="r", case_id="c", variant_id="v", label="sculptable", score=5))
            labels = load_labels(path)
        self.assertEqual(len(labels), 1)
        self.assertIn("v", labels_by_variant(labels))

    def test_invalid_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HumanLabel(run_id="r", case_id="c", variant_id="v", label="bad", score=5)


if __name__ == "__main__":
    unittest.main()
