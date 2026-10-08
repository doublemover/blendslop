"""Tests for shared JSON serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from utils.json_io import append_jsonl, json_safe, stable_hash, write_json, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.utils.json_io import (
        append_jsonl,
        json_safe,
        stable_hash,
        write_json,
        write_jsonl,
    )


class JsonIoTests(unittest.TestCase):
    def test_json_safe_normalizes_common_non_json_types(self) -> None:
        class Color(Enum):
            RED = "red"

        @dataclass(frozen=True)
        class Payload:
            path: Path
            color: Color
            values: set[int]
            scalar: object
            bad_float: float

        payload = Payload(
            path=Path("temp") / "asset.png",
            color=Color.RED,
            values={3, 1, 2},
            scalar=np.float32(1.25),
            bad_float=math.inf,
        )

        safe = json_safe(payload)

        self.assertEqual(
            safe,
            {
                "path": "temp/asset.png",
                "color": "red",
                "values": [1, 2, 3],
                "scalar": 1.25,
                "bad_float": None,
            },
        )

    def test_stable_hash_ignores_mapping_order(self) -> None:
        left = {"b": [2, 3], "a": {"x": 1}}
        right = {"a": {"x": 1}, "b": [2, 3]}

        self.assertEqual(stable_hash(left), stable_hash(right))
        self.assertEqual(len(stable_hash(left, length=16)), 16)

    def test_write_json_and_jsonl_create_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "nested" / "payload.json"
            rows_path = root / "nested" / "rows.jsonl"

            write_json(json_path, {"path": Path("a") / "b", "nan": math.nan})
            write_jsonl(rows_path, [{"id": 1}], append=False)
            append_jsonl(rows_path, {"id": 2})

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(payload, {"nan": None, "path": "a/b"})
            self.assertEqual(rows, [{"id": 1}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
