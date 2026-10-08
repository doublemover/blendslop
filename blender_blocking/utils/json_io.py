"""Shared JSON serialization, hashing, and JSONL helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def json_safe(value: Any) -> Any:
    """Convert common Python values into deterministic JSON-safe structures."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return value


def stable_hash(payload: Any, *, length: int = 12) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return read_json(path)
    except Exception:
        return default


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), indent=indent, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output


def write_jsonl(
    path: str | Path,
    rows: Iterable[Any],
    *,
    append: bool = True,
) -> Path:
    output = Path(path)
    serialized = [
        json.dumps(json_safe(row), sort_keys=True, default=str)
        for row in rows
    ]
    if not serialized:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        handle.write("\n".join(serialized) + "\n")
    return output


def append_jsonl(path: str | Path, row: Any) -> Path:
    return write_jsonl(path, (row,), append=True)
