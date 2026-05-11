"""Path helpers for Blender-safe generated artifact names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


def json_safe_for_hash(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_safe_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_for_hash(item) for item in value]
    if isinstance(value, set):
        return [json_safe_for_hash(item) for item in sorted(value, key=str)]
    return value


def stable_path_hash(payload: Any, *, length: int = 10) -> str:
    encoded = json.dumps(
        json_safe_for_hash(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def safe_path_slug(value: object, *, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or fallback


def compact_path_segment(
    value: object,
    *,
    max_length: int = 48,
    hash_length: int = 10,
    fallback: str = "item",
) -> str:
    """Return a readable deterministic path segment with a hard length cap."""
    max_length = max(16, int(max_length))
    hash_length = max(6, min(int(hash_length), max_length - 2))
    slug = safe_path_slug(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = stable_path_hash({"value": value}, length=hash_length)
    stem_length = max(1, max_length - hash_length - 1)
    stem = slug[:stem_length].rstrip("_-") or fallback
    return f"{stem}-{digest}"


def compact_path_stem(
    *parts: object,
    max_length: int = 64,
    hash_length: int = 10,
    fallback: str = "item",
    separator: str = "-",
) -> str:
    raw = separator.join(
        safe_path_slug(part, fallback=fallback) for part in parts if str(part)
    )
    return compact_path_segment(
        raw or fallback,
        max_length=max_length,
        hash_length=hash_length,
        fallback=fallback,
    )
