from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.progress import progress_bar

from .contracts import BenchResult, SCHEMA_VERSION


def _format_duration(ms: float) -> str:
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f} s"
    return f"{ms:.2f} ms"


def _format_rate(value: float, unit: str) -> str:
    if value <= 0:
        return f"0 {unit}/s"
    scales = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for scale, suffix in scales:
        if value >= scale:
            return f"{value / scale:.2f} {suffix}{unit}/s"
    return f"{value:.2f} {unit}/s"


def _print_result(result: BenchResult) -> None:
    if result.status == "skip":
        print(f"SKIP: {result.name}: {result.skip_reason}")
        print()
        return

    total_ms = result.elapsed_s * 1000.0
    per_iter_ms = result.per_iter_ms
    print(f"OK: {result.name}")
    print(
        f"  total: {_format_duration(total_ms)}  "
        f"({result.iterations} iters, {_format_duration(per_iter_ms)} /iter)"
    )

    meta = dict(result.meta)
    throughput = meta.pop("throughput", None)
    throughput_unit = meta.pop("throughput_unit", None)
    throughput_label = meta.pop("throughput_label", "throughput")

    if throughput is not None and throughput_unit:
        print(
            f"  rate: {_format_rate(float(throughput), str(throughput_unit))} "
            f"({throughput_label})"
        )

    if meta:
        print("  meta:")
        for key in sorted(meta.keys()):
            print(f"    - {key}: {meta[key]}")
    print()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment_payload() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _result_payload(result: BenchResult) -> dict[str, object]:
    payload = asdict(result)
    metrics = {
        "iterations": result.iterations,
        "elapsed_s": result.elapsed_s,
        "per_iter_ms": result.per_iter_ms,
        "throughput": result.meta.get("throughput"),
        "status_ok": 1.0 if result.status == "ok" else 0.0,
    }
    payload["metrics"] = metrics
    return payload


def _results_payload(
    results: Sequence[BenchResult],
    *,
    budget_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    cases = sorted({result.case for result in results})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "environment": _environment_payload(),
        "cases": cases,
        "results": [_result_payload(result) for result in results],
    }
    if budget_report is not None:
        payload["budget_report"] = dict(budget_report)
    return payload


def _write_json(
    path: Path,
    results: Sequence[BenchResult],
    *,
    budget_report: Optional[Mapping[str, Any]] = None,
) -> dict[str, object]:
    payload = _results_payload(results, budget_report=budget_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
