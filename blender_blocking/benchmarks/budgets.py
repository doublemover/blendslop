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

from .contracts import BenchResult
from .results import _write_json


def _load_budget_report(
    *,
    current_payload: Mapping[str, Any],
    current_path: Optional[Path],
    budget_path: Optional[str],
    baseline_path: Optional[str],
    report_path: Optional[str],
) -> Optional[dict[str, object]]:
    if not budget_path:
        return None
    try:
        from scripts.quality_budget import evaluate_budget_payloads, write_report
    except Exception as exc:
        return {
            "passed": False,
            "error": f"failed to import scripts.quality_budget: {exc}",
        }

    baseline_payload = None
    if baseline_path:
        baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    report = evaluate_budget_payloads(
        current_payload,
        json.loads(Path(budget_path).read_text(encoding="utf-8")),
        baseline_payload=baseline_payload,
        artifact_path=str(current_path) if current_path else None,
        baseline_path=baseline_path,
    )
    if report_path:
        write_report(Path(report_path), report)
    return report
