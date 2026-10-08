from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

SCHEMA_VERSION = "benchmark_perf_results_v2"


@dataclass
class BenchResult:
    name: str
    iterations: int
    elapsed_s: float
    per_iter_ms: float
    case: str = "ad-hoc"
    status: str = "ok"
    meta: Dict[str, object] = field(default_factory=dict)
    skip_reason: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    benches: Tuple[str, ...]
    description: str
    overrides: Mapping[str, object] = field(default_factory=dict)
