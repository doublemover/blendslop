"""Stage-level cost accounting for reconstruction and evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from .schemas import json_safe


@dataclass(frozen=True)
class StageCost:
    stage: str
    status: str
    wall_ms: float
    peak_memory_mb: float | None = None
    work_units: Mapping[str, float] = field(default_factory=dict)
    artifact_bytes: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "status": self.status,
            "wall_ms": self.wall_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "work_units": json_safe(self.work_units),
            "artifact_bytes": self.artifact_bytes,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CostReport:
    total_wall_ms: float
    peak_memory_mb: float | None = None
    stages: tuple[StageCost, ...] = ()
    cache: Mapping[str, float] = field(default_factory=dict)
    throughput: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_wall_ms": self.total_wall_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "stages": [stage.to_dict() for stage in self.stages],
            "cache": json_safe(self.cache),
            "throughput": json_safe(self.throughput),
        }


class StageTimer:
    """Small context manager for future timed stages."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.start = 0.0
        self.wall_ms = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = perf_counter()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.wall_ms = (perf_counter() - self.start) * 1000.0

    def cost(self, *, status: str = "pass", notes: tuple[str, ...] = ()) -> StageCost:
        return StageCost(stage=self.stage, status=status, wall_ms=float(self.wall_ms), notes=notes)


def cost_report_from_candidate(result: Any) -> CostReport:
    metrics = getattr(result, "metric_result", None)
    elapsed_s = float(getattr(metrics, "elapsed_s", 0.0) or 0.0)
    extras = getattr(metrics, "extras", {}) or {}
    optimization = extras.get("optimization") if isinstance(extras, Mapping) else None
    stages = []
    if elapsed_s > 0.0:
        stages.append(StageCost("backend_reconstruct", "pass", elapsed_s * 1000.0))
    if isinstance(optimization, Mapping):
        opt_elapsed = float(optimization.get("elapsed_s", 0.0) or 0.0)
        stages.append(
            StageCost(
                "primitive_fit",
                "pass",
                opt_elapsed * 1000.0,
                work_units={
                    "objective_evaluations": float(optimization.get("objective_evaluations", 0.0) or 0.0)
                },
            )
        )
    total = sum(stage.wall_ms for stage in stages)
    return CostReport(total_wall_ms=total, stages=tuple(stages))

