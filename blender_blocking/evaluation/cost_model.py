"""Stage-level cost accounting for reconstruction and evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping
import tracemalloc

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
class CacheCost:
    namespace: str
    hits: int = 0
    misses: int = 0
    entries: int = 0
    bytes_stored: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entries,
            "bytes_stored": self.bytes_stored,
            "hit_rate": self.hit_rate,
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
    """Context manager for stage-level wall-time and memory accounting."""

    def __init__(
        self,
        stage: str,
        *,
        recorder: "CostRecorder | None" = None,
        status: str = "pass",
        work_units: Mapping[str, float] | None = None,
        artifact_bytes: int = 0,
        notes: tuple[str, ...] = (),
        track_memory: bool = True,
    ) -> None:
        self.stage = stage
        self.recorder = recorder
        self.status = status
        self.work_units = dict(work_units or {})
        self.artifact_bytes = int(artifact_bytes)
        self.notes = tuple(notes)
        self.track_memory = bool(track_memory)
        self.start = 0.0
        self.wall_ms = 0.0
        self.peak_memory_mb: float | None = None
        self._started_tracemalloc = False

    def __enter__(self) -> "StageTimer":
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, _tb: object) -> None:
        self.wall_ms = (perf_counter() - self.start) * 1000.0
        if self.track_memory and tracemalloc.is_tracing():
            _current, peak = tracemalloc.get_traced_memory()
            self.peak_memory_mb = float(peak / (1024.0 * 1024.0))
            if self._started_tracemalloc:
                tracemalloc.stop()
        if exc_type is not None and self.status == "pass":
            self.status = "error"
            self.notes = self.notes + (str(exc) if exc else str(exc_type),)
        if self.recorder is not None:
            self.recorder.add_stage(self.cost())

    def cost(
        self,
        *,
        status: str | None = None,
        notes: tuple[str, ...] | None = None,
    ) -> StageCost:
        return StageCost(
            stage=self.stage,
            status=status or self.status,
            wall_ms=float(self.wall_ms),
            peak_memory_mb=self.peak_memory_mb,
            work_units=dict(self.work_units),
            artifact_bytes=self.artifact_bytes,
            notes=self.notes if notes is None else tuple(notes),
        )


class CostRecorder:
    """Mutable run-cost accumulator for orchestration code."""

    def __init__(self, *, track_memory: bool = True) -> None:
        self.track_memory = bool(track_memory)
        self.start = perf_counter()
        self.stages: list[StageCost] = []
        self.cache: dict[str, CacheCost] = {}

    def stage(
        self,
        name: str,
        *,
        status: str = "pass",
        work_units: Mapping[str, float] | None = None,
        artifact_bytes: int = 0,
        notes: tuple[str, ...] = (),
    ) -> StageTimer:
        return StageTimer(
            name,
            recorder=self,
            status=status,
            work_units=work_units,
            artifact_bytes=artifact_bytes,
            notes=notes,
            track_memory=self.track_memory,
        )

    def add_stage(self, cost: StageCost) -> None:
        self.stages.append(cost)

    def record_cache(
        self,
        namespace: str,
        *,
        hits: int = 0,
        misses: int = 0,
        entries: int = 0,
        bytes_stored: int = 0,
    ) -> None:
        key = str(namespace)
        current = self.cache.get(key, CacheCost(key))
        self.cache[key] = CacheCost(
            namespace=key,
            hits=current.hits + int(hits),
            misses=current.misses + int(misses),
            entries=max(current.entries, int(entries)),
            bytes_stored=max(current.bytes_stored, int(bytes_stored)),
        )

    def cache_hit(self, namespace: str, *, entries: int = 0, bytes_stored: int = 0) -> None:
        self.record_cache(
            namespace,
            hits=1,
            entries=entries,
            bytes_stored=bytes_stored,
        )

    def cache_miss(self, namespace: str, *, entries: int = 0, bytes_stored: int = 0) -> None:
        self.record_cache(
            namespace,
            misses=1,
            entries=entries,
            bytes_stored=bytes_stored,
        )

    def report(
        self,
        *,
        throughput: Mapping[str, float] | None = None,
        include_unaccounted_time: bool = True,
    ) -> CostReport:
        stages = list(self.stages)
        total_stage_ms = sum(stage.wall_ms for stage in stages)
        elapsed_ms = (perf_counter() - self.start) * 1000.0
        if include_unaccounted_time and elapsed_ms > total_stage_ms:
            unaccounted = elapsed_ms - total_stage_ms
            if unaccounted > 0.001:
                stages.append(
                    StageCost(
                        stage="unaccounted",
                        status="pass",
                        wall_ms=float(unaccounted),
                        notes=("time outside named stages",),
                    )
                )
        peak_values = [
            stage.peak_memory_mb
            for stage in stages
            if stage.peak_memory_mb is not None
        ]
        cache_payload = _flatten_cache(tuple(self.cache.values()))
        return CostReport(
            total_wall_ms=sum(stage.wall_ms for stage in stages),
            peak_memory_mb=max(peak_values) if peak_values else None,
            stages=tuple(stages),
            cache=cache_payload,
            throughput=dict(throughput or {}),
        )


def cost_report_from_candidate(result: Any) -> CostReport:
    metrics = getattr(result, "metric_result", None)
    elapsed_s = float(getattr(metrics, "elapsed_s", 0.0) or 0.0)
    extras = getattr(metrics, "extras", {}) or {}
    if isinstance(extras, Mapping):
        explicit_cost = extras.get("cost_report", extras.get("cost"))
        if isinstance(explicit_cost, Mapping):
            return cost_report_from_mapping(explicit_cost)
    optimization = extras.get("optimization") if isinstance(extras, Mapping) else None
    objective = extras.get("objective") if isinstance(extras, Mapping) else None
    visual_hull = extras.get("visual_hull_stats") if isinstance(extras, Mapping) else None
    mesh_extraction = extras.get("mesh_extraction") if isinstance(extras, Mapping) else None
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
    if isinstance(objective, Mapping) and not isinstance(optimization, Mapping):
        opt_elapsed = float(
            extras.get("optimizer_elapsed_s", objective.get("elapsed_s", 0.0)) or 0.0
        )
        stages.append(
            StageCost(
                "primitive_objective",
                "pass" if bool(objective.get("improved", True)) else "warn",
                opt_elapsed * 1000.0,
                work_units={
                    "objective_evaluations": float(objective.get("objective_evaluations", 0.0) or 0.0),
                    "history_length": float(objective.get("history_length", 0.0) or 0.0),
                },
                notes=(str(objective.get("termination_reason", "")),)
                if objective.get("termination_reason")
                else (),
            )
        )
    if isinstance(mesh_extraction, Mapping):
        elapsed = float(mesh_extraction.get("elapsed_s", 0.0) or 0.0)
        stages.append(
            StageCost(
                "mesh_extraction",
                str(mesh_extraction.get("status", "pass")),
                elapsed * 1000.0,
                work_units={
                    "vertices": float(
                        mesh_extraction.get("vertex_count", mesh_extraction.get("vertices", 0.0)) or 0.0
                    ),
                    "faces": float(
                        mesh_extraction.get("face_count", mesh_extraction.get("faces", 0.0)) or 0.0
                    ),
                },
            )
        )
    total = sum(stage.wall_ms for stage in stages)
    throughput = {}
    if isinstance(visual_hull, Mapping):
        elapsed_ms = max(total, 1e-9)
        active = float(visual_hull.get("active_voxels", 0.0) or 0.0)
        total_voxels = float(visual_hull.get("total_voxels", 0.0) or 0.0)
        if active:
            throughput["active_voxels_per_ms"] = active / elapsed_ms
        if total_voxels:
            throughput["voxels_per_ms"] = total_voxels / elapsed_ms
    return CostReport(total_wall_ms=total, stages=tuple(stages), throughput=throughput)


def cost_report_from_mapping(payload: Mapping[str, Any]) -> CostReport:
    stages_payload = payload.get("stages", ()) or ()
    cache_payload = payload.get("cache", {}) or {}
    stages = tuple(
        stage_cost_from_mapping(stage)
        for stage in stages_payload
        if isinstance(stage, Mapping)
    )
    total = _optional_float(payload.get("total_wall_ms"))
    if total is None:
        total = sum(stage.wall_ms for stage in stages)
    return CostReport(
        total_wall_ms=float(total),
        peak_memory_mb=_optional_float(payload.get("peak_memory_mb")),
        stages=stages,
        cache=dict(cache_payload) if isinstance(cache_payload, Mapping) else {},
        throughput=dict(payload.get("throughput", {}) or {})
        if isinstance(payload.get("throughput", {}), Mapping)
        else {},
    )


def stage_cost_from_mapping(payload: Mapping[str, Any]) -> StageCost:
    return StageCost(
        stage=str(payload.get("stage", "")),
        status=str(payload.get("status", "pass")),
        wall_ms=float(payload.get("wall_ms", 0.0) or 0.0),
        peak_memory_mb=_optional_float(payload.get("peak_memory_mb")),
        work_units=dict(payload.get("work_units", {}) or {})
        if isinstance(payload.get("work_units", {}), Mapping)
        else {},
        artifact_bytes=int(payload.get("artifact_bytes", 0) or 0),
        notes=tuple(str(item) for item in payload.get("notes", ()) or ()),
    )


def _flatten_cache(cache_items: tuple[CacheCost, ...]) -> dict[str, float]:
    payload: dict[str, float] = {}
    total_hits = 0
    total_misses = 0
    for item in cache_items:
        prefix = f"{item.namespace}."
        payload[prefix + "hits"] = float(item.hits)
        payload[prefix + "misses"] = float(item.misses)
        payload[prefix + "entries"] = float(item.entries)
        payload[prefix + "bytes_stored"] = float(item.bytes_stored)
        payload[prefix + "hit_rate"] = item.hit_rate
        total_hits += item.hits
        total_misses += item.misses
    if cache_items:
        total = total_hits + total_misses
        payload["hits"] = float(total_hits)
        payload["misses"] = float(total_misses)
        payload["hit_rate"] = float(total_hits / total) if total else 0.0
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
