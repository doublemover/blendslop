"""Contracts for ambitious reconstruction research experiments.

Moonshots are intentionally not reported as reconstruction successes until a
backend produces artifacts and metrics.  These contracts make that status
explicit so speculative work can be scheduled, surfaced, and benchmarked later
without silent fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MoonshotRunner = Callable[["MoonshotRequest"], "MoonshotResult"]


@dataclass(frozen=True)
class MoonshotExperiment:
    experiment_id: str
    title: str
    subsystem: str
    hypothesis: str
    expected_wins: Mapping[str, str | float] = field(default_factory=dict)
    required_inputs: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    validation_metrics: tuple[str, ...] = ()
    papers: tuple[str, ...] = ()
    risk: str = "research"
    runner: MoonshotRunner | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "subsystem": self.subsystem,
            "hypothesis": self.hypothesis,
            "expected_wins": dict(self.expected_wins),
            "required_inputs": list(self.required_inputs),
            "optional_dependencies": list(self.optional_dependencies),
            "validation_metrics": list(self.validation_metrics),
            "papers": list(self.papers),
            "risk": self.risk,
        }


@dataclass(frozen=True)
class MoonshotRequest:
    experiment_id: str
    target: Any | None = None
    candidate: Any | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    artifact_root: str | None = None
    allow_research_execution: bool = False


@dataclass(frozen=True)
class MoonshotResult:
    experiment_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    degradation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "next_steps": list(self.next_steps),
            "degradation": dict(self.degradation),
        }


@dataclass(frozen=True)
class MoonshotRunBundle:
    """Shared result shape emitted by implemented moonshot sidecars."""

    experiment_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    degradation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "moonshot_run_bundle_v1",
            "experiment_id": self.experiment_id,
            "status": self.status,
            "metrics": _json_safe(dict(self.metrics)),
            "evidence": _json_safe(dict(self.evidence)),
            "artifacts": dict(self.artifacts),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "next_steps": list(self.next_steps),
            "degradation": _json_safe(dict(self.degradation)),
        }

    def to_result(self) -> MoonshotResult:
        degradation = dict(self.degradation)
        if self.evidence:
            degradation.setdefault("evidence", _json_safe(dict(self.evidence)))
        return MoonshotResult(
            experiment_id=self.experiment_id,
            status=self.status,
            metrics=dict(self.metrics),
            artifacts=dict(self.artifacts),
            warnings=tuple(self.warnings),
            errors=tuple(self.errors),
            next_steps=tuple(self.next_steps),
            degradation=degradation,
        )


def write_moonshot_artifact(
    request: MoonshotRequest,
    filename: str,
    payload: Mapping[str, Any],
) -> str | None:
    if not request.artifact_root:
        return None
    from blender_blocking.utils.path_safety import compact_path_segment
    base = Path(str(request.artifact_root))
    safe_filename = _safe_filename(filename)
    # Blender's bundled Windows Python can still enforce MAX_PATH. Leave room
    # for the filename and legacy mkdir limit without relocating the given root.
    base_length = len(str(base.resolve()))
    room = min(48, 245 - base_length - 1, 258 - base_length - len(safe_filename) - 2)
    segment = compact_path_segment(_safe_segment(request.experiment_id), max_length=room)
    root = base / segment
    root.mkdir(parents=True, exist_ok=True)
    path = root / safe_filename
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path.as_posix()


def bundle_result(
    request: MoonshotRequest,
    *,
    status: str,
    metrics: Mapping[str, float] | None = None,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    next_steps: Sequence[str] = (),
    degradation: Mapping[str, Any] | None = None,
    artifact_name: str = "bundle.json",
) -> MoonshotResult:
    bundle = MoonshotRunBundle(
        experiment_id=request.experiment_id,
        status=status,
        metrics=dict(metrics or {}),
        evidence=dict(evidence or {}),
        artifacts=dict(artifacts or {}),
        warnings=tuple(str(item) for item in warnings),
        errors=tuple(str(item) for item in errors),
        next_steps=tuple(str(item) for item in next_steps),
        degradation=dict(degradation or {}),
    )
    artifact_path = write_moonshot_artifact(request, artifact_name, bundle.to_dict())
    if artifact_path:
        bundle = MoonshotRunBundle(
            experiment_id=bundle.experiment_id,
            status=bundle.status,
            metrics=bundle.metrics,
            evidence=bundle.evidence,
            artifacts={**dict(bundle.artifacts), "bundle": artifact_path},
            warnings=bundle.warnings,
            errors=bundle.errors,
            next_steps=bundle.next_steps,
            degradation=bundle.degradation,
        )
    return bundle.to_result()


def research_candidate_result(
    request: MoonshotRequest,
    *,
    next_steps: Sequence[str],
    warnings: Sequence[str] = (),
) -> MoonshotResult:
    return MoonshotResult(
        experiment_id=request.experiment_id,
        status="research_candidate",
        warnings=tuple(warnings),
        next_steps=tuple(next_steps),
        degradation={
            "executed": False,
            "reason": "moonshot requires dedicated implementation and validation before success status",
        },
    )


def skipped_result(
    request: MoonshotRequest,
    *,
    reason: str,
    warnings: Sequence[str] = (),
    next_steps: Sequence[str] = (),
) -> MoonshotResult:
    return bundle_result(
        request,
        status="skipped",
        metrics={"ran": 0.0},
        evidence={"reason": reason},
        warnings=warnings,
        next_steps=next_steps,
        degradation={"executed": False, "reason": reason},
    )


def unsupported_result(
    request: MoonshotRequest,
    *,
    reason: str,
    warnings: Sequence[str] = (),
    next_steps: Sequence[str] = (),
) -> MoonshotResult:
    return bundle_result(
        request,
        status="unsupported",
        metrics={"ran": 0.0},
        evidence={"reason": reason},
        warnings=warnings,
        next_steps=next_steps,
        degradation={"executed": False, "reason": reason},
    )


def error_result(
    request: MoonshotRequest,
    *,
    error: str,
    warnings: Sequence[str] = (),
) -> MoonshotResult:
    return bundle_result(
        request,
        status="error",
        metrics={"ran": 0.0},
        evidence={"error": error},
        warnings=warnings,
        errors=(error,),
        degradation={"executed": False, "reason": "sidecar error"},
    )


def _safe_segment(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return text.strip("_") or "moonshot"


def _safe_filename(value: str) -> str:
    text = value.replace("\\", "/").split("/")[-1]
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return text.strip("._") or "artifact.json"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
