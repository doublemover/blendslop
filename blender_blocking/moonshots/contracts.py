"""Contracts for ambitious reconstruction research experiments.

Moonshots are intentionally not reported as reconstruction successes until a
backend produces artifacts and metrics.  These contracts make that status
explicit so speculative work can be scheduled, surfaced, and benchmarked later
without silent fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
