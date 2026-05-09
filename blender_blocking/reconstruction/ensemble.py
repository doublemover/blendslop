"""Candidate ensemble orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_scoring import rank_candidates, select_best
from .registry import get_backend
from .types import (
    CandidateBudget,
    CandidateRequest,
    CandidateResult,
    ReconstructionTarget,
)


@dataclass(frozen=True)
class CandidateConfig:
    backend_name: str
    enabled: bool = True
    config: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = ""


@dataclass(frozen=True)
class EnsembleRunResult:
    selected: CandidateResult | None
    candidates: tuple[CandidateResult, ...]
    scores: tuple[Mapping[str, Any], ...]
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scores": list(self.scores),
            "policy": self.policy,
        }


class EnsembleRunner:
    """Run multiple backends and select a candidate using a scoring policy."""

    def __init__(self, *, selection_policy: str = "best_score") -> None:
        self.selection_policy = selection_policy

    def build_requests(
        self,
        *,
        target: ReconstructionTarget,
        candidates: Sequence[CandidateConfig],
        artifact_root: str | Path | None = None,
        context: Any = None,
        budget: CandidateBudget = CandidateBudget(),
    ) -> list[CandidateRequest]:
        requests: list[CandidateRequest] = []
        root = Path(artifact_root) if artifact_root is not None else None
        for index, candidate in enumerate(candidates):
            if not candidate.enabled:
                continue
            candidate_id = candidate.candidate_id or f"{candidate.backend_name}_{index:02d}"
            requests.append(
                CandidateRequest(
                    candidate_id=candidate_id,
                    backend_name=candidate.backend_name,
                    target=target,
                    config=dict(candidate.config),
                    budget=budget,
                    artifact_root=root,
                    context=context,
                )
            )
        return requests

    def run_requests(self, requests: Sequence[CandidateRequest]) -> EnsembleRunResult:
        results: list[CandidateResult] = []
        for request in requests:
            try:
                backend = get_backend(request.backend_name)
                errors = backend.validate_config(request.config)
                if errors:
                    results.append(
                        CandidateResult(
                            candidate_id=request.candidate_id,
                            backend_name=request.backend_name,
                            status="failed",
                            errors=tuple(errors),
                        )
                    )
                    continue
                results.append(backend.reconstruct(request))
            except Exception as exc:
                results.append(
                    CandidateResult(
                        candidate_id=request.candidate_id,
                        backend_name=request.backend_name,
                        status="failed",
                        errors=(str(exc),),
                    )
                )
        selected, ranked = select_best(results, policy=self.selection_policy)
        return EnsembleRunResult(
            selected=selected,
            candidates=tuple(results),
            scores=tuple(score.to_dict() for _, score in ranked),
            policy=self.selection_policy,
        )

    def run(
        self,
        *,
        target: ReconstructionTarget,
        candidates: Sequence[CandidateConfig],
        artifact_root: str | Path | None = None,
        context: Any = None,
        budget: CandidateBudget = CandidateBudget(),
    ) -> EnsembleRunResult:
        requests = self.build_requests(
            target=target,
            candidates=candidates,
            artifact_root=artifact_root,
            context=context,
            budget=budget,
        )
        return self.run_requests(requests)
