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
    evaluation_bundles: tuple[Any, ...] = ()
    autopsy_packs: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scores": list(self.scores),
            "policy": self.policy,
            "evaluation_bundles": [
                bundle.to_dict() if hasattr(bundle, "to_dict") else bundle
                for bundle in self.evaluation_bundles
            ],
            "autopsy_packs": [dict(pack) for pack in self.autopsy_packs],
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
        bundles, autopsy_packs = _evaluation_outputs(
            results=results,
            requests=requests,
            ranked=ranked,
            selected=selected,
            policy=self.selection_policy,
        )
        return EnsembleRunResult(
            selected=selected,
            candidates=tuple(results),
            scores=tuple(score.to_dict() for _, score in ranked),
            policy=self.selection_policy,
            evaluation_bundles=bundles,
            autopsy_packs=autopsy_packs,
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


def _evaluation_outputs(
    *,
    results: Sequence[CandidateResult],
    requests: Sequence[CandidateRequest],
    ranked: Sequence[tuple[CandidateResult, Any]],
    selected: CandidateResult | None,
    policy: str,
) -> tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]:
    try:
        from blender_blocking.evaluation.autopsy import autopsy_pack_from_bundle
        from blender_blocking.evaluation.selection import (
            attach_selection,
            pareto_dominators,
        )
    except Exception:  # pragma: no cover - legacy script import path
        from evaluation.autopsy import autopsy_pack_from_bundle  # type: ignore
        from evaluation.selection import attach_selection, pareto_dominators  # type: ignore

    request_by_id = {request.candidate_id: request for request in requests}
    rank_by_id = {
        result.candidate_id: (index, score)
        for index, (result, score) in enumerate(ranked, start=1)
    }
    selected_id = "" if selected is None else selected.candidate_id
    raw_bundles = []
    for result in results:
        request = request_by_id.get(result.candidate_id)
        target = request.target if request is not None else None
        context = request.context if request is not None else None
        suite = str(getattr(context, "requested_mode", "") or "")
        raw_bundles.append(
            result.to_evaluation_bundle(
                target=target,
                suite=suite,
                run_id=result.candidate_id,
            )
        )
    dominators = pareto_dominators(tuple(raw_bundles))
    bundles = []
    autopsy_packs = []
    for bundle in raw_bundles:
        rank, score = rank_by_id.get(bundle.candidate_id, (0, None))
        bundle = attach_selection(
            bundle,
            policy=policy,
            rank=rank,
            selected=bundle.candidate_id == selected_id,
            score=score,
            dominated_by=dominators.get(bundle.candidate_id, ()),
        )
        bundles.append(bundle)
        autopsy_packs.append(autopsy_pack_from_bundle(bundle).to_dict())
    return tuple(bundles), tuple(autopsy_packs)
