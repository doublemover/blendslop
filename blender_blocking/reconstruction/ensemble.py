"""Candidate ensemble orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .candidate_scoring import rank_candidates, select_best
from .pareto import pareto_report_from_candidates
from .registry import get_backend
from .types import (
    CandidateBudget,
    CandidateMetrics,
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
    pareto_report: Mapping[str, Any] = field(default_factory=dict)

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
            "pareto_report": dict(self.pareto_report),
        }


class EnsembleRunner:
    """Run multiple backends and select a candidate using a scoring policy."""

    def __init__(self, *, selection_policy: str = "best_score", evidence_routing: bool = False, max_render_candidates: int = 3) -> None:
        self.selection_policy = selection_policy
        self.evidence_routing = evidence_routing
        self.max_render_candidates = max_render_candidates

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

    def run_requests(
        self,
        requests: Sequence[CandidateRequest],
        *,
        total_timeout_s: float | None = None,
        max_parallel_candidates: int = 1,
        process_budget=None,
    ) -> EnsembleRunResult:
        from .process_executor import (executor_scope, request_process_budget,
                                       candidate_payload, candidate_outcome)
        process_budget = request_process_budget(requests, process_budget=process_budget)
        if not requests:
            return self._result_from_candidates([], requests)
        manager = executor_scope(max_parallel_candidates, context=requests[0].context,
                                 process_budget=process_budget, require_submit_result=True)
        with manager as executor:
            if self.evidence_routing:
                from .measured_selection import routing_run
                results, selected, ledger = routing_run(
                    requests, total_timeout_s, self.max_render_candidates,
                    executor=executor, process_budget=process_budget)
                run = self._result_from_candidates(results, requests, selected_override=selected, use_override=True)
                return replace(run, pareto_report={**run.pareto_report, "routing_ledger": ledger})
            deadline = None if total_timeout_s is None else time.time() + float(total_timeout_s)
            ordinary = [r for r in requests if r.backend_name not in {"hybrid_loft_hull", "implicit_residual"}]
            ids = [executor.submit("candidate", candidate_payload(request, executor),
                timeout_s=request.budget.timeout_s, deadline=deadline) for request in ordinary]
            results = [candidate_outcome(request, executor.result(job_id)) for request, job_id in zip(ordinary, ids)]
            seeds = {r.backend_name: r for r in results if r.backend_name in {"profile_loft", "visual_hull_voxel"}}
            for request in requests:
                if request.backend_name == "hybrid_loft_hull":
                    attached = replace(request, config={**request.config, "seed_results": seeds}) if seeds else request
                    job_id = executor.submit("candidate", candidate_payload(attached, executor),
                        timeout_s=request.budget.timeout_s, deadline=deadline)
                    results.append(candidate_outcome(request, executor.result(job_id)))
                elif request.backend_name == "implicit_residual":
                    sources = {r.backend_name:r for r in results if r.succeeded and r.geometry is not None}
                    attached = replace(request, config={**request.config, "seed_results": sources}) if sources and not request.config.get("seed_results") else request
                    job_id = executor.submit("candidate", candidate_payload(attached, executor),
                        timeout_s=request.budget.timeout_s, deadline=deadline)
                    results.append(candidate_outcome(request, executor.result(job_id)))
            by_id = {r.candidate_id: r for r in results}
            results = [by_id[r.candidate_id] for r in requests]
            return self._result_from_candidates(results, requests)

    def _run_requests_parallel(
        self,
        requests: Sequence[CandidateRequest],
        *,
        max_parallel_candidates: int,
        total_timeout_s: float | None = None,
        process_budget=None,
    ) -> EnsembleRunResult:
        return self.run_requests(requests, total_timeout_s=total_timeout_s,
                                 max_parallel_candidates=max_parallel_candidates,
                                 process_budget=process_budget)

    def _result_from_candidates(
        self,
        results: Sequence[CandidateResult],
        requests: Sequence[CandidateRequest],
        *,
        selected_override: CandidateResult | None = None,
        use_override: bool = False,
    ) -> EnsembleRunResult:
        selected, ranked = select_best(results, policy=self.selection_policy)
        if use_override:
            selected = selected_override
        from blender_blocking.reconstruction.native_geometry import NativeOwnedGeometry
        if selected is not None and selected.geometry is not None and not isinstance(selected.geometry, NativeOwnedGeometry):
            context = next((r.context for r in requests if r.candidate_id == selected.candidate_id), None)
            if getattr(context, "blender_available", False):
                owner = NativeOwnedGeometry(selected.geometry, "Selected_" + selected.candidate_id,
                                            getattr(context, "cost_recorder", None))
                selected = replace(selected, geometry=owner, payload=owner.attach())
                results = [selected if r.candidate_id == selected.candidate_id else r for r in results]
        for result in results:
            if isinstance(result.geometry, NativeOwnedGeometry):
                if selected is not None and result.candidate_id == selected.candidate_id:
                    selected = replace(result, payload=result.geometry.attach())
                else:
                    result.geometry.release()
        results = [selected if selected is not None and r.candidate_id == selected.candidate_id else r for r in results]
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
            pareto_report=pareto_report_from_candidates(
                results,
                selected_id="" if selected is None else selected.candidate_id,
                policy=self.selection_policy,
            ).to_dict(),
        )

    def run(
        self,
        *,
        target: ReconstructionTarget,
        candidates: Sequence[CandidateConfig],
        artifact_root: str | Path | None = None,
        context: Any = None,
        budget: CandidateBudget = CandidateBudget(),
        total_timeout_s: float | None = None,
        max_parallel_candidates: int = 1,
        process_budget=None,
    ) -> EnsembleRunResult:
        requests = self.build_requests(
            target=target,
            candidates=candidates,
            artifact_root=artifact_root,
            context=context,
            budget=budget,
        )
        return self.run_requests(
            requests,
            total_timeout_s=total_timeout_s,
            max_parallel_candidates=max_parallel_candidates,
            process_budget=process_budget,
        )


def _run_candidate_request(request: CandidateRequest, *, project_diagnostics=True) -> CandidateResult:
    try:
        from .quality_config import quality_config
        request = replace(request, config=quality_config(request.config))
        backend = get_backend(request.backend_name)
        errors = backend.validate_config(request.config)
        if errors:
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=request.backend_name,
                status="failed",
                errors=tuple(errors),
            )
        from blender_blocking.evaluation.cost_model import timed_call
        recorder = getattr(request.context, "cost_recorder", None)
        started = time.perf_counter()
        from .option_receipts import reconstruct_with_receipt
        result = timed_call(recorder, "candidate_backend", reconstruct_with_receipt, backend, request)
        result = replace(result, metric_result=replace(result.metric_result, elapsed_s=time.perf_counter()-started))
        if request.artifact_root is not None and getattr(request.context,"blender_available",False):
            from .measured_selection import freeze_geometry
            result = freeze_geometry(result, request, project_diagnostics=project_diagnostics)
        return result
    except Exception as exc:
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=request.backend_name,
            status="failed",
            errors=(str(exc),),
        )


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
