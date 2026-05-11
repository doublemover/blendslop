"""Closed-loop adaptive refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .adaptive import RefinementProposal, merge_proposals, proposals_from_result_payload
from .contracts import ExperimentPlan, ExperimentResult, ExperimentVariant, json_safe, safe_slug, stable_hash
from .matrix import build_experiment_plan
from .parameter_search import promotion_decision, rank_results
from .runner import RunOptions, runner_for_plan

try:
    from blender_blocking.config import BlockingConfig
    from blender_blocking.utils.json_io import write_json as _write_json
except ImportError:  # pragma: no cover
    from config import BlockingConfig
    from utils.json_io import write_json as _write_json


class _RunnerLike(Protocol):
    def run(self) -> tuple[bool, list[ExperimentResult]]:
        ...


RunnerFactory = Callable[[ExperimentPlan, RunOptions, BlockingConfig | None], _RunnerLike]


@dataclass(frozen=True)
class AdaptiveLoopOptions:
    generations: int = 3
    parent_top_k: int = 3
    children_per_parent: int = 4
    stop_when_no_children: bool = True
    allow_diagnostic_children: bool = False
    run_options: RunOptions = field(default_factory=RunOptions)

    def __post_init__(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.parent_top_k < 1:
            raise ValueError("parent_top_k must be >= 1")
        if self.children_per_parent < 1:
            raise ValueError("children_per_parent must be >= 1")


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    run_root: Path
    plan_path: Path
    ok: bool
    result_count: int
    selected_parent_ids: tuple[str, ...]
    proposal_count: int
    child_variant_count: int
    proposal_path: Path
    variant_path: Path
    failure_code: str = ""
    error: str = ""
    error_path: Path | None = None
    parent_health: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "generation": self.generation,
            "run_root": self.run_root.as_posix(),
            "plan_path": self.plan_path.as_posix(),
            "ok": self.ok,
            "result_count": self.result_count,
            "selected_parent_ids": list(self.selected_parent_ids),
            "proposal_count": self.proposal_count,
            "child_variant_count": self.child_variant_count,
            "proposal_path": self.proposal_path.as_posix(),
            "variant_path": self.variant_path.as_posix(),
            "parent_health": json_safe(dict(self.parent_health)),
        }
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        if self.error:
            payload["error"] = self.error
        if self.error_path is not None:
            payload["error_path"] = self.error_path.as_posix()
        return payload


@dataclass(frozen=True)
class AdaptiveLoopSummary:
    loop_root: Path
    summary_path: Path
    generations: tuple[GenerationRecord, ...]
    final_child_variants: tuple[ExperimentVariant, ...]
    stopped_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "refinement_adaptive_loop_v1",
            "loop_root": self.loop_root.as_posix(),
            "stopped_reason": self.stopped_reason,
            "generation_count": len(self.generations),
            "generations": [record.to_dict() for record in self.generations],
            "final_child_variant_count": len(self.final_child_variants),
            "final_child_variants": [
                variant.to_dict() for variant in self.final_child_variants
            ],
        }


def run_adaptive_loop(
    *,
    suite: str,
    track: str,
    search: str,
    objective: str,
    output_root: Path,
    seed: int = 0,
    case_count: int | None = None,
    max_runs: int | None = None,
    top_k: int = 10,
    external_variants: Sequence[ExperimentVariant | Mapping[str, Any]] = (),
    external_variant_mode: str = "append",
    options: AdaptiveLoopOptions = AdaptiveLoopOptions(),
    base_config: BlockingConfig | None = None,
    runner_factory: RunnerFactory | None = None,
) -> AdaptiveLoopSummary:
    """Run successive adaptive generations and write a loop summary."""

    root = Path(output_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    runner_factory = runner_factory or _default_runner_factory
    records: list[GenerationRecord] = []
    current_variants: tuple[ExperimentVariant | Mapping[str, Any], ...] = tuple(
        external_variants
    )
    current_variant_mode = external_variant_mode
    stopped_reason = "max_generations"

    for generation in range(options.generations):
        generation_root = root / f"g{generation:02d}"
        plan = build_experiment_plan(
            suite=suite,
            track=track,
            search=search,
            objective=objective,
            output_root=generation_root,
            seed=seed + generation,
            case_count=case_count,
            max_runs=max_runs,
            top_k=top_k,
            external_variants=current_variants,
            external_variant_mode=current_variant_mode,
        )
        plan = _with_loop_metadata(
            plan,
            loop_root=root,
            generation=generation,
            parent_variant_count=len(current_variants),
        )
        plan.write(generation_root / "plan.json")
        failure_code = ""
        error = ""
        error_path: Path | None = None
        try:
            ok, results = runner_factory(plan, options.run_options, base_config).run()
        except Exception as exc:
            ok = False
            results = []
            failure_code = "runner_exception"
            error = str(exc)
            error_path = generation_root / "adaptive-loop-error.json"
            _write_json(
                error_path,
                {
                    "schema_version": "refinement_adaptive_loop_error_v1",
                    "generation": generation,
                    "failure_code": failure_code,
                    "error": error,
                    "exception_type": type(exc).__name__,
                },
            )
        selected = _select_parent_results(
            results,
            objective=objective,
            parent_top_k=options.parent_top_k,
            allow_diagnostics=options.allow_diagnostic_children,
        )
        parent_health = _parent_health_summary(
            results,
            objective=objective,
            parent_top_k=options.parent_top_k,
        )
        proposals, child_variants = _child_variants_from_results(
            selected,
            generation=generation,
            children_per_parent=options.children_per_parent,
        )
        proposal_path = generation_root / "adaptive-loop-proposals.json"
        variant_path = generation_root / "adaptive-loop-variants.json"
        _write_generation_adaptive_outputs(
            proposal_path=proposal_path,
            variant_path=variant_path,
            generation=generation,
            selected=selected,
            proposals=proposals,
            child_variants=child_variants,
            parent_health=parent_health,
        )
        records.append(
            GenerationRecord(
                generation=generation,
                run_root=generation_root,
                plan_path=generation_root / "plan.json",
                ok=ok,
                result_count=len(results),
                selected_parent_ids=tuple(result.variant_id for result in selected),
                proposal_count=len(proposals),
                child_variant_count=len(child_variants),
                proposal_path=proposal_path,
                variant_path=variant_path,
                failure_code=failure_code,
                error=error,
                error_path=error_path,
                parent_health=parent_health,
            )
        )
        if failure_code:
            stopped_reason = failure_code
            current_variants = ()
            break
        if not selected and options.stop_when_no_children:
            stopped_reason = "no_promotable_parents"
            current_variants = ()
            break
        if not child_variants and options.stop_when_no_children:
            stopped_reason = "no_child_variants"
            current_variants = ()
            break
        current_variants = child_variants
        current_variant_mode = "replace"
    else:
        child_variants = tuple(
            current_variants
            if all(isinstance(item, ExperimentVariant) for item in current_variants)
            else ()
        )

    summary_path = root / "adaptive-loop-summary.json"
    summary = AdaptiveLoopSummary(
        loop_root=root,
        summary_path=summary_path,
        generations=tuple(records),
        final_child_variants=tuple(
            variant for variant in current_variants if isinstance(variant, ExperimentVariant)
        ),
        stopped_reason=stopped_reason,
    )
    _write_json(summary_path, summary.to_dict())
    return summary


def _default_runner_factory(
    plan: ExperimentPlan,
    options: RunOptions,
    base_config: BlockingConfig | None,
) -> _RunnerLike:
    return runner_for_plan(plan, options=options, base_config=base_config)


def _with_loop_metadata(
    plan: ExperimentPlan,
    *,
    loop_root: Path,
    generation: int,
    parent_variant_count: int,
) -> ExperimentPlan:
    metadata = {
        **dict(plan.metadata),
        "adaptive_loop": {
            "loop_root": loop_root.as_posix(),
            "generation": generation,
            "parent_variant_count": parent_variant_count,
        },
    }
    return replace(plan, metadata=metadata)


def _select_parent_results(
    results: Sequence[ExperimentResult],
    *,
    objective: str,
    parent_top_k: int,
    allow_diagnostics: bool = False,
) -> tuple[ExperimentResult, ...]:
    ranked = rank_results(results, objective=objective)
    promotable = [
        result
        for result, _score in ranked
        if promotion_decision(result).promotable
    ]
    if promotable:
        return tuple(promotable[:parent_top_k])
    parent_selectable = [
        result
        for result, _score in ranked
        if promotion_decision(result).parent_selectable
    ]
    if parent_selectable:
        return tuple(parent_selectable[:parent_top_k])
    if allow_diagnostics:
        return tuple(result for result, _score in ranked[:parent_top_k])
    return ()


def _parent_health_summary(
    results: Sequence[ExperimentResult],
    *,
    objective: str,
    parent_top_k: int,
) -> Mapping[str, Any]:
    ranked = rank_results(results, objective=objective)
    states: dict[str, int] = {}
    tiers: dict[str, int] = {}
    promotable: list[ExperimentResult] = []
    parent_selectable: list[ExperimentResult] = []
    quality_blocked_parent_selectable: list[ExperimentResult] = []
    blocked_topology: list[ExperimentResult] = []
    for result in results:
        decision = promotion_decision(result)
        states[decision.state] = states.get(decision.state, 0) + 1
        tiers[decision.tier] = tiers.get(decision.tier, 0) + 1
        if decision.promotable:
            promotable.append(result)
        if decision.parent_selectable:
            parent_selectable.append(result)
            if not decision.promotable:
                quality_blocked_parent_selectable.append(result)
        if decision.state == "blocked_topology":
            blocked_topology.append(result)
    render_ranked = sorted(
        results,
        key=lambda result: (result.min_iou, result.avg_iou, -result.elapsed_s),
        reverse=True,
    )
    return {
        "result_count": len(results),
        "promotable_count": len(promotable),
        "parent_selectable_count": len(parent_selectable),
        "quality_blocked_parent_selectable_count": len(quality_blocked_parent_selectable),
        "blocked_count": len(results) - len(promotable),
        "promotion_state_counts": dict(sorted(states.items())),
        "promotion_tier_counts": dict(sorted(tiers.items())),
        "topology_blocked_count": len(blocked_topology),
        "top_promotable_parent_ids": [
            result.variant_id
            for result, _score in ranked
            if promotion_decision(result).promotable
        ][:parent_top_k],
        "top_parent_selectable_ids": [
            result.variant_id
            for result, _score in ranked
            if promotion_decision(result).parent_selectable
        ][:parent_top_k],
        "quality_blocked_parent_selectable": [
            _parent_health_result_row(result)
            for result in quality_blocked_parent_selectable[:parent_top_k]
        ],
        "top_render_winners": [
            _parent_health_result_row(result)
            for result in render_ranked[:parent_top_k]
        ],
        "top_ranked_candidates": [
            _parent_health_result_row(result)
            for result, _score in ranked[:parent_top_k]
        ],
    }


def _parent_health_result_row(result: ExperimentResult) -> Mapping[str, Any]:
    decision = promotion_decision(result)
    return {
        "variant_id": result.variant_id,
        "case_id": result.case_id,
        "mode": result.mode,
        "status": result.status,
        "avg_iou": result.avg_iou,
        "min_iou": result.min_iou,
        "elapsed_s": result.elapsed_s,
        "promotion_state": decision.state,
        "promotion_tier": decision.tier,
        "promotable": decision.promotable,
        "parent_selectable": decision.parent_selectable,
        "blockers": list(decision.blockers),
        "parent_selection_blockers": list(decision.blocking_for_parent_selection),
    }


def _child_variants_from_results(
    results: Sequence[ExperimentResult],
    *,
    generation: int,
    children_per_parent: int,
) -> tuple[tuple[RefinementProposal, ExperimentResult], tuple[ExperimentVariant, ...]]:
    proposal_pairs: list[tuple[RefinementProposal, ExperimentResult]] = []
    variants: list[ExperimentVariant] = []
    seen_ids: set[str] = set()
    seen_effective: dict[str, int] = {}
    for result in results:
        payload = _proposal_payload_from_result(result)
        proposals = merge_proposals(
            proposals_from_result_payload(
                payload,
                max_proposals=children_per_parent,
            ),
            max_proposals=children_per_parent,
        )
        for proposal in proposals:
            proposal_pairs.append((proposal, result))
            variant = _variant_for_child(
                proposal,
                parent=result,
                generation=generation,
                seen_ids=seen_ids,
                force_diagnostic_only=not promotion_decision(result).parent_selectable,
            )
            effective_key = _effective_child_variant_key(variant)
            existing_index = seen_effective.get(effective_key)
            if existing_index is not None:
                variants[existing_index] = _with_contributing_parent(
                    variants[existing_index],
                    result,
                )
                continue
            seen_effective[effective_key] = len(variants)
            variant = _with_contributing_parent(variant, result)
            variants.append(variant)
    return tuple(proposal_pairs), tuple(variants)


def _proposal_payload_from_result(result: ExperimentResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": result.status,
        "metrics": dict(result.metrics),
        "backend_result": dict(result.backend_result),
    }
    if result.autopsy:
        payload["autopsy_pack"] = result.autopsy
    if result.bounds_debug:
        payload["bounds_debug"] = result.bounds_debug
    return payload


def _variant_for_child(
    proposal: RefinementProposal,
    *,
    parent: ExperimentResult,
    generation: int,
    seen_ids: set[str],
    force_diagnostic_only: bool = False,
) -> ExperimentVariant:
    base = proposal.to_variant(parent_variant_id=parent.variant_id)
    unique_hash = stable_hash(
        {
            "generation": generation,
            "parent": parent.variant_id,
            "proposal": proposal.proposal_id,
            "mode": base.mode,
            "cli_args": base.cli_args,
            "config_overrides": base.config_overrides,
        }
    )[:8]
    parent_token = _adaptive_id_token(parent.variant_id, max_length=36)
    proposal_token = _adaptive_id_token(proposal.proposal_id, max_length=44)
    variant_id = safe_slug(
        f"g{generation + 1:02d}_{parent_token}_{proposal_token}_{unique_hash}"
    )
    while variant_id in seen_ids:
        unique_hash = stable_hash({"variant_id": variant_id, "count": len(seen_ids)})[:8]
        variant_id = safe_slug(
            f"g{generation + 1:02d}_{parent_token}_{proposal_token}_{unique_hash}"
        )
    seen_ids.add(variant_id)
    parameters = {
        **dict(base.parameters),
        "adaptive_loop_generation": generation + 1,
        "adaptive_loop_parent_result": parent.variant_id,
        "adaptive_loop_parent_mode": parent.mode,
    }
    tags = tuple(dict.fromkeys(("adaptive-loop",) + tuple(base.tags)))
    return ExperimentVariant(
        variant_id=variant_id,
        label=f"G{generation + 1}: {base.label}",
        mode=base.mode,
        validation_mode=base.validation_mode,
        parameters=parameters,
        cli_args=base.cli_args,
        config_overrides=base.config_overrides,
        expected_artifacts=base.expected_artifacts,
        tags=tags,
        parent_variant_id=parent.variant_id,
        stage=f"adaptive_loop_generation_{generation + 1}",
        diagnostic_only=base.diagnostic_only or force_diagnostic_only,
    )


def _effective_child_variant_key(variant: ExperimentVariant) -> str:
    ignored_parameters = {
        "adaptive_loop_generation",
        "adaptive_loop_parent_result",
        "adaptive_loop_parent_mode",
        "adaptive_loop_contributing_parent_results",
    }
    return stable_hash(
        {
            "mode": variant.mode,
            "validation_mode": variant.validation_mode,
            "parameters": {
                key: value
                for key, value in dict(variant.parameters).items()
                if key not in ignored_parameters
            },
            "cli_args": variant.cli_args,
            "config_overrides": variant.config_overrides,
            "diagnostic_only": variant.diagnostic_only,
        },
        length=24,
    )


def _with_contributing_parent(
    variant: ExperimentVariant,
    parent: ExperimentResult,
) -> ExperimentVariant:
    parameters = dict(variant.parameters)
    existing = tuple(parameters.get("adaptive_loop_contributing_parent_results") or ())
    parent_id = str(parent.variant_id)
    if parent_id not in existing:
        parameters["adaptive_loop_contributing_parent_results"] = existing + (parent_id,)
    return replace(variant, parameters=parameters)


def _adaptive_id_token(value: str, *, max_length: int) -> str:
    slug = safe_slug(value, fallback="variant")
    if len(slug) <= max_length:
        return slug
    digest = stable_hash({"adaptive_id": value}, length=8)
    stem = slug[: max(1, max_length - 9)].rstrip("_-") or "variant"
    return f"{stem}_{digest}"


def _write_generation_adaptive_outputs(
    *,
    proposal_path: Path,
    variant_path: Path,
    generation: int,
    selected: Sequence[ExperimentResult],
    proposals: Sequence[tuple[RefinementProposal, ExperimentResult]],
    child_variants: Sequence[ExperimentVariant],
    parent_health: Mapping[str, Any],
) -> None:
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_payload = {
        "schema_version": "refinement_adaptive_loop_proposals_v1",
        "generation": generation,
        "selected_parent_ids": [result.variant_id for result in selected],
        "parent_health": dict(parent_health),
        "proposal_count": len(proposals),
        "proposals": [
            {
                **proposal.to_dict(),
                "parent_variant_id": result.variant_id,
                "parent_case_id": result.case_id,
                "parent_mode": result.mode,
            }
            for proposal, result in proposals
        ],
    }
    variant_payload = {
        "schema_version": "refinement_adaptive_loop_variants_v1",
        "generation": generation + 1,
        "parent_generation": generation,
        "parent_health": dict(parent_health),
        "variant_count": len(child_variants),
        "variants": [variant.to_dict() for variant in child_variants],
    }
    _write_json(proposal_path, proposal_payload)
    _write_json(variant_path, variant_payload)
