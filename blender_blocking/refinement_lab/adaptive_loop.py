"""Closed-loop adaptive refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .adaptive import RefinementProposal, merge_proposals, proposals_from_result_payload
from .contracts import ExperimentPlan, ExperimentResult, ExperimentVariant, json_safe, safe_slug, stable_hash
from .matrix import build_experiment_plan
from .parameter_search import promotion_decision, rank_results
from .runner import RunOptions, runner_for_plan

try:
    from blender_blocking.config import BlockingConfig
except ImportError:  # pragma: no cover
    from config import BlockingConfig


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

    def to_dict(self) -> dict[str, object]:
        return {
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
        }


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
        ok, results = runner_factory(plan, options.run_options, base_config).run()
        selected = _select_parent_results(
            results,
            objective=objective,
            parent_top_k=options.parent_top_k,
            allow_diagnostics=options.allow_diagnostic_children,
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
            )
        )
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
    summary_path.write_text(
        json.dumps(json_safe(summary.to_dict()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    if allow_diagnostics:
        return tuple(result for result, _score in ranked[:parent_top_k])
    return ()


def _child_variants_from_results(
    results: Sequence[ExperimentResult],
    *,
    generation: int,
    children_per_parent: int,
) -> tuple[tuple[RefinementProposal, ExperimentResult], tuple[ExperimentVariant, ...]]:
    proposal_pairs: list[tuple[RefinementProposal, ExperimentResult]] = []
    variants: list[ExperimentVariant] = []
    seen_ids: set[str] = set()
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
                force_diagnostic_only=not promotion_decision(result).promotable,
            )
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
) -> None:
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_payload = {
        "schema_version": "refinement_adaptive_loop_proposals_v1",
        "generation": generation,
        "selected_parent_ids": [result.variant_id for result in selected],
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
        "variant_count": len(child_variants),
        "variants": [variant.to_dict() for variant in child_variants],
    }
    proposal_path.write_text(
        json.dumps(json_safe(proposal_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    variant_path.write_text(
        json.dumps(json_safe(variant_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
