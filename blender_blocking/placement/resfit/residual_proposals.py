"""Bounded input-residual proposals accepted by the complete fitting objective."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from ..resfit_initialization import PrimitiveInitializationConfig, deterministic_kmeans
from ..resfit_objective import ResFitObjectiveResult, primitive_sdf_matrix
from ..resfit_optimizer import (CoordinateDescentConfig, OptimizationBudget,
                                OptimizationBudgetExhausted, OptimizationRecord,
                                coordinate_descent_optimize)
from .initialization import get_initializer


@dataclass(frozen=True)
class ResidualProposal:
    label: str
    primitives: tuple[object, ...]
    priority_parts: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResidualRefinementResult:
    primitives: tuple[object, ...]
    final_result: ResFitObjectiveResult
    history: tuple[OptimizationRecord, ...]
    proposals: tuple[Mapping[str, object], ...]
    accepted_proposals: int = 0
    objective_evaluations: int = 0


def _cluster_points(points: np.ndarray, count: int, minimum: int) -> list[np.ndarray]:
    if len(points) < minimum * count:
        return []
    _, labels = deterministic_kmeans(points, count, iterations=6, seed_strategy="farthest")
    return [points[labels == index] for index in range(count)
            if np.count_nonzero(labels == index) >= minimum]


def generate_residual_proposals(
    primitives: Sequence[object], target_points: np.ndarray, primitive_family: str,
    *, max_parts: int = 12, max_proposals: int = 4, residual_quantile: float = 0.75,
    minimum_cluster_points: int = 8,
    initialization_config: PrimitiveInitializationConfig | None = None,
) -> tuple[ResidualProposal, ...]:
    """Propose addition, replacement and split using only input-derived points.

    Proposals are independent alternatives to the same state. At the part cap,
    replace a low-responsibility part or reallocate it to split the highest-error
    part. No alternative is accepted here, and no benchmark truth is consumed.
    """
    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("target_points must be finite with shape (N,3)")
    if max_parts < 1 or max_proposals < 1 or minimum_cluster_points < 2:
        raise ValueError("proposal limits must be positive")
    if not 0.0 <= residual_quantile < 1.0:
        raise ValueError("residual_quantile must be in [0,1)")
    if len(primitives) > max_parts:
        raise ValueError("current primitive count exceeds max_parts")
    if len(points) < minimum_cluster_points:
        return ()
    initializer = get_initializer(primitive_family)
    seed_config = initialization_config or PrimitiveInitializationConfig()

    def seed(cluster: np.ndarray) -> tuple[object, ...]:
        config = replace(seed_config, primitive_count=1, kmeans_seed="farthest")
        return tuple(initializer(cluster, config))[:1]

    def candidate(label: str, parts: Sequence[object]) -> ResidualProposal:
        from ..resfit_objective import _geometry_key
        old_keys = {_geometry_key(part) for part in primitives}
        changed = tuple(index for index, part in enumerate(parts)
                        if _geometry_key(part) not in old_keys or _geometry_key(part) is None)
        return ResidualProposal(label, tuple(deepcopy(part) for part in parts), changed)

    if not primitives:
        parts = seed(points)
        return (candidate("initialize_input_residual", parts),) if parts else ()
    fields = primitive_sdf_matrix(primitives, points)
    union = np.min(fields, axis=0)
    errors = np.abs(union)
    threshold = float(np.quantile(errors, residual_quantile))
    high_mask = (errors >= threshold) & (errors > 1e-9)
    if np.count_nonzero(high_mask) < minimum_cluster_points:
        return ()
    # Near-boundary responsibility avoids handing every contour to a deeply
    # enclosing primitive. Soft ownership prevents abrupt allocation at ties.
    length = max(float(np.max(np.ptp(points, axis=0))), np.finfo(float).tiny)
    distances = np.abs(fields)
    logits = -(distances - distances.min(axis=0)) / (length * .03)
    ownership = np.exp(np.maximum(logits, -700.))
    ownership /= ownership.sum(axis=0)
    owners = np.argmax(ownership, axis=0)
    responsibilities = ownership.sum(axis=1)
    part_error = (ownership * errors[None, :] ** 2).sum(axis=1)
    worst = int(np.argmax(part_error))
    low = int(np.argmin(responsibilities))
    proposals: list[ResidualProposal] = []
    # Positive union residual denotes missing coverage. Separate error regions
    # before seeding so two distant missing parts are not averaged into one.
    missing = points[high_mask & (union > 0.0)]
    from blender_blocking.reconstruction.spatial_regions import connected_point_regions
    missing_clusters = connected_point_regions(missing, minimum=minimum_cluster_points, max_regions=3)
    if not missing_clusters:
        missing_clusters = _cluster_points(missing, 2, minimum_cluster_points)
    if not missing_clusters and len(missing) >= minimum_cluster_points:
        missing_clusters = [missing]
    for index, cluster in enumerate(missing_clusters):
        addition = seed(cluster)
        if not addition:
            continue
        if len(primitives) < max_parts:
            proposals.append(candidate(f"add_residual_cluster_{index}", (*primitives, *addition)))
        else:
            kept = [part for part_index, part in enumerate(primitives) if part_index != low]
            proposals.append(candidate(f"replace_low_responsibility_{low}_cluster_{index}", (*kept, *addition)))
    # Re-seed one poorly explained component, retaining every other component.
    local = points[owners == worst]
    local_seeds = seed(local) if len(local) >= minimum_cluster_points else ()
    if local_seeds:
        refined = list(primitives)
        refined[worst] = local_seeds[0]
        proposals.append(candidate(f"refine_residual_part_{worst}", refined))
    split_clusters = _cluster_points(local, 2, minimum_cluster_points)
    if len(split_clusters) == 2:
        split = tuple(part for cluster in split_clusters for part in seed(cluster))
        if len(split) == 2:
            if len(primitives) < max_parts:
                kept = [part for index, part in enumerate(primitives) if index != worst]
                proposals.append(candidate(f"split_residual_part_{worst}", (*kept, *split)))
            elif len(primitives) > 1:
                other = min((index for index in range(len(primitives)) if index != worst),
                            key=lambda index: (responsibilities[index], index))
                kept = [part for index, part in enumerate(primitives) if index not in (worst, other)]
                proposals.append(candidate(f"reallocate_part_{other}_split_{worst}", (*kept, *split)))
    # Regrow ownership partitions from nearest input neighborhoods, then fit a
    # different family to the worst region. Prune and merge are complete-objective
    # alternatives and cannot be accepted by responsibility counts alone.
    from scipy.spatial import cKDTree
    _, adjacent = cKDTree(points).query(points, k=min(7, len(points)))
    regrown = np.array(owners, copy=True)
    for i, row in enumerate(adjacent):
        labels, counts = np.unique(owners[row], return_counts=True)
        regrown[i] = labels[np.argmax(counts)]
    region = points[regrown == worst]
    for family in ("ellipsoid", "superquadric", "superfrustum"):
        if family == primitive_family or len(region) < minimum_cluster_points:
            continue
        try:
            mixed_seed = tuple(get_initializer(family)(region, replace(seed_config, primitive_count=1)))
            if mixed_seed:
                kept = [p for i, p in enumerate(primitives) if i != worst]
                proposals.append(candidate(f"regrow_mixed_{family}_{worst}", (*kept, mixed_seed[0])))
        except (ValueError, KeyError):
            continue
    if len(primitives) > 1:
        proposals.append(candidate(f"prune_low_responsibility_{low}", [p for i, p in enumerate(primitives) if i != low]))
        centers = np.array([points[owners == i].mean(0) if np.any(owners == i) else points.mean(0) for i in range(len(primitives))])
        separation = np.linalg.norm(centers[:, None]-centers[None], axis=2)
        np.fill_diagonal(separation, np.inf)
        a, b = np.unravel_index(np.argmin(separation), separation.shape)
        merged_seed = seed(points[(owners == a)|(owners == b)])
        if merged_seed:
            proposals.append(candidate(f"merge_neighbors_{a}_{b}", (*[p for i, p in enumerate(primitives) if i not in {a, b}], *merged_seed)))
    # Round-robin operations before a second seed from the same operation.
    # Quality's eight-proposal cap reaches regrow, prune and merge as well.
    def operation(label):
        if label.startswith(('add_', 'replace_')):
            return 'add'
        if label.startswith(('split_', 'reallocate_')):
            return 'split'
        if label.startswith('regrow_'):
            return 'regrow'
        return label.split('_', 1)[0]
    buckets = {}
    for proposal in proposals:
        buckets.setdefault(operation(proposal.label), []).append(proposal)
    # Preserve separate missing-region seeds before operation alternatives.
    ordered = list(buckets.pop("add", ()))
    while any(buckets.values()):
        for kind in ('add', 'split', 'regrow', 'prune', 'merge', 'refine'):
            if buckets.get(kind):
                ordered.append(buckets[kind].pop(0))
    return tuple(ordered[:int(max_proposals)])



def refine_residual_parts(
    primitives: Sequence[object], target_points: np.ndarray, primitive_family: str,
    objective, optimizer_config: CoordinateDescentConfig, *, budget: OptimizationBudget,
    initial_result: ResFitObjectiveResult | None = None, max_parts: int = 12,
    max_proposals: int = 4, refinement_steps: int = 2, rounds: int = 1,
    initialization_config: PrimitiveInitializationConfig | None = None,
) -> ResidualRefinementResult:
    """Refine independent proposals and retain only a full-objective improvement.

    The shared existing budget counts scoring and refinement. A failed or timed
    out proposal never discards an already accepted state. Competing proposals
    are not combined; the next round is regenerated from the retained winner.
    """
    if refinement_steps < 0 or rounds < 0:
        raise ValueError("refinement_steps and rounds must be nonnegative")
    current = tuple(deepcopy(part) for part in primitives)
    evaluation_start = budget.objective_evaluations
    current_result = initial_result or budget.evaluate(objective, current)
    if not np.isfinite(current_result.total):
        raise ValueError("initial objective must be finite")
    summaries: list[Mapping[str, object]] = []
    history: list[OptimizationRecord] = []
    accepted = 0
    for round_index in range(int(rounds)):
        if budget.reason() is not None:
            break
        proposals = generate_residual_proposals(
            current, target_points, primitive_family, max_parts=max_parts,
            max_proposals=max_proposals, initialization_config=initialization_config,
        )
        winner = None
        winner_loss = current_result.total
        for index, proposal in enumerate(proposals):
            if budget.reason() is not None:
                break
            allowance = budget.child(slots=len(proposals) - index)
            config = replace(optimizer_config, iterations=int(refinement_steps))
            try:
                optimized = coordinate_descent_optimize(
                    proposal.primitives, objective, config, budget=allowance,
                    priority_parts=proposal.priority_parts,
                )
                final = optimized.final_result
                if final is None or not np.isfinite(final.total):
                    raise ValueError("proposal objective must be finite")
                improved = final.total < winner_loss - 1e-12 * max(1.0, abs(winner_loss))
                summaries.append({"label": proposal.label, "round": round_index,
                                  "status": "improved" if improved else "rejected",
                                  "total": float(final.total), "parts": len(optimized.primitives),
                                  "objective_evaluations": optimized.objective_evaluations,
                                  "priority_parts": list(proposal.priority_parts),
                                  "parameter_visits": list(optimized.parameter_visits)})
                if improved:
                    winner, winner_loss = optimized, float(final.total)
            except OptimizationBudgetExhausted:
                break
            except Exception as exc:
                summaries.append({"label": proposal.label, "round": round_index,
                                  "status": "failed", "error": str(exc)})
        if winner is None:
            break
        current = winner.primitives
        current_result = winner.final_result
        history.extend(winner.history)
        accepted += 1
    return ResidualRefinementResult(
        current, current_result, tuple(history), tuple(summaries), accepted,
        budget.objective_evaluations - evaluation_start,
    )
