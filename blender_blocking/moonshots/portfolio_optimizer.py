"""Cross-sidecar portfolio optimizer for moonshot next actions."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
)
from .support import bounded, candidate_rows, row_metric, target_signals


EXPERIMENT = MoonshotExperiment(
    experiment_id="moonshot_portfolio_optimizer",
    title="Cross-sidecar moonshot action portfolio",
    subsystem="refinement_planning",
    hypothesis=(
        "The most useful moonshot result is usually not the largest isolated delta; "
        "it is a bounded action portfolio that combines view capture, editable "
        "program search, SDF smoothing, retopology, and learned constraints with "
        "explicit risk gates."
    ),
    expected_wins={
        "planning": "turn sidecar evidence into ranked next actions instead of disconnected reports",
        "quality": "prefer high-delta repairs only when their silhouette/editability guards are credible",
    },
    required_inputs=("moonshot_sidecar_results", "candidate_metrics", "target_signals"),
    validation_metrics=(
        "portfolio_expected_quality_delta",
        "portfolio_expected_editability_delta",
        "portfolio_risk",
    ),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        rows = candidate_rows(request.candidate)
        signals = target_signals(request, include_profile_rows=True)
        prior_results = _prior_results(request)
        if not prior_results and not rows and not any(
            bool(group.get("available")) for group in signals.values()
        ):
            return skipped_result(
                request,
                reason="portfolio optimizer needs candidate metrics, target signals, or prior sidecar results",
                next_steps=("run after at least one moonshot sidecar or candidate evaluation",),
            )
        return _optimize_portfolio(
            request,
            rows=rows,
            signals=signals,
            prior_results=prior_results,
        )
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _optimize_portfolio(
    request: MoonshotRequest,
    *,
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    prior_results: Sequence[Mapping[str, Any]],
) -> MoonshotResult:
    config = dict(request.config or {})
    max_actions = max(1, int(config.get("portfolio_max_actions", 4) or 4))
    budget = max(0.5, float(config.get("portfolio_budget", 3.0) or 3.0))
    baseline = _baseline(rows)
    ranking_priors = _ranking_priors_from_results(prior_results)
    actions = []
    for result in prior_results:
        actions.extend(_actions_from_result(result, baseline=baseline))
    actions.extend(_fallback_actions(rows=rows, signals=signals, baseline=baseline))
    actions = _dedupe_actions(actions)
    ranked = sorted(
        (
            _score_action(
                action,
                baseline=baseline,
                ranking_priors=ranking_priors,
            )
            for action in actions
        ),
        key=lambda action: (float(action["score"]), str(action["action_id"])),
        reverse=True,
    )
    selected = _select_portfolio(ranked, max_actions=max_actions, budget=budget)
    rejected = [
        {
            "action_id": action["action_id"],
            "kind": action["kind"],
            "source": action.get("source", ""),
            "score": action["score"],
            "rejection": _rejection_reason(
                action,
                selected=selected,
                budget=budget,
                ranked=ranked,
            ),
        }
        for action in ranked
        if action["action_id"] not in {item["action_id"] for item in selected}
    ][:8]
    totals = _portfolio_totals(selected)
    gates = _portfolio_gates(selected, baseline=baseline)
    dependency_edges = _dependency_edges(selected)
    execution_plan = _execution_plan(selected, dependency_edges, gates)
    risk_register = _risk_register(
        selected,
        gates=gates,
        baseline=baseline,
        prior_results=prior_results,
    )
    evidence = {
        "baseline": baseline,
        "budget": budget,
        "max_actions": max_actions,
        "ranking_priors": ranking_priors,
        "prior_result_count": len(prior_results),
        "candidate_action_count": len(ranked),
        "available_actions": ranked,
        "selected_actions": selected,
        "rejected_actions": rejected,
        "dependency_edges": dependency_edges,
        "execution_plan": execution_plan,
        "risk_register": risk_register,
        "gates": gates,
        "does_not_change_reconstruction_contract": True,
    }
    metrics = {
        "ran": 1.0,
        "action_count": float(len(ranked)),
        "selected_action_count": float(len(selected)),
        "ranking_prior_count": float(len(ranking_priors)),
        "dependency_edge_count": float(len(dependency_edges)),
        "execution_stage_count": float(len(execution_plan)),
        "risk_register_count": float(len(risk_register)),
        "portfolio_budget_used": sum(float(item.get("cost", 0.0) or 0.0) for item in selected),
        "portfolio_expected_quality_delta": totals["quality_delta"],
        "portfolio_expected_editability_delta": totals["editability_delta"],
        "portfolio_expected_uncertainty_reduction": totals["uncertainty_reduction"],
        "portfolio_risk": totals["risk"],
        "best_action_score": float(selected[0]["score"]) if selected else 0.0,
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifact_name="bundle.json",
        next_steps=tuple(str(gate["next_step"]) for gate in gates[:4]),
    )


def _prior_results(request: MoonshotRequest) -> tuple[Mapping[str, Any], ...]:
    config = request.config or {}
    if not isinstance(config, Mapping):
        return ()
    for key in ("prior_moonshot_results", "moonshot_results", "results"):
        value = config.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
            rows = tuple(item for item in value if isinstance(item, Mapping))
            if rows:
                return rows
    return ()


def _actions_from_result(
    result: Mapping[str, Any],
    *,
    baseline: Mapping[str, float],
) -> list[dict[str, Any]]:
    experiment_id = str(result.get("experiment_id", ""))
    if result.get("status") != "ran":
        return []
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    evidence = _evidence(result)
    if experiment_id == "shape_grammar_search":
        return _grammar_actions(metrics, evidence, baseline=baseline)
    if experiment_id == "active_view_planning":
        return _active_view_actions(evidence)
    if experiment_id == "implicit_sdf_proxy":
        return _sdf_actions(metrics, evidence)
    if experiment_id == "editable_retopology":
        return _retopology_actions(metrics, evidence)
    if experiment_id == "human_constraint_learning":
        return _constraint_actions(metrics, evidence)
    if experiment_id == "differentiable_primitives":
        return _differentiable_actions(metrics, evidence)
    return []


def _grammar_actions(
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    baseline: Mapping[str, float],
) -> list[dict[str, Any]]:
    selected_id = str(evidence.get("selected_candidate_id") or "")
    candidates = evidence.get("candidates")
    selected = {}
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for item in candidates:
            if isinstance(item, Mapping) and str(item.get("candidate_id")) == selected_id:
                selected = dict(item)
                break
    candidate_metrics = (
        selected.get("moonshot_metrics")
        if isinstance(selected.get("moonshot_metrics"), Mapping)
        else {}
    )
    compile_plan = evidence.get("selected_compile_plan")
    compile_plan = compile_plan if isinstance(compile_plan, Mapping) else {}
    validation_summary = evidence.get("validation_summary")
    validation_summary = validation_summary if isinstance(validation_summary, Mapping) else {}
    family_hints = evidence.get("family_hints")
    family_hints = family_hints if isinstance(family_hints, Sequence) and not isinstance(family_hints, (str, bytes)) else ()
    quality_delta = bounded(
        float(candidate_metrics.get("silhouette_proxy", metrics.get("selected_silhouette_proxy", 0.0)) or 0.0)
        - baseline["min_iou"],
        -0.08,
        0.22,
    )
    editability_delta = bounded(
        float(candidate_metrics.get("editability", metrics.get("selected_editability", 0.0)) or 0.0)
        - baseline["editability"],
        -0.10,
        0.35,
    )
    complexity = float(candidate_metrics.get("complexity", 0.25) or 0.25)
    compile_ready = bool(compile_plan.get("compile_ready", True))
    compile_risk = 0.0 if compile_ready else 0.18
    compile_guards = compile_plan.get("guards")
    if not isinstance(compile_guards, Sequence) or isinstance(compile_guards, (str, bytes)):
        compile_guards = ()
    return [
        _action(
            "shape_grammar_search",
            "compile_selected_shape_program",
            quality_delta=quality_delta,
            editability_delta=editability_delta,
            uncertainty_reduction=0.03,
            risk=bounded(0.22 + compile_risk + complexity * 0.28 - max(editability_delta, 0.0) * 0.15),
            cost=1.1,
            payload={
                "selected_candidate_id": selected_id,
                "selected_fingerprint": str(evidence.get("selected_fingerprint", "")),
                "grammar_id": str(evidence.get("grammar_id", "")),
                "beam_width": int(evidence.get("beam_width", 0) or 0),
                "compile_plan": dict(compile_plan),
                "family_hints": [dict(item) for item in family_hints if isinstance(item, Mapping)][:6],
                "validation_summary": dict(validation_summary),
            },
            guards=(
                "compile program in Blender before promotion",
                "reject if render.min_view_iou regresses",
                *tuple(str(item) for item in compile_guards),
            ),
            stage_hint="model_generation",
            prerequisites=("inject_learned_constraints",),
        )
    ]


def _active_view_actions(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    requests = evidence.get("requests")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        return []
    sequence = evidence.get("sequence_plan")
    sequence_by_view: dict[str, Mapping[str, Any]] = {}
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
        for item in sequence:
            if isinstance(item, Mapping):
                sequence_by_view[str(item.get("view_id", ""))] = item
    actions = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            continue
        sequence_item = sequence_by_view.get(str(request.get("view_id", "")), {})
        delta = float(request.get("expected_metric_delta", 0.0) or 0.0)
        uncertainty = float(request.get("expected_uncertainty_reduction", 0.0) or 0.0)
        actions.append(
            _action(
                "active_view_planning",
                "capture_next_best_view",
                quality_delta=delta,
                editability_delta=0.0,
                uncertainty_reduction=uncertainty,
                risk=0.08 if not request.get("already_captured") else 0.16,
                cost=float(request.get("capture_cost", 1.0) or 1.0),
                payload={
                    "view_id": str(request.get("view_id", f"view_{index:02d}")),
                    "azimuth_deg": float(request.get("azimuth_deg", 0.0) or 0.0),
                    "elevation_deg": float(request.get("elevation_deg", 0.0) or 0.0),
                    "related_views": list(request.get("related_views", ()) or ()),
                    "sequence_order": int(sequence_item.get("order", index + 1) or index + 1),
                    "marginal_expected_metric_delta": float(
                        sequence_item.get("marginal_expected_metric_delta", delta) or delta
                    ),
                    "decision_factors": dict(request.get("decision_factors", {}) or {}),
                },
                guards=(
                    "measure actual metric delta after capture",
                    str(sequence_item.get("guard", "rerank candidates before follow-on actions")),
                ),
                stage_hint="evidence_capture",
            )
        )
    return actions


def _sdf_actions(metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    mesh_proxy = evidence.get("mesh_proxy") if isinstance(evidence.get("mesh_proxy"), Mapping) else {}
    extraction_plan = evidence.get("extraction_plan")
    extraction_plan = extraction_plan if isinstance(extraction_plan, Mapping) else {}
    plan_guards = extraction_plan.get("guards")
    if not isinstance(plan_guards, Sequence) or isinstance(plan_guards, (str, bytes)):
        plan_guards = ()
    sparse_voxels = float(metrics.get("sparse_voxels", 0.0) or 0.0)
    quality_delta = float(metrics.get("volumetric_iou_delta", 0.0) or 0.0)
    normal_delta = float(metrics.get("normal_consistency_delta", 0.0) or 0.0)
    return [
        _action(
            "implicit_sdf_proxy",
            "extract_guarded_sdf_proxy",
            quality_delta=quality_delta,
            editability_delta=0.02 if mesh_proxy.get("editable_retopology_recommended") else 0.0,
            uncertainty_reduction=normal_delta,
            risk=bounded(0.18 + min(0.22, sparse_voxels / 1_000_000.0)),
            cost=1.35,
            payload={
                "estimated_vertices": int(mesh_proxy.get("estimated_vertices", 0) or 0),
                "estimated_faces": int(mesh_proxy.get("estimated_faces", 0) or 0),
                "postprocess": str(mesh_proxy.get("postprocess", "")),
                "extraction_plan": dict(extraction_plan),
                "strategy": str(extraction_plan.get("strategy", "")),
                "level_count": int(extraction_plan.get("level_count", metrics.get("level_count", 0)) or 0),
                "worst_view": dict(extraction_plan.get("worst_view", {}) or {}),
            },
            guards=(
                "run topology QA before export",
                "reject if silhouette boundary IoU regresses",
                *tuple(_guard_to_text(item) for item in plan_guards),
            ),
            stage_hint="model_generation",
            prerequisites=("capture_next_best_view",),
        )
    ]


def _retopology_actions(metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    qa = evidence.get("qa") if isinstance(evidence.get("qa"), Mapping) else {}
    repair_plan = evidence.get("repair_plan")
    plan_count = len(repair_plan) if isinstance(repair_plan, Sequence) and not isinstance(repair_plan, (str, bytes)) else 0
    phase_plan = evidence.get("phase_plan")
    phase_plan = phase_plan if isinstance(phase_plan, Sequence) and not isinstance(phase_plan, (str, bytes)) else ()
    acceptance_gates = evidence.get("acceptance_gates")
    acceptance_gates = acceptance_gates if isinstance(acceptance_gates, Sequence) and not isinstance(acceptance_gates, (str, bytes)) else ()
    human_review = bool(qa.get("human_review_required"))
    return [
        _action(
            "editable_retopology",
            "apply_guarded_retopology_plan",
            quality_delta=max(0.0, float(metrics.get("topology_score_delta", 0.0) or 0.0) * 0.35),
            editability_delta=float(metrics.get("editability_score_delta", 0.0) or 0.0),
            uncertainty_reduction=0.0,
            risk=bounded(float(metrics.get("silhouette_risk", qa.get("silhouette_risk", 0.35)) or 0.35) + (0.08 if human_review else 0.0)),
            cost=0.85 + min(0.8, plan_count * 0.12),
            payload={
                "repair_step_count": plan_count,
                "phase_plan": [dict(item) for item in phase_plan if isinstance(item, Mapping)],
                "acceptance_gates": [dict(item) for item in acceptance_gates if isinstance(item, Mapping)],
                "safe_automatic": bool(qa.get("safe_automatic")),
                "human_review_required": human_review,
            },
            guards=(
                "execute only if safe_automatic is true or reviewer accepts plan",
                "rerender required silhouettes after repair",
                *tuple(_guard_to_text(item) for item in acceptance_gates),
            ),
            stage_hint="mesh_repair",
            prerequisites=("extract_guarded_sdf_proxy", "compile_selected_shape_program"),
        )
    ]


def _constraint_actions(metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    constraints = evidence.get("constraints")
    count = len(constraints) if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)) else 0
    constraints = constraints if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)) else ()
    ranking_priors = evidence.get("ranking_priors")
    ranking_priors = ranking_priors if isinstance(ranking_priors, Sequence) and not isinstance(ranking_priors, (str, bytes)) else ()
    satisfaction = evidence.get("satisfaction") if isinstance(evidence.get("satisfaction"), Mapping) else {}
    mean_satisfaction = float(satisfaction.get("mean_satisfaction", 0.75) or 0.75)
    confidence = float(metrics.get("mean_confidence", 0.0) or 0.0)
    return [
        _action(
            "human_constraint_learning",
            "inject_learned_constraints",
            quality_delta=bounded((1.0 - mean_satisfaction) * 0.08 + confidence * 0.04),
            editability_delta=bounded(confidence * 0.08),
            uncertainty_reduction=0.03,
            risk=bounded(0.12 + max(0.0, 0.55 - confidence) * 0.25),
            cost=0.45,
            payload={
                "constraint_count": count,
                "mean_confidence": confidence,
                "hard_constraint_count": int(metrics.get("hard_constraint_count", 0.0) or 0.0),
                "constraint_kinds": [str(item.get("kind", "")) for item in constraints if isinstance(item, Mapping)][:12],
                "ranking_priors": [dict(item) for item in ranking_priors if isinstance(item, Mapping)][:12],
            },
            guards=("keep learned constraints explicit in candidate config",),
            stage_hint="evidence_capture",
        )
    ]


def _differentiable_actions(metrics: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule = evidence.get("finite_difference_schedule")
    schedule = schedule if isinstance(schedule, Sequence) and not isinstance(schedule, (str, bytes)) else ()
    objective_terms = evidence.get("objective_terms")
    objective_terms = objective_terms if isinstance(objective_terms, Sequence) and not isinstance(objective_terms, (str, bytes)) else ()
    probe_count = sum(
        int(item.get("probe_count", 0) or 0)
        for item in schedule
        if isinstance(item, Mapping)
    )
    return [
        _action(
            "differentiable_primitives",
            "run_finite_difference_refinement_probe",
            quality_delta=float(metrics.get("expected_objective_delta", 0.0) or 0.0),
            editability_delta=0.02,
            uncertainty_reduction=max(0.02, abs(float(metrics.get("expected_signed_distance_loss_delta", 0.0) or 0.0))),
            risk=bounded(0.22 + min(0.16, probe_count / 1200.0)),
            cost=1.2 + min(0.45, probe_count / 600.0),
            payload={
                "primitive_count": int(evidence.get("primitive_count", metrics.get("primitive_count", 0)) or 0),
                "gradient_mode": str(evidence.get("gradient_mode", "")),
                "finite_difference_schedule": [dict(item) for item in schedule if isinstance(item, Mapping)],
                "objective_terms": [dict(item) for item in objective_terms if isinstance(item, Mapping)],
                "probe_count": probe_count,
                "trust_region": dict(evidence.get("trust_region", {}) or {}),
            },
            guards=("accept updates only when every required view improves or stays neutral",),
            stage_hint="model_generation",
            prerequisites=("compile_selected_shape_program",),
        )
    ]


def _fallback_actions(
    *,
    rows: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, float],
) -> list[dict[str, Any]]:
    actions = []
    uncertainty = float(signals["uncertainty"].get("overall_boundary_uncertainty_mean", 0.0) or 0.0)
    topology = float(signals["topology"].get("score", baseline["topology"]) or baseline["topology"])
    if baseline["min_iou"] < 0.65 or uncertainty > 0.16:
        actions.append(
            _action(
                "fallback",
                "capture_next_best_view",
                quality_delta=bounded(0.035 + uncertainty * 0.16),
                editability_delta=0.0,
                uncertainty_reduction=bounded(0.12 + uncertainty * 0.35),
                risk=0.10,
                cost=1.0,
                payload={"reason": "low silhouette confidence"},
                guards=("rerank candidates with measured view metrics",),
                stage_hint="evidence_capture",
            )
        )
    if topology < 0.76 or baseline["editability"] < 0.56:
        actions.append(
            _action(
                "fallback",
                "force_editable_shape_program_probe",
                quality_delta=0.04,
                editability_delta=bounded(0.12 + (0.56 - baseline["editability"]) * 0.25),
                uncertainty_reduction=0.0,
                risk=0.26,
                cost=1.0,
                payload={"reason": "editability/topology below floor"},
                guards=("keep backend status unchanged until render QA passes",),
                stage_hint="model_generation",
            )
        )
    if not actions and rows:
        actions.append(
            _action(
                "fallback",
                "no_op_collect_more_evidence",
                quality_delta=0.0,
                editability_delta=0.0,
                uncertainty_reduction=0.0,
                risk=0.02,
                cost=0.1,
                payload={"reason": "candidate already has no obvious moonshot pressure"},
                guards=("do not mutate candidate",),
                stage_hint="qa",
            )
        )
    return actions


def _action(
    source: str,
    kind: str,
    *,
    quality_delta: float,
    editability_delta: float,
    uncertainty_reduction: float,
    risk: float,
    cost: float,
    payload: Mapping[str, Any],
    guards: Sequence[str],
    stage_hint: str = "",
    prerequisites: Sequence[str] = (),
) -> dict[str, Any]:
    stable_bits = "|".join(str(payload.get(key, "")) for key in sorted(payload))
    digest = hashlib.sha1(f"{source}|{kind}|{stable_bits}".encode("utf-8")).hexdigest()[:10]
    return {
        "schema_version": "moonshot_portfolio_action_v1",
        "action_id": f"{source}:{kind}:{digest}",
        "source": source,
        "kind": kind,
        "expected_quality_delta": float(quality_delta),
        "expected_editability_delta": float(editability_delta),
        "expected_uncertainty_reduction": float(uncertainty_reduction),
        "risk": bounded(risk),
        "cost": max(0.05, float(cost)),
        "payload": dict(payload),
        "guards": tuple(str(item) for item in guards),
        "stage_hint": stage_hint or _default_stage(kind),
        "prerequisites": tuple(str(item) for item in prerequisites if item),
    }


def _score_action(
    action: Mapping[str, Any],
    *,
    baseline: Mapping[str, float],
    ranking_priors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    quality = float(action.get("expected_quality_delta", 0.0) or 0.0)
    editability = float(action.get("expected_editability_delta", 0.0) or 0.0)
    uncertainty = float(action.get("expected_uncertainty_reduction", 0.0) or 0.0)
    risk = float(action.get("risk", 0.0) or 0.0)
    cost = max(0.05, float(action.get("cost", 1.0) or 1.0))
    urgency = (
        max(0.0, 0.68 - baseline["min_iou"]) * 0.35
        + max(0.0, 0.72 - baseline["topology"]) * 0.22
        + max(0.0, 0.62 - baseline["editability"]) * 0.22
    )
    prior_bonus = _ranking_prior_bonus(action, ranking_priors)
    score = (
        quality * 1.35
        + editability * 0.85
        + uncertainty * 0.45
        + urgency
        + prior_bonus
        - risk * 0.42
    ) / cost
    return {**dict(action), "ranking_prior_bonus": prior_bonus, "score": float(score)}


def _select_portfolio(
    ranked: Sequence[Mapping[str, Any]],
    *,
    max_actions: int,
    budget: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_budget = 0.0
    used_sources: set[str] = set()
    selected_ids: set[str] = set()

    def add(action: Mapping[str, Any]) -> None:
        nonlocal used_budget
        selected.append(
            {
                **dict(action),
                "_selection_index": len(selected),
            }
        )
        selected_ids.add(str(action.get("action_id", "")))
        used_sources.add(str(action.get("source", "")))
        used_budget += float(action.get("cost", 1.0) or 1.0)

    # First pass: buy source diversity, one useful action per sidecar family.
    for action in ranked:
        if len(selected) >= max_actions:
            break
        source = str(action.get("source", ""))
        if source in used_sources:
            continue
        if not _useful_action(action):
            continue
        if not _can_select_action(
            action,
            selected=selected,
            used_budget=used_budget,
            budget=budget,
            ranked=ranked,
        ):
            continue
        add(action)

    # Second pass: fill remaining capacity by rank once source breadth is present.
    changed = True
    while changed and len(selected) < max_actions:
        changed = False
        for action in ranked:
            if len(selected) >= max_actions:
                break
            if str(action.get("action_id", "")) in selected_ids:
                continue
            if not _useful_action(action):
                continue
            if not _can_select_action(
                action,
                selected=selected,
                used_budget=used_budget,
                budget=budget,
                ranked=ranked,
            ):
                continue
            add(action)
            changed = True

    if not selected:
        for action in ranked:
            cost = float(action.get("cost", 1.0) or 1.0)
            if cost <= budget:
                add(action)
                break
    return _ordered_portfolio(selected)


def _can_select_action(
    action: Mapping[str, Any],
    *,
    selected: Sequence[Mapping[str, Any]],
    used_budget: float,
    budget: float,
    ranked: Sequence[Mapping[str, Any]],
) -> bool:
    cost = float(action.get("cost", 1.0) or 1.0)
    if used_budget + cost > budget:
        return False
    kind = str(action.get("kind", ""))
    selected_kinds = {str(item.get("kind", "")) for item in selected}
    if kind in selected_kinds and kind != "capture_next_best_view":
        return False
    return not _missing_available_prerequisites(action, selected=selected, ranked=ranked)


def _useful_action(action: Mapping[str, Any]) -> bool:
    return any(
        float(action.get(key, 0.0) or 0.0) > 0.0
        for key in (
            "expected_quality_delta",
            "expected_editability_delta",
            "expected_uncertainty_reduction",
        )
    )


def _missing_available_prerequisites(
    action: Mapping[str, Any],
    *,
    selected: Sequence[Mapping[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    prerequisites = tuple(str(item) for item in action.get("prerequisites", ()) or ())
    if not prerequisites:
        return ()
    selected_kinds = {str(item.get("kind", "")) for item in selected}
    available_kinds = {str(item.get("kind", "")) for item in ranked}
    return tuple(
        prerequisite
        for prerequisite in prerequisites
        if prerequisite in available_kinds and prerequisite not in selected_kinds
    )


def _ordered_portfolio(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(selected, key=_portfolio_selection_order_key)
    output: list[dict[str, Any]] = []
    for index, action in enumerate(ordered):
        payload = {key: value for key, value in dict(action).items() if key != "_selection_index"}
        payload["portfolio_order"] = index + 1
        output.append(payload)
    return output


def _portfolio_selection_order_key(action: Mapping[str, Any]) -> tuple[int, int, int, float, str]:
    stage_order = {
        "evidence_capture": 1,
        "model_generation": 2,
        "mesh_repair": 3,
        "qa": 4,
    }
    kind_order = {
        "inject_learned_constraints": 1,
        "capture_next_best_view": 2,
        "compile_selected_shape_program": 3,
        "extract_guarded_sdf_proxy": 4,
        "run_finite_difference_refinement_probe": 5,
        "apply_guarded_retopology_plan": 6,
    }
    stage = str(action.get("stage_hint", "")) or _default_stage(str(action.get("kind", "")))
    return (
        stage_order.get(stage, 99),
        kind_order.get(str(action.get("kind", "")), 50),
        int(action.get("_selection_index", 0) or 0),
        -float(action.get("score", 0.0) or 0.0),
        str(action.get("action_id", "")),
    )


def _dedupe_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for action in actions:
        kind = str(action.get("kind", ""))
        source = str(action.get("source", ""))
        payload = action.get("payload") if isinstance(action.get("payload"), Mapping) else {}
        key = (kind, str(payload.get("view_id", payload.get("selected_candidate_id", source))))
        scored = dict(action)
        current = best.get(key)
        if current is None or _raw_action_value(scored) > _raw_action_value(current):
            best[key] = scored
    return list(best.values())


def _portfolio_totals(selected: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not selected:
        return {
            "quality_delta": 0.0,
            "editability_delta": 0.0,
            "uncertainty_reduction": 0.0,
            "risk": 0.0,
        }
    quality = sum(max(0.0, float(item.get("expected_quality_delta", 0.0) or 0.0)) for item in selected)
    editability = sum(max(0.0, float(item.get("expected_editability_delta", 0.0) or 0.0)) for item in selected)
    uncertainty = sum(max(0.0, float(item.get("expected_uncertainty_reduction", 0.0) or 0.0)) for item in selected)
    risk = sum(float(item.get("risk", 0.0) or 0.0) for item in selected) / len(selected)
    return {
        "quality_delta": bounded(quality, 0.0, 0.45),
        "editability_delta": bounded(editability, 0.0, 0.45),
        "uncertainty_reduction": bounded(uncertainty, 0.0, 0.75),
        "risk": bounded(risk),
    }


def _portfolio_gates(
    selected: Sequence[Mapping[str, Any]],
    *,
    baseline: Mapping[str, float],
) -> list[dict[str, Any]]:
    gates = []
    for action in selected:
        gate_floor = min(
            0.95,
            max(0.45, baseline["min_iou"] - 0.01 + max(0.0, float(action.get("expected_quality_delta", 0.0) or 0.0)) * 0.35),
        )
        gates.append(
            {
                "action_id": action.get("action_id", ""),
                "kind": action.get("kind", ""),
                "minimum_min_view_iou": gate_floor,
                "maximum_risk": max(0.2, float(action.get("risk", 0.0) or 0.0) + 0.08),
                "required_guards": list(action.get("guards", ()) or ()),
                "next_step": f"{action.get('kind', '')}: run guarded evidence and rerank without changing pass/fail contract",
            }
        )
    return gates


def _rejection_reason(
    action: Mapping[str, Any],
    *,
    selected: Sequence[Mapping[str, Any]],
    budget: float,
    ranked: Sequence[Mapping[str, Any]],
) -> str:
    selected_kinds = {str(item.get("kind", "")) for item in selected}
    selected_cost = sum(float(item.get("cost", 0.0) or 0.0) for item in selected)
    if not _useful_action(action):
        return "no_positive_delta"
    if _missing_available_prerequisites(action, selected=selected, ranked=ranked):
        return "blocked_prerequisite"
    if str(action.get("kind", "")) in selected_kinds and str(action.get("kind", "")) != "capture_next_best_view":
        return "duplicate_kind"
    if selected_cost + float(action.get("cost", 1.0) or 1.0) > budget:
        return "budget_exhausted"
    return "lower_rank"


def _ranking_priors_from_results(
    prior_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    priors: dict[tuple[str, str], dict[str, Any]] = {}
    for result in prior_results:
        if result.get("status") != "ran":
            continue
        evidence = _evidence(result)
        rows = evidence.get("ranking_priors")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            target = str(row.get("target_experiment", ""))
            kind = str(row.get("constraint_kind", ""))
            if not target:
                continue
            key = (target, kind)
            weight = bounded(float(row.get("rank_weight", 0.0) or 0.0), 0.0, 0.35)
            current = priors.get(key)
            if current is None:
                priors[key] = {
                    "target_experiment": target,
                    "constraint_kind": kind,
                    "rank_weight": weight,
                    "strength": str(row.get("strength", "soft")),
                    "support_count": 1,
                }
                continue
            current["rank_weight"] = max(float(current.get("rank_weight", 0.0) or 0.0), weight)
            current["support_count"] = int(current.get("support_count", 1) or 1) + 1
            if row.get("strength") == "hard":
                current["strength"] = "hard"
    return sorted(
        priors.values(),
        key=lambda item: (
            float(item.get("rank_weight", 0.0) or 0.0),
            str(item.get("target_experiment", "")),
            str(item.get("constraint_kind", "")),
        ),
        reverse=True,
    )


def _ranking_prior_bonus(
    action: Mapping[str, Any],
    ranking_priors: Sequence[Mapping[str, Any]],
) -> float:
    source = str(action.get("source", ""))
    kind = str(action.get("kind", ""))
    bonus = 0.0
    for prior in ranking_priors:
        target = str(prior.get("target_experiment", ""))
        constraint_kind = str(prior.get("constraint_kind", ""))
        weight = float(prior.get("rank_weight", 0.0) or 0.0)
        if str(prior.get("strength", "soft")) == "hard":
            weight *= 1.15
        if target == source:
            bonus += weight
        elif constraint_kind == "require_additional_view" and kind == "capture_next_best_view":
            bonus += weight * 0.7
        elif constraint_kind == "prioritize_boundary_refinement" and "sdf" in kind:
            bonus += weight * 0.65
        elif constraint_kind == "prefer_topology_repair" and "retopology" in kind:
            bonus += weight * 0.65
    return bounded(bonus, 0.0, 0.35)


def _dependency_edges(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    by_kind: dict[str, Mapping[str, Any]] = {}
    for action in selected:
        kind = str(action.get("kind", ""))
        by_kind.setdefault(kind, action)

    def add(
        before_kind: str,
        after_action: Mapping[str, Any],
        *,
        reason: str,
        dependency_type: str = "evidence",
    ) -> None:
        before = by_kind.get(before_kind)
        if before is None:
            return
        before_id = str(before.get("action_id", ""))
        after_id = str(after_action.get("action_id", ""))
        if not before_id or not after_id or before_id == after_id:
            return
        edge = {
            "before": before_id,
            "after": after_id,
            "before_kind": before_kind,
            "after_kind": str(after_action.get("kind", "")),
            "type": dependency_type,
            "reason": reason,
        }
        if edge not in edges:
            edges.append(edge)

    for action in selected:
        prerequisites = action.get("prerequisites")
        if isinstance(prerequisites, Sequence) and not isinstance(prerequisites, (str, bytes)):
            for prerequisite in prerequisites:
                add(
                    str(prerequisite),
                    action,
                    reason="declared sidecar action prerequisite",
                    dependency_type="declared",
                )
    capture = by_kind.get("capture_next_best_view")
    if capture is not None:
        for action in selected:
            if str(action.get("stage_hint", "")) == "model_generation":
                add(
                    "capture_next_best_view",
                    action,
                    reason="new view evidence should rerank model-generation actions",
                    dependency_type="rerank",
                )
    constraints = by_kind.get("inject_learned_constraints")
    if constraints is not None:
        for action in selected:
            if str(action.get("stage_hint", "")) in {"model_generation", "mesh_repair"}:
                add(
                    "inject_learned_constraints",
                    action,
                    reason="learned constraints must be explicit before downstream search",
                    dependency_type="constraint",
                )
    if by_kind.get("extract_guarded_sdf_proxy") is not None:
        for action in selected:
            if str(action.get("kind", "")) == "apply_guarded_retopology_plan":
                add(
                    "extract_guarded_sdf_proxy",
                    action,
                    reason="retopology QA should consume the extracted SDF proxy",
                    dependency_type="artifact",
                )
    if by_kind.get("compile_selected_shape_program") is not None:
        for action in selected:
            if str(action.get("kind", "")) == "run_finite_difference_refinement_probe":
                add(
                    "compile_selected_shape_program",
                    action,
                    reason="finite-difference updates are safest against a compiled editable program",
                    dependency_type="artifact",
                )
    return edges


def _execution_plan(
    selected: Sequence[Mapping[str, Any]],
    dependency_edges: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stage_order = {
        "evidence_capture": 1,
        "model_generation": 2,
        "mesh_repair": 3,
        "qa": 4,
    }
    stage_titles = {
        "evidence_capture": "capture_and_constraints",
        "model_generation": "candidate_generation_and_refinement",
        "mesh_repair": "editable_mesh_repair",
        "qa": "contract_preserving_qa",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for action in sorted(selected, key=lambda item: int(item.get("portfolio_order", 0) or 0)):
        stage = str(action.get("stage_hint", "")) or _default_stage(str(action.get("kind", "")))
        grouped.setdefault(stage, []).append(_action_plan_item(action, dependency_edges))
    plan = [
        {
            "stage": stage_titles.get(stage, stage),
            "order": stage_order.get(stage, 99),
            "actions": actions,
            "exit_rule": _stage_exit_rule(stage),
        }
        for stage, actions in grouped.items()
    ]
    if gates:
        plan.append(
            {
                "stage": "contract_preserving_qa",
                "order": 100,
                "actions": [
                    {
                        "action_id": str(gate.get("action_id", "")),
                        "kind": str(gate.get("kind", "")),
                        "required_guards": list(gate.get("required_guards", ()) or ()),
                        "minimum_min_view_iou": float(gate.get("minimum_min_view_iou", 0.0) or 0.0),
                    }
                    for gate in gates
                ],
                "exit_rule": "sidecars may update evidence and ranking only; backend pass/fail remains authoritative",
            }
        )
    return sorted(plan, key=lambda item: (int(item["order"]), str(item["stage"])))


def _action_plan_item(
    action: Mapping[str, Any],
    dependency_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    action_id = str(action.get("action_id", ""))
    inbound = [
        str(edge.get("before", ""))
        for edge in dependency_edges
        if str(edge.get("after", "")) == action_id
    ]
    payload = action.get("payload") if isinstance(action.get("payload"), Mapping) else {}
    return {
        "action_id": action_id,
        "kind": str(action.get("kind", "")),
        "source": str(action.get("source", "")),
        "portfolio_order": int(action.get("portfolio_order", 0) or 0),
        "cost": float(action.get("cost", 0.0) or 0.0),
        "risk": float(action.get("risk", 0.0) or 0.0),
        "expected_quality_delta": float(action.get("expected_quality_delta", 0.0) or 0.0),
        "depends_on": inbound,
        "payload_keys": sorted(str(key) for key in payload)[:12],
    }


def _risk_register(
    selected: Sequence[Mapping[str, Any]],
    *,
    gates: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, float],
    prior_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    gate_by_action = {str(gate.get("action_id", "")): gate for gate in gates}
    for action in selected:
        risk = float(action.get("risk", 0.0) or 0.0)
        if risk < 0.20:
            continue
        action_id = str(action.get("action_id", ""))
        gate = gate_by_action.get(action_id, {})
        guards = list(action.get("guards", ()) or ())
        risks.append(
            {
                "risk_id": f"portfolio_action:{action_id}",
                "level": "high" if risk >= 0.40 else "medium",
                "kind": str(action.get("kind", "")),
                "risk": risk,
                "trigger": "action risk exceeds advisory threshold",
                "mitigation": guards[0] if guards else str(gate.get("next_step", "rerun guarded QA")),
            }
        )
    if baseline["min_iou"] < 0.60:
        risks.append(
            {
                "risk_id": "baseline:min_view_iou",
                "level": "medium",
                "kind": "baseline_quality",
                "risk": 1.0 - baseline["min_iou"],
                "trigger": "baseline min-view IoU is weak",
                "mitigation": "treat sidecar deltas as ranking evidence until required views rerender cleanly",
            }
        )
    for result in prior_results:
        evidence = _evidence(result)
        experiment_id = str(result.get("experiment_id", ""))
        if experiment_id == "shape_grammar_search":
            summary = evidence.get("validation_summary")
            if isinstance(summary, Mapping) and int(summary.get("invalid_count", 0) or 0) > 0:
                risks.append(
                    {
                        "risk_id": "shape_grammar:invalid_candidates",
                        "level": "medium",
                        "kind": "compile_validation",
                        "risk": bounded(int(summary.get("invalid_count", 0) or 0) / max(1, int(summary.get("candidate_count", 1) or 1))),
                        "trigger": "grammar beam includes invalid candidate programs",
                        "mitigation": "compile only selected_compile_plan entries with compile_ready=true",
                    }
                )
        if experiment_id == "editable_retopology":
            qa = evidence.get("qa")
            if isinstance(qa, Mapping) and qa.get("human_review_required"):
                risks.append(
                    {
                        "risk_id": "retopology:manual_review",
                        "level": "high",
                        "kind": "topology_repair",
                        "risk": float(qa.get("silhouette_risk", 0.4) or 0.4),
                        "trigger": "retopology sidecar marked human_review_required",
                        "mitigation": "do not apply repair phase without reviewer approval and render QA",
                    }
                )
        if experiment_id == "active_view_planning":
            ambiguity = float(evidence.get("ambiguity_index", 0.0) or 0.0)
            if ambiguity > 0.30:
                risks.append(
                    {
                        "risk_id": "active_view:ambiguity",
                        "level": "medium",
                        "kind": "view_evidence",
                        "risk": ambiguity,
                        "trigger": "active-view ambiguity remains elevated",
                        "mitigation": "capture the first sequence_plan view before committing downstream repairs",
                    }
                )
    if not risks:
        risks.append(
            {
                "risk_id": "portfolio:low_risk",
                "level": "low",
                "kind": "research_sidecar",
                "risk": 0.05,
                "trigger": "no selected action exceeded advisory risk threshold",
                "mitigation": "keep sidecar evidence contract-preserving",
            }
        )
    return risks[:12]


def _stage_exit_rule(stage: str) -> str:
    if stage == "evidence_capture":
        return "new labels or views are attached as explicit evidence before reranking"
    if stage == "model_generation":
        return "candidate artifacts compile or extract and clear local guard metrics"
    if stage == "mesh_repair":
        return "topology/editability improve without render metric regression"
    return "backend success/failure contract remains unchanged"


def _default_stage(kind: str) -> str:
    if kind in {"capture_next_best_view", "inject_learned_constraints"}:
        return "evidence_capture"
    if "retopology" in kind:
        return "mesh_repair"
    if kind.startswith("no_op") or "qa" in kind:
        return "qa"
    return "model_generation"


def _guard_to_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    metric = str(value.get("metric", "guard"))
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    reason = str(value.get("reason", "")).strip()
    threshold = ""
    if minimum is not None:
        threshold = f" >= {minimum}"
    elif maximum is not None:
        threshold = f" <= {maximum}"
    suffix = f": {reason}" if reason else ""
    return f"{metric}{threshold}{suffix}"


def _baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"min_iou": 0.55, "topology": 0.75, "editability": 0.55}
    min_iou = max(
        (
            row_metric(
                row,
                "render.min_view_iou",
                "min_view_iou",
                "area_iou_min",
                "backend.area_iou_min",
                default=0.0,
            )
            for row in rows
        ),
        default=0.0,
    )
    topology = max(
        (row_metric(row, "topology.score", "topology_score", default=0.0) for row in rows),
        default=0.0,
    )
    editability = max(
        (
            row_metric(row, "editability.qa_score", "editability_score", default=0.0)
            for row in rows
        ),
        default=0.0,
    )
    return {
        "min_iou": bounded(min_iou or 0.55),
        "topology": bounded(topology or 0.75),
        "editability": bounded(editability or 0.55),
    }


def _raw_action_value(action: Mapping[str, Any]) -> float:
    return (
        float(action.get("expected_quality_delta", 0.0) or 0.0)
        + float(action.get("expected_editability_delta", 0.0) or 0.0) * 0.6
        + float(action.get("expected_uncertainty_reduction", 0.0) or 0.0) * 0.3
        - float(action.get("risk", 0.0) or 0.0) * 0.2
    )


def _evidence(result: Mapping[str, Any]) -> Mapping[str, Any]:
    degradation = result.get("degradation")
    if isinstance(degradation, Mapping):
        evidence = degradation.get("evidence")
        if isinstance(evidence, Mapping):
            return evidence
    evidence = result.get("evidence")
    return evidence if isinstance(evidence, Mapping) else {}
