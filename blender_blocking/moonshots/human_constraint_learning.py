"""Human-in-the-loop constraint learning moonshot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    MoonshotExperiment,
    MoonshotRequest,
    MoonshotResult,
    bundle_result,
    error_result,
    skipped_result,
)
from .papers import VISUAL_HULL
from .support import (
    bounded,
    candidate_rows,
    per_view_boundary_iou,
    row_failures,
    row_metric,
    target_signals,
)


EXPERIMENT = MoonshotExperiment(
    experiment_id="human_constraint_learning",
    title="Human constraint learning from edit and rating feedback",
    subsystem="constraints",
    hypothesis=(
        "Small human labels about intended symmetry, thin structures, and editable parts "
        "can be converted into reusable constraints that improve future reconstructions."
    ),
    expected_wins={
        "quality": "recover intended semantics that silhouettes alone cannot disambiguate",
        "throughput": "reduce repeated manual tuning by turning edits into priors",
    },
    required_inputs=("candidate_rankings", "human_labels", "constraint_schema"),
    validation_metrics=("label_satisfaction", "candidate_rank_delta", "editability_index"),
    papers=(VISUAL_HULL,),
)


def run(request: MoonshotRequest) -> MoonshotResult:
    try:
        rows = _input_rows(request)
        labels = _labels(request)
        signals = target_signals(request)
        if not rows and not labels and not any(bool(group.get("available")) for group in signals.values()):
            return skipped_result(
                request,
                reason="constraint learning needs failed rows, target signals, or human labels",
                next_steps=("run after failed candidates or review labels are available",),
            )
        return _learn_constraints(request, rows=rows, labels=labels, signals=signals)
    except Exception as exc:
        return error_result(request, error=f"{type(exc).__name__}: {exc}")


def _learn_constraints(
    request: MoonshotRequest,
    *,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Mapping[str, Any]],
) -> MoonshotResult:
    constraints = []
    constraints.extend(_constraints_from_rows(rows))
    constraints.extend(_constraints_from_labels(labels))
    constraints.extend(_constraints_from_signals(signals))
    constraints = _dedupe_constraints(constraints)
    satisfaction = _constraint_satisfaction(rows, constraints)
    examples = _training_examples(rows, labels, constraints)
    priors = _ranking_priors(constraints)
    evidence = {
        "input_row_count": len(rows),
        "label_count": len(labels),
        "constraints": constraints,
        "satisfaction": satisfaction,
        "training_examples": examples,
        "ranking_priors": priors,
        "signal_summary": {key: dict(value) for key, value in signals.items()},
    }
    metrics = {
        "ran": 1.0,
        "constraint_count": float(len(constraints)),
        "label_count": float(len(labels)),
        "failed_row_count": float(len(rows)),
        "mean_confidence": (
            sum(float(item.get("confidence", 0.0) or 0.0) for item in constraints)
            / max(1, len(constraints))
        ),
        "constraint_satisfaction": satisfaction["mean_satisfaction"],
        "training_example_count": float(len(examples)),
        "hard_constraint_count": float(sum(1 for item in constraints if item.get("strength") == "hard")),
        "ranking_prior_count": float(len(priors)),
    }
    return bundle_result(
        request,
        status="ran",
        metrics=metrics,
        evidence=evidence,
        artifact_name="human-constraints.json",
        next_steps=(
            "feed learned constraints into the next refinement plan as explicit config records",
            "separate hard constraints from soft ranking priors during promotion",
        ),
    )


def _input_rows(request: MoonshotRequest) -> tuple[Mapping[str, Any], ...]:
    rows = list(candidate_rows(request.candidate))
    config = request.config or {}
    if isinstance(config, Mapping):
        for key in ("failed_rows", "candidate_rankings", "rows"):
            value = config.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(item for item in value if isinstance(item, Mapping))
    return tuple(rows)


def _labels(request: MoonshotRequest) -> tuple[Mapping[str, Any], ...]:
    config = request.config or {}
    labels = config.get("human_labels") if isinstance(config, Mapping) else None
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        return tuple(item for item in labels if isinstance(item, Mapping))
    return ()


def _constraints_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    constraints = []
    for row in rows:
        autopsy = row.get("autopsy")
        category = str(autopsy.get("category", "")) if isinstance(autopsy, Mapping) else ""
        failures = set(row_failures(row))
        min_iou = row_metric(row, "render.min_view_iou", "min_view_iou", default=0.0)
        topology = row_metric(row, "topology.score", "topology_score", default=1.0)
        boundary = per_view_boundary_iou(row)
        weakest_boundary = min(boundary.values()) if boundary else 1.0
        if category in {"catastrophic_view_failure", "hybrid_loft_one_view_failure"} or (0.0 < min_iou < 0.45):
            constraints.append(
                _constraint(
                    "require_additional_view",
                    confidence=0.76,
                    source="failed_row",
                    parameters={"min_view_iou": min_iou, "category": category},
                )
            )
        if category == "axis_or_transform_suspect" or "axis_or_transform_suspect" in failures:
            constraints.append(
                _constraint(
                    "calibrate_axis_and_origin",
                    confidence=0.74,
                    source="failed_row",
                    strength="hard",
                    parameters={"category": category, "variant_id": str(row.get("variant_id", ""))},
                )
            )
        if weakest_boundary < 0.35 or "proxy_render_disagreement" in failures:
            constraints.append(
                _constraint(
                    "prioritize_boundary_refinement",
                    confidence=0.70,
                    source="failed_row",
                    parameters={
                        "weakest_boundary_iou": weakest_boundary,
                        "variant_id": str(row.get("variant_id", "")),
                    },
                )
            )
        if topology < 0.75:
            constraints.append(
                _constraint(
                    "prefer_topology_repair",
                    confidence=0.72,
                    source="failed_row",
                    strength="hard" if topology < 0.5 else "soft",
                    parameters={"topology_score": topology},
                )
            )
        if row_metric(row, "editability.qa_score", "editability_score", default=1.0) < 0.55:
            constraints.append(
                _constraint(
                    "prefer_editable_primitives",
                    confidence=0.68,
                    source="failed_row",
                    parameters={"variant_id": str(row.get("variant_id", ""))},
                )
            )
    return constraints


def _constraints_from_labels(labels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    constraints = []
    for label in labels:
        name = str(label.get("label", "")).strip().lower()
        score = float(label.get("score", 3) or 3)
        confidence = bounded(score / 5.0)
        if name in {"sculptable", "useful_proxy"}:
            constraints.append(
                _constraint(
                    "prefer_editable_primitives",
                    confidence=confidence,
                    source="human_label",
                    parameters={"label": name, "notes": str(label.get("notes", ""))},
                )
            )
        elif name in {"too_blobby", "underfit_silhouette"}:
            constraints.append(
                _constraint(
                    "increase_part_salience",
                    confidence=confidence,
                    source="human_label",
                    parameters={"label": name},
                )
            )
        elif name in {"bad_topology"}:
            constraints.append(
                _constraint(
                    "prefer_topology_repair",
                    confidence=confidence,
                    source="human_label",
                    parameters={"label": name},
                )
            )
        elif name in {"symmetric", "needs_symmetry", "mirrorable"}:
            constraints.append(
                _constraint(
                    "preserve_symmetry",
                    confidence=confidence,
                    source="human_label",
                    parameters={"axis": str(label.get("axis", "auto")), "label": name},
                )
            )
        elif name in {"thin_parts", "missing_thin_parts", "spindly"}:
            constraints.append(
                _constraint(
                    "preserve_thin_structures",
                    confidence=confidence,
                    source="human_label",
                    parameters={"label": name, "notes": str(label.get("notes", ""))},
                )
            )
    return constraints


def _constraints_from_signals(signals: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    constraints = []
    profile = signals["profile"]
    if int(profile.get("hole_count", 0) or 0) > 0:
        constraints.append(
            _constraint(
                "preserve_profile_holes",
                confidence=0.66,
                source="target_signal",
                parameters={"hole_count": int(profile.get("hole_count", 0) or 0)},
            )
        )
    interval_count = int(profile.get("interval_count", 0) or 0)
    band_samples = max(1, int(profile.get("band_samples", 1) or 1))
    if interval_count / band_samples > 1.2:
        constraints.append(
            _constraint(
                "preserve_thin_structures",
                confidence=0.63,
                source="target_signal",
                parameters={"intervals_per_band": interval_count / band_samples},
            )
        )
    if float(profile.get("complexity", 0.0) or 0.0) > 0.55:
        constraints.append(
            _constraint(
                "increase_part_salience",
                confidence=0.62,
                source="target_signal",
                parameters={"profile_complexity": float(profile.get("complexity", 0.0) or 0.0)},
            )
        )
    uncertainty = signals["uncertainty"]
    if float(uncertainty.get("overall_boundary_uncertainty_mean", 0.0) or 0.0) > 0.2:
        constraints.append(
            _constraint(
                "require_additional_view",
                confidence=0.64,
                source="target_signal",
                parameters={
                    "boundary_uncertainty_mean": float(
                        uncertainty.get("overall_boundary_uncertainty_mean", 0.0) or 0.0
                    )
                },
            )
        )
    topology = signals["topology"]
    if float(topology.get("complexity", 0.0) or 0.0) > 0.45:
        constraints.append(
            _constraint(
                "prefer_part_decomposition",
                confidence=0.61,
                source="target_signal",
                parameters={"topology_complexity": float(topology.get("complexity", 0.0) or 0.0)},
            )
        )
    return constraints


def _constraint(
    kind: str,
    *,
    confidence: float,
    source: str,
    parameters: Mapping[str, Any],
    strength: str = "soft",
) -> dict[str, Any]:
    return {
        "schema_version": "learned_constraint_v1",
        "kind": kind,
        "confidence": bounded(confidence),
        "source": source,
        "strength": strength,
        "support_count": 1,
        "parameters": dict(parameters),
    }


def _dedupe_constraints(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_kind: dict[str, dict[str, Any]] = {}
    for item in items:
        kind = str(item.get("kind", ""))
        current = by_kind.get(kind)
        if current is None:
            by_kind[kind] = dict(item)
            continue
        current_conf = float(current.get("confidence", 0.0) or 0.0)
        item_conf = float(item.get("confidence", 0.0) or 0.0)
        current["confidence"] = max(current_conf, item_conf)
        current["support_count"] = int(current.get("support_count", 1) or 1) + 1
        if item.get("strength") == "hard":
            current["strength"] = "hard"
        sources = set(str(current.get("source", "")).split("+"))
        sources.add(str(item.get("source", "")))
        current["source"] = "+".join(sorted(source for source in sources if source))
    return [by_kind[key] for key in sorted(by_kind)]


def _constraint_satisfaction(
    rows: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not constraints:
        return {"mean_satisfaction": 1.0, "items": []}
    items = []
    for constraint in constraints:
        kind = str(constraint.get("kind", ""))
        if kind == "prefer_topology_repair":
            value = max(
                (
                    row_metric(row, "topology.score", "topology_score", default=0.75)
                    for row in rows
                ),
                default=0.75,
            )
        elif kind == "prefer_editable_primitives":
            value = max(
                (
                    row_metric(row, "editability.qa_score", "editability_score", default=0.6)
                    for row in rows
                ),
                default=0.6,
            )
        elif kind == "prioritize_boundary_refinement":
            boundary_values = []
            for row in rows:
                boundary_values.extend(per_view_boundary_iou(row).values())
            value = max(boundary_values) if boundary_values else 0.5
        elif kind == "calibrate_axis_and_origin":
            value = 0.25 if rows else 0.75
        else:
            value = 0.5 if rows else 0.75
        items.append({"kind": kind, "satisfaction": bounded(value)})
    mean = sum(float(item["satisfaction"]) for item in items) / max(1, len(items))
    return {"mean_satisfaction": bounded(mean), "items": items}


def _training_examples(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    examples = []
    constraint_kinds = [str(item.get("kind", "")) for item in constraints]
    for row in rows:
        failures = row_failures(row)
        examples.append(
            {
                "schema_version": "learned_constraint_example_v1",
                "variant_id": str(row.get("variant_id", "")),
                "mode": str(row.get("mode", "")),
                "failures": list(failures),
                "render_min_view_iou": row_metric(row, "render.min_view_iou", "min_view_iou", default=0.0),
                "topology_score": row_metric(row, "topology.score", "topology_score", default=0.0),
                "suggested_constraints": constraint_kinds,
            }
        )
    for label in labels:
        examples.append(
            {
                "schema_version": "learned_constraint_example_v1",
                "label": str(label.get("label", "")),
                "score": float(label.get("score", 0.0) or 0.0),
                "notes": str(label.get("notes", "")),
                "suggested_constraints": constraint_kinds,
            }
        )
    return examples[:24]


def _ranking_priors(constraints: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    weights = {
        "require_additional_view": ("active_view_planning", 0.18),
        "prioritize_boundary_refinement": ("implicit_sdf_proxy", 0.14),
        "prefer_topology_repair": ("editable_retopology", 0.16),
        "prefer_editable_primitives": ("shape_grammar_search", 0.13),
        "calibrate_axis_and_origin": ("active_view_planning", 0.12),
        "preserve_thin_structures": ("shape_grammar_search", 0.10),
        "preserve_profile_holes": ("shape_grammar_search", 0.09),
        "prefer_part_decomposition": ("shape_grammar_search", 0.08),
    }
    priors = []
    for constraint in constraints:
        kind = str(constraint.get("kind", ""))
        if kind not in weights:
            continue
        target, weight = weights[kind]
        confidence = float(constraint.get("confidence", 0.0) or 0.0)
        priors.append(
            {
                "target_experiment": target,
                "constraint_kind": kind,
                "rank_weight": bounded(weight * confidence, 0.0, 0.25),
                "strength": str(constraint.get("strength", "soft")),
            }
        )
    return sorted(priors, key=lambda item: float(item["rank_weight"]), reverse=True)
