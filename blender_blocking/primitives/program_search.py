"""Candidate search over editable shape programs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .grammar import PrimitiveGrammar, ProductionRule, default_shape_program_grammar
from .shape_dsl import program_to_dsl
from .shape_program import ResidualPatch, ShapeNode, ShapeProgram, validate_shape_program


@dataclass(frozen=True)
class ShapeProgramCandidate:
    candidate_id: str
    program: ShapeProgram
    score: float
    applied_rules: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "applied_rules": list(self.applied_rules),
            "metrics": dict(self.metrics),
            "diagnostics": dict(self.diagnostics),
            "program": self.program.to_dict(),
            "dsl": program_to_dsl(self.program),
        }


@dataclass(frozen=True)
class ShapeProgramSearchResult:
    selected: ShapeProgramCandidate
    candidates: tuple[ShapeProgramCandidate, ...]
    grammar: PrimitiveGrammar
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected.candidate_id,
            "selected_score": self.selected.score,
            "candidate_count": len(self.candidates),
            "grammar": self.grammar.to_dict(),
            "evidence": dict(self.evidence),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def search_shape_program_candidates(
    seed_program: ShapeProgram,
    *,
    grammar: PrimitiveGrammar | None = None,
    max_candidates: int = 4,
    objective: str = "editable_balanced",
) -> ShapeProgramSearchResult:
    grammar = grammar or default_shape_program_grammar()
    evidence = evidence_from_program(seed_program)
    rules = grammar.rules_for_evidence(evidence)
    candidates = [_candidate("seed", seed_program, (), objective=objective)]
    for rule in rules:
        expanded = _apply_rule(seed_program, rule, evidence)
        if expanded is None:
            continue
        candidates.append(
            _candidate(
                rule.rule_id,
                expanded,
                (rule.rule_id,),
                objective=objective,
                rule=rule,
            )
        )
    candidates = _dedupe_candidates(candidates)
    ranked = tuple(
        sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[
            : max(1, int(max_candidates))
        ]
    )
    return ShapeProgramSearchResult(
        selected=ranked[0],
        candidates=ranked,
        grammar=grammar,
        evidence=evidence,
    )


def evidence_from_program(program: ShapeProgram) -> Mapping[str, Any]:
    profile_curve_rows = 0
    multi_interval_rows = 0
    hole_rows = 0
    has_bounds = bool(program.metadata.get("bounds"))
    has_uncertainty = False
    for node in program.root_nodes:
        curve = node.parameters.get("profile_curve")
        if isinstance(curve, Sequence) and not isinstance(curve, (str, bytes)):
            profile_curve_rows = max(profile_curve_rows, len(curve))
        if bool(node.parameters.get("preserves_multiple_intervals")):
            multi_interval_rows += int(node.parameters.get("band_count", 1) or 1)
        if bool(node.parameters.get("preserves_hole_hints")):
            hole_rows += int(node.parameters.get("band_count", 1) or 1)
    for constraint in program.constraints:
        if constraint.kind == "boundary_uncertainty_weighted_fit":
            has_uncertainty = True
    return {
        "bounds": has_bounds,
        "profile_curve": profile_curve_rows > 0,
        "profile_curve_rows": profile_curve_rows,
        "multi_interval_rows": multi_interval_rows,
        "hole_rows": hole_rows,
        "uncertainty": has_uncertainty,
        "residual_patch_count": program.residual_patch_count(),
        "node_count": program.node_count(),
    }


def _apply_rule(
    program: ShapeProgram,
    rule: ProductionRule,
    evidence: Mapping[str, Any],
) -> ShapeProgram | None:
    if rule.rule_id == "profile_lathe_root":
        return _with_search_metadata(program, rule)
    if rule.rule_id == "bounds_rounded_box_root":
        return _replace_root_primitive(program, rule, "bounds rounded-box proxy")
    if rule.rule_id == "superquadric_proxy_root":
        return _replace_root_primitive(program, rule, "smooth superquadric proxy")
    if rule.rule_id == "multi_interval_residual_parts":
        count = min(4, max(1, int(evidence.get("multi_interval_rows", 1))))
        return _with_part_hints(program, rule, count=count)
    if rule.rule_id == "hole_cutout_hint":
        return _with_cutout_hint(program, rule)
    if rule.rule_id == "uncertainty_boundary_patch":
        return _with_uncertainty_patch(program, rule)
    return None


def _replace_root_primitive(
    program: ShapeProgram,
    rule: ProductionRule,
    name: str,
) -> ShapeProgram:
    size = _size_from_program(program)
    old_root_id = program.root_nodes[0].node_id if program.root_nodes else ""
    new_root_id = f"{rule.rule_id}_00"
    node = ShapeNode(
        node_id=new_root_id,
        operation=rule.operation,
        primitive_type=rule.output_primitive,
        name=name,
        parameters={
            **dict(rule.default_parameters),
            "width_world": size[0],
            "depth_world": size[1],
            "height_world": size[2],
            "source_rule": rule.rule_id,
        },
    )
    constraints = tuple(
        replace(
            constraint,
            target_nodes=tuple(
                new_root_id if target == old_root_id else target
                for target in constraint.target_nodes
            ),
        )
        for constraint in program.constraints
    )
    return replace(
        program,
        root_nodes=(node,) + tuple(program.root_nodes[1:]),
        constraints=constraints,
        metadata={**dict(program.metadata), "selected_rule": rule.rule_id},
    )


def _with_part_hints(
    program: ShapeProgram,
    rule: ProductionRule,
    *,
    count: int,
) -> ShapeProgram:
    size = _size_from_program(program)
    existing = tuple(program.root_nodes)
    new_nodes = []
    for index in range(count):
        offset = (index - (count - 1) / 2.0) / max(1.0, float(count))
        new_nodes.append(
            ShapeNode(
                node_id=f"{rule.rule_id}_{index:02d}",
                operation=rule.operation,
                primitive_type=rule.output_primitive,
                name=f"profile interval part {index + 1}",
                parameters={
                    "radius_x": size[0] * 0.12,
                    "radius_y": size[1] * 0.12,
                    "radius_z": size[2] * 0.18,
                    "offset_x_normalized": offset,
                    "source_rule": rule.rule_id,
                },
            )
        )
    return replace(
        program,
        root_nodes=existing + tuple(new_nodes),
        metadata={**dict(program.metadata), "selected_rule": rule.rule_id},
    )


def _with_cutout_hint(program: ShapeProgram, rule: ProductionRule) -> ShapeProgram:
    patch = ResidualPatch(
        patch_id=f"{rule.rule_id}_00",
        category="hole_cutout",
        source_view=str(program.metadata.get("dominant_profile_view", "")),
        confidence=0.65,
        suggested_node=ShapeNode(
            node_id=f"{rule.rule_id}_node_00",
            operation=rule.operation,
            primitive_type=rule.output_primitive,
            name="editable silhouette hole cutout",
            parameters={"source_rule": rule.rule_id},
        ),
        notes=("Convert high-confidence hole residuals into editable boolean cutouts.",),
    )
    return replace(
        program,
        residual_patches=tuple(program.residual_patches) + (patch,),
        metadata={**dict(program.metadata), "selected_rule": rule.rule_id},
    )


def _with_uncertainty_patch(program: ShapeProgram, rule: ProductionRule) -> ShapeProgram:
    patch = ResidualPatch(
        patch_id=f"{rule.rule_id}_00",
        category="uncertain_boundary",
        source_view="",
        confidence=0.5,
        suggested_node=ShapeNode(
            node_id=f"{rule.rule_id}_node_00",
            operation=rule.operation,
            primitive_type=rule.output_primitive,
            name="uncertain boundary edit patch",
            parameters={"source_rule": rule.rule_id},
        ),
        notes=("Low-confidence boundary evidence should remain editable, not baked.",),
    )
    return replace(
        program,
        residual_patches=tuple(program.residual_patches) + (patch,),
        metadata={**dict(program.metadata), "selected_rule": rule.rule_id},
    )


def _with_search_metadata(program: ShapeProgram, rule: ProductionRule) -> ShapeProgram:
    return replace(
        program,
        metadata={**dict(program.metadata), "selected_rule": rule.rule_id},
    )


def _candidate(
    candidate_id: str,
    program: ShapeProgram,
    applied_rules: tuple[str, ...],
    *,
    objective: str,
    rule: ProductionRule | None = None,
) -> ShapeProgramCandidate:
    validation_errors = validate_shape_program(program)
    metrics = _candidate_metrics(program, validation_errors, rule=rule)
    score = _score_metrics(metrics, objective=objective)
    return ShapeProgramCandidate(
        candidate_id=candidate_id,
        program=program,
        score=score,
        applied_rules=applied_rules,
        metrics=metrics,
        diagnostics={"validation_errors": list(validation_errors)},
    )


def _candidate_metrics(
    program: ShapeProgram,
    validation_errors: Sequence[str],
    *,
    rule: ProductionRule | None,
) -> Mapping[str, float]:
    node_count = max(1, program.node_count())
    residual_count = program.residual_patch_count()
    editable_count = sum(1 for node in program.root_nodes if node.editable)
    primitive_diversity = len({node.primitive_type for node in program.root_nodes})
    editability = editable_count / float(node_count)
    complexity = min(1.0, (node_count + residual_count * 0.5) / 24.0)
    grammar_gain = 0.0 if rule is None else float(rule.editability_gain)
    validation_penalty = min(1.0, len(validation_errors) * 0.25)
    return {
        "editability": editability,
        "complexity": complexity,
        "primitive_diversity": min(1.0, primitive_diversity / 6.0),
        "residual_patch_count": float(residual_count),
        "grammar_gain": grammar_gain,
        "validation_penalty": validation_penalty,
    }


def _score_metrics(metrics: Mapping[str, float], *, objective: str) -> float:
    editability = float(metrics.get("editability", 0.0))
    diversity = float(metrics.get("primitive_diversity", 0.0))
    complexity = float(metrics.get("complexity", 0.0))
    grammar_gain = float(metrics.get("grammar_gain", 0.0))
    validation_penalty = float(metrics.get("validation_penalty", 0.0))
    residual_bonus = min(0.12, float(metrics.get("residual_patch_count", 0.0)) * 0.02)
    if objective == "minimal":
        return editability + grammar_gain - complexity - validation_penalty
    if objective == "part_aware":
        return editability + diversity * 0.4 + residual_bonus + grammar_gain - validation_penalty
    return editability + diversity * 0.2 + residual_bonus + grammar_gain - complexity * 0.25 - validation_penalty


def _dedupe_candidates(
    candidates: Sequence[ShapeProgramCandidate],
) -> tuple[ShapeProgramCandidate, ...]:
    by_signature: dict[str, ShapeProgramCandidate] = {}
    for candidate in candidates:
        signature = _program_signature(candidate.program)
        current = by_signature.get(signature)
        if current is None or candidate.score > current.score:
            by_signature[signature] = candidate
    return tuple(by_signature.values())


def _program_signature(program: ShapeProgram) -> str:
    primitives = tuple(node.primitive_type for node in program.root_nodes)
    rules = (str(program.metadata.get("selected_rule", "")),)
    return repr((primitives, program.node_count(), program.residual_patch_count(), rules))


def _size_from_program(program: ShapeProgram) -> tuple[float, float, float]:
    for node in program.root_nodes:
        width = node.parameters.get("width_world")
        depth = node.parameters.get("depth_world")
        height = node.parameters.get("height_world")
        if width is not None and depth is not None and height is not None:
            return (float(width), float(depth), float(height))
    return (1.0, 1.0, 1.0)
