"""Primitive grammar definitions for editable shape programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .shape_program import OPERATIONS, PRIMITIVE_TYPES


@dataclass(frozen=True)
class ProductionRule:
    rule_id: str
    output_primitive: str
    operation: str = "add"
    required_evidence: tuple[str, ...] = ()
    default_parameters: Mapping[str, float | int | str | bool] = field(default_factory=dict)
    description: str = ""
    cost: float = 1.0
    editability_gain: float = 0.0

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.output_primitive not in PRIMITIVE_TYPES:
            errors.append(f"unknown output primitive {self.output_primitive!r}")
        if self.operation not in OPERATIONS:
            errors.append(f"unknown operation {self.operation!r}")
        if self.cost < 0:
            errors.append(f"rule {self.rule_id!r} cost must be >= 0")
        return tuple(errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "output_primitive": self.output_primitive,
            "operation": self.operation,
            "required_evidence": list(self.required_evidence),
            "default_parameters": dict(self.default_parameters),
            "description": self.description,
            "cost": self.cost,
            "editability_gain": self.editability_gain,
        }


@dataclass(frozen=True)
class PrimitiveGrammar:
    grammar_id: str
    rules: tuple[ProductionRule, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def rule_ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.rules)

    def rules_for_evidence(self, evidence: Mapping[str, Any]) -> tuple[ProductionRule, ...]:
        available = {
            key
            for key, value in evidence.items()
            if bool(value) and not (isinstance(value, (int, float)) and float(value) <= 0.0)
        }
        return tuple(
            rule
            for rule in self.rules
            if all(item in available for item in rule.required_evidence)
        )

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []
        seen: set[str] = set()
        for rule in self.rules:
            if rule.rule_id in seen:
                errors.append(f"duplicate grammar rule {rule.rule_id!r}")
            seen.add(rule.rule_id)
            errors.extend(rule.validate())
        return tuple(errors)

    def to_dict(self) -> dict[str, object]:
        return {
            "grammar_id": self.grammar_id,
            "rules": [rule.to_dict() for rule in self.rules],
            "metadata": dict(self.metadata),
        }


def default_shape_program_grammar() -> PrimitiveGrammar:
    """Return the built-in editable reconstruction grammar."""

    return PrimitiveGrammar(
        grammar_id="editable_silhouette_grammar_v1",
        rules=(
            ProductionRule(
                "profile_lathe_root",
                "lathe_profile",
                required_evidence=("profile_curve",),
                description="Use the strongest silhouette profile as an editable lathe root.",
                cost=1.0,
                editability_gain=0.2,
            ),
            ProductionRule(
                "bounds_rounded_box_root",
                "rounded_box",
                required_evidence=("bounds",),
                description="Use target bounds as a conservative editable blockout.",
                cost=0.7,
                editability_gain=0.15,
            ),
            ProductionRule(
                "superquadric_proxy_root",
                "superquadric",
                required_evidence=("bounds",),
                default_parameters={"exponent_u": 0.65, "exponent_v": 0.65},
                description="Fit a smooth superquadric proxy for blobby silhouettes.",
                cost=1.25,
                editability_gain=0.18,
            ),
            ProductionRule(
                "multi_interval_residual_parts",
                "ellipsoid",
                required_evidence=("multi_interval_rows",),
                description="Represent repeated profile intervals as editable ellipsoid part hints.",
                cost=1.6,
                editability_gain=0.08,
            ),
            ProductionRule(
                "hole_cutout_hint",
                "residual_mesh_patch",
                operation="difference",
                required_evidence=("hole_rows",),
                description="Preserve silhouette hole evidence as explicit cutout patch hints.",
                cost=1.8,
                editability_gain=0.04,
            ),
            ProductionRule(
                "uncertainty_boundary_patch",
                "plane_patch",
                required_evidence=("uncertainty",),
                description="Attach low-confidence boundary patches for human refinement.",
                cost=1.4,
                editability_gain=0.05,
            ),
        ),
        metadata={
            "representation": "editable primitive program",
            "intended_inputs": ("front", "side", "top", "profile_bands", "bounds"),
            "literature_roots": (
                "visual hull / shape-from-silhouette",
                "superquadric primitive recovery",
                "editable proxy reconstruction",
            ),
        },
    )


def validate_grammar(grammar: PrimitiveGrammar) -> tuple[str, ...]:
    return grammar.validate()
