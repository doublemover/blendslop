"""Adaptive refinement proposal contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import ExperimentVariant, safe_slug


@dataclass(frozen=True)
class RefinementProposal:
    proposal_id: str
    title: str
    hypothesis: str
    expected_win: Mapping[str, str | float] = field(default_factory=dict)
    mode: str = "ensemble"
    validation_mode: str = "backend-status"
    cli_args: tuple[str, ...] = ()
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    priority: int = 50
    risk: str = "medium"
    source_evidence: Mapping[str, Any] = field(default_factory=dict)
    diagnostic_only: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "expected_win": dict(self.expected_win),
            "mode": self.mode,
            "validation_mode": self.validation_mode,
            "cli_args": list(self.cli_args),
            "config_overrides": dict(self.config_overrides),
            "tags": list(self.tags),
            "priority": self.priority,
            "risk": self.risk,
            "source_evidence": dict(self.source_evidence),
            "diagnostic_only": self.diagnostic_only,
        }

    def to_variant(self, *, parent_variant_id: str = "") -> ExperimentVariant:
        variant_id = safe_slug(f"adaptive_{self.proposal_id}")
        return ExperimentVariant(
            variant_id=variant_id,
            label=self.title,
            mode=self.mode,
            validation_mode=self.validation_mode,
            parameters={
                "proposal_id": self.proposal_id,
                "hypothesis": self.hypothesis,
                "priority": self.priority,
                "risk": self.risk,
            },
            cli_args=_complete_cli_args(
                mode=self.mode,
                validation_mode=self.validation_mode,
                cli_args=self.cli_args,
            ),
            config_overrides=self.config_overrides,
            expected_artifacts=("evaluation_bundle", "autopsy_pack"),
            tags=("adaptive",) + self.tags,
            parent_variant_id=parent_variant_id,
            stage="adaptive_refinement",
            diagnostic_only=self.diagnostic_only,
        )


def _complete_cli_args(
    *,
    mode: str,
    validation_mode: str,
    cli_args: Sequence[str],
) -> tuple[str, ...]:
    args = list(str(arg) for arg in cli_args)
    prefix: list[str] = []
    if "--reconstruction-mode" not in args:
        prefix.extend(("--reconstruction-mode", mode))
    if "--validation-mode" not in args:
        prefix.extend(("--validation-mode", validation_mode))
    return tuple(prefix + args)
