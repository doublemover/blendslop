"""Validation helpers for refinement parameter catalog coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .apply import known_parameter_name
from .catalog import CONFIG_PARAMETER_PATHS, SPECIAL_PARAMETER_NAMES


@dataclass(frozen=True)
class ParameterCatalogIssue:
    parameter: str
    issue: str
    track: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "track": self.track,
            "issue": self.issue,
        }


def missing_catalog_entries(
    parameters: Iterable[Any],
    *,
    track: str = "",
) -> tuple[ParameterCatalogIssue, ...]:
    issues: list[ParameterCatalogIssue] = []
    for parameter in parameters:
        name = str(getattr(parameter, "name", ""))
        if not name:
            issues.append(ParameterCatalogIssue(name, "missing parameter name", track))
            continue
        if bool(getattr(parameter, "lab_only", False)):
            continue
        if known_parameter_name(name):
            continue
        issues.append(
            ParameterCatalogIssue(
                name,
                "non-lab parameter is missing from refinement parameter catalog",
                track,
            )
        )
    return tuple(issues)


def validate_parameter_catalog(
    *,
    tracks: Iterable[Any],
    config: Any | None = None,
) -> tuple[ParameterCatalogIssue, ...]:
    issues: list[ParameterCatalogIssue] = []
    for track in tracks:
        issues.extend(
            missing_catalog_entries(
                getattr(track, "parameters", ()) or (),
                track=str(getattr(track, "name", "")),
            )
        )
    if config is not None:
        issues.extend(_validate_config_paths(config))
    return tuple(issues)


def _validate_config_paths(config: Any) -> tuple[ParameterCatalogIssue, ...]:
    issues: list[ParameterCatalogIssue] = []
    for parameter, (group_name, attr_name) in CONFIG_PARAMETER_PATHS.items():
        if not hasattr(config, group_name):
            issues.append(
                ParameterCatalogIssue(
                    parameter,
                    f"config group {group_name!r} does not exist",
                )
            )
            continue
        group = getattr(config, group_name)
        if not hasattr(group, attr_name):
            issues.append(
                ParameterCatalogIssue(
                    parameter,
                    f"config attribute {group_name}.{attr_name} does not exist",
                )
            )
    for special in SPECIAL_PARAMETER_NAMES:
        if not special:
            issues.append(ParameterCatalogIssue(special, "empty special parameter"))
    return tuple(issues)
