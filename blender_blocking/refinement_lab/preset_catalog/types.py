from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..contracts import ParameterSpec

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class SuitePreset:
    name: str
    source: str
    description: str
    synthetic_suites: tuple[str, ...] = ()
    reference_paths: Mapping[str, Path] = field(default_factory=dict)
    default_seed: int = 1234
    default_count: int | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "synthetic_suites": list(self.synthetic_suites),
            "reference_paths": {
                key: path.as_posix() for key, path in self.reference_paths.items()
            },
            "default_seed": self.default_seed,
            "default_count": self.default_count,
            "tags": list(self.tags),
        }

@dataclass(frozen=True)
class TrackPreset:
    name: str
    description: str
    modes: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    default_validation_mode: str = "render-iou"
    default_search: str = "grid"
    default_objective: str = "quality_win"
    max_runs_hint: int | None = None
    force_bounds_debug: bool = False
    force_autopsy: bool = True
    force_overlays: bool = True
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "modes": list(self.modes),
            "default_validation_mode": self.default_validation_mode,
            "default_search": self.default_search,
            "default_objective": self.default_objective,
            "max_runs_hint": self.max_runs_hint,
            "force_bounds_debug": self.force_bounds_debug,
            "force_autopsy": self.force_autopsy,
            "force_overlays": self.force_overlays,
            "tags": list(self.tags),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

def _p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    value_type: str = "string",
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        cli_arg=cli_arg,
        config_path=config_path,
        value_type=value_type,
        values=tuple(values),
        group=group,
        description=description,
    )

def _csv_p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return _p(
        name,
        cli_arg,
        values,
        value_type="csv",
        config_path=config_path,
        group=group,
        description=description,
    )

def _json_p(
    name: str,
    cli_arg: str,
    values: Sequence[object],
    *,
    config_path: str | None = None,
    group: str = "",
    description: str = "",
) -> ParameterSpec:
    return _p(
        name,
        cli_arg,
        values,
        value_type="json",
        config_path=config_path,
        group=group,
        description=description,
    )

_MASK_PARAMETERS = (
    _p(
        "ref_polarity",
        "--ref-polarity",
        ("auto", "dark_foreground", "light_foreground"),
        group="silhouette",
    ),
    _p(
        "ref_gray_threshold",
        "--ref-gray-threshold",
        (96, 112, 128, 144, 160),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_morph_close",
        "--ref-morph-close",
        (0, 3, 5, 7),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_morph_open",
        "--ref-morph-open",
        (0, 1, 3),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_min_component_area_px",
        "--ref-min-component-area-px",
        (0, 16, 64, 256),
        value_type="int",
        group="silhouette",
    ),
    _p(
        "ref_fill_holes",
        "--ref-fill-holes",
        (True, False),
        value_type="bool",
        group="silhouette",
    ),
    _p(
        "ref_largest_component",
        "--ref-largest-component",
        (True, False),
        value_type="bool",
        group="silhouette",
    ),
    _p(
        "canonical_padding_frac",
        "--canonical-padding-frac",
        (0.06, 0.08, 0.10, 0.12),
        value_type="float",
        group="canonical",
    ),
    _p(
        "canonical_anchor",
        "--canonical-anchor",
        ("bottom_center", "center"),
        group="canonical",
    ),
)
