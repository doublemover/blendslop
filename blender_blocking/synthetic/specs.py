"""Typed contracts for deterministic synthetic shape generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

try:
    from blender_blocking.utils.json_io import json_safe
except ImportError:  # pragma: no cover - script-style imports
    from utils.json_io import json_safe


GENERATOR_VERSION = "synthetic-shape-factory-v1"


class ShapeFamily(str, Enum):
    ANALYTIC_PRIMITIVE = "analytic_primitive"
    PROFILE_LATHE = "profile_lathe"
    FURNITURE = "furniture"
    VEHICLE_MECHANICAL = "vehicle_mechanical"
    ADVERSARIAL_SILHOUETTE = "adversarial_silhouette"
    CAPTURE_NOISE = "capture_noise"


class ChallengeTag(str, Enum):
    ANALYTIC_EXACT = "analytic_exact"
    THIN_SUPPORTS = "thin_supports"
    HOLES = "holes"
    MULTI_PART = "multi_part"
    CONCAVITY = "concavity"
    AMBIGUOUS_POLARITY = "ambiguous_polarity"
    BORDER_TOUCHING = "border_touching"
    OUTLIER_PIXELS = "outlier_pixels"
    DUST_CLUSTERS = "dust_clusters"
    LOW_CONTRAST = "low_contrast"
    OCCLUSION = "occlusion"
    PROFILE_BANDS = "profile_bands"
    MATERIAL_REGIONS = "material_regions"
    UV_STRETCH = "uv_stretch"
    TEXTURE_ONLY_DETAIL = "texture_only_detail"
    PBR_CHANNELS = "pbr_channels"


class FailureModeTag(str, Enum):
    SILHOUETTE_AMBIGUITY = "silhouette_ambiguity"
    THIN_LEG_LOSS = "thin_leg_loss"
    INTERNAL_OVERLAP = "internal_overlap"
    NON_MANIFOLD_RISK = "non_manifold_risk"
    VISUAL_HULL_CONCAVITY_FILL = "visual_hull_concavity_fill"
    POLARITY_MISCLASSIFICATION = "polarity_misclassification"
    OUTLIER_BBOX_EXPANSION = "outlier_bbox_expansion"
    COMPONENT_DROPOUT = "component_dropout"
    INSUFFICIENT_VIEWS = "insufficient_views"
    UV_INVALID = "uv_invalid"
    MATERIAL_CHANNEL_DROPOUT = "material_channel_dropout"
    TEXTURE_ONLY_HALLUCINATION = "texture_only_hallucination"


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class SyntheticShapeSpec:
    shape_id: str
    family: str
    seed: int
    parameters: Mapping[str, object]
    transforms: Mapping[str, object] = field(default_factory=dict)
    materials: Mapping[str, object] = field(default_factory=dict)
    intended_challenges: tuple[str, ...] = ()
    symmetry: tuple[str, ...] = ()
    known_dimensions: Mapping[str, float] = field(default_factory=dict)
    expected_failure_modes: tuple[str, ...] = ()
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.shape_id:
            raise ValueError("shape_id is required")
        if not self.family:
            raise ValueError("family is required")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        _validate_json_safe_mapping("parameters", self.parameters)
        _validate_json_safe_mapping("transforms", self.transforms)
        _validate_json_safe_mapping("materials", self.materials)
        if any(value <= 0.0 for value in self.known_dimensions.values()):
            raise ValueError("known_dimensions values must be positive")

    def to_dict(self) -> dict[str, object]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SyntheticShapeSpec":
        return cls(
            shape_id=str(data["shape_id"]),
            family=str(data["family"]),
            seed=int(data["seed"]),
            parameters=dict(data.get("parameters", {})),
            transforms=dict(data.get("transforms", {})),
            materials=dict(data.get("materials", {})),
            intended_challenges=tuple(str(x) for x in data.get("intended_challenges", ())),
            symmetry=tuple(str(x) for x in data.get("symmetry", ())),
            known_dimensions={str(k): float(v) for k, v in dict(data.get("known_dimensions", {})).items()},
            expected_failure_modes=tuple(str(x) for x in data.get("expected_failure_modes", ())),
            generator_version=str(data.get("generator_version", GENERATOR_VERSION)),
        )


@dataclass(frozen=True)
class SyntheticViewSpec:
    view_name: str
    camera_type: Literal["orthographic", "perspective"]
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float = 0.0
    resolution: tuple[int, int] = (256, 256)
    padding: float = 0.08
    occluders: tuple[Mapping[str, object], ...] = ()
    expected_visible_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.view_name:
            raise ValueError("view_name is required")
        if self.camera_type not in ("orthographic", "perspective"):
            raise ValueError("camera_type must be orthographic or perspective")
        width, height = self.resolution
        if width <= 0 or height <= 0:
            raise ValueError("resolution values must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        for index, occluder in enumerate(self.occluders):
            _validate_json_safe_mapping(f"occluders[{index}]", occluder)

    def to_dict(self) -> dict[str, object]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SyntheticViewSpec":
        resolution = data.get("resolution", (256, 256))
        return cls(
            view_name=str(data["view_name"]),
            camera_type=str(data.get("camera_type", "orthographic")),  # type: ignore[arg-type]
            azimuth_deg=float(data.get("azimuth_deg", 0.0)),
            elevation_deg=float(data.get("elevation_deg", 0.0)),
            roll_deg=float(data.get("roll_deg", 0.0)),
            resolution=(int(resolution[0]), int(resolution[1])),  # type: ignore[index]
            padding=float(data.get("padding", 0.08)),
            occluders=tuple(dict(x) for x in data.get("occluders", ())),
            expected_visible_features=tuple(str(x) for x in data.get("expected_visible_features", ())),
        )


@dataclass(frozen=True)
class SyntheticArtifactSet:
    shape_spec_path: Path
    mesh_paths: Mapping[str, Path] = field(default_factory=dict)
    volume_paths: Mapping[str, Path] = field(default_factory=dict)
    image_paths: Mapping[str, Path] = field(default_factory=dict)
    mask_paths: Mapping[str, Path] = field(default_factory=dict)
    metric_paths: Mapping[str, Path] = field(default_factory=dict)
    manifest_path: Path = Path("manifest.json")

    def to_dict(self) -> dict[str, object]:
        return json_safe(
            {
                "shape_spec_path": self.shape_spec_path,
                "mesh_paths": self.mesh_paths,
                "volume_paths": self.volume_paths,
                "image_paths": self.image_paths,
                "mask_paths": self.mask_paths,
                "metric_paths": self.metric_paths,
                "manifest_path": self.manifest_path,
            }
        )


DEFAULT_VIEWS: tuple[SyntheticViewSpec, ...] = (
    SyntheticViewSpec("front", "orthographic", azimuth_deg=0.0, elevation_deg=0.0),
    SyntheticViewSpec("side", "orthographic", azimuth_deg=90.0, elevation_deg=0.0),
    SyntheticViewSpec("top", "orthographic", azimuth_deg=0.0, elevation_deg=90.0),
)


def stable_shape_id(kind: str, seed: int, variant: str | None = None) -> str:
    suffix = f"_{variant}" if variant else ""
    return f"{kind}{suffix}_seed_{seed:04d}"


def challenge_values(values: Iterable[ChallengeTag | str]) -> tuple[str, ...]:
    return tuple(value.value if isinstance(value, ChallengeTag) else str(value) for value in values)


def failure_values(values: Iterable[FailureModeTag | str]) -> tuple[str, ...]:
    return tuple(value.value if isinstance(value, FailureModeTag) else str(value) for value in values)


def _validate_json_safe_mapping(name: str, mapping: Mapping[str, object]) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        _validate_json_safe_value(f"{name}.{key}", value)


def _validate_json_safe_value(name: str, value: object) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Mapping):
        _validate_json_safe_mapping(name, value)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_safe_value(f"{name}[{index}]", item)
        return
    raise TypeError(f"{name} must be JSON-safe, got {type(value).__name__}")
