"""Typed reconstruction data contracts.

These classes are intentionally lightweight and Blender-free.  They form the
wire format between extraction, backend execution, candidate scoring, and
manifest generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from blender_blocking.utils.path_safety import compact_path_segment
except ImportError:  # pragma: no cover - script-style imports
    from utils.path_safety import compact_path_segment


JsonMap = Dict[str, Any]


def _json_value(value: Any) -> Any:
    """Convert common dataclass values into JSON-safe structures."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    return value


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return bool(value)


def _normalize_per_view_metrics(
    per_view: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    if per_view is None:
        return normalized
    for view, metrics in per_view.items():
        if hasattr(metrics, "to_dict"):
            data = dict(metrics.to_dict())
        elif isinstance(metrics, Mapping):
            data = dict(metrics)
        else:
            data = {}

        if "area_iou" not in data and "area_iou_loss" in data:
            value = _optional_float(data.get("area_iou_loss"))
            data["area_iou"] = None if value is None else max(0.0, 1.0 - value)
        if "soft_iou" not in data and "soft_iou_loss" in data:
            value = _optional_float(data.get("soft_iou_loss"))
            data["soft_iou"] = None if value is None else max(0.0, 1.0 - value)

        data.setdefault("area_iou", 0.0)
        data.setdefault("boundary_iou", None)
        data.setdefault("soft_iou", None)
        data.setdefault("signed_distance_loss", None)

        required = _metric_bool(data.get("required", data.get("pass_required")), True)
        passed = _metric_bool(data.get("passed", data.get("pass")), not required)
        reason = str(data.get("reason", ""))
        if required and not passed and not reason:
            reason = "required per-view metrics failed"

        data["required"] = required
        data["passed"] = passed
        data["pass"] = passed
        data["reason"] = reason
        normalized[str(view)] = data
    return normalized


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _shape_of(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(part) for part in shape)
    except (TypeError, ValueError):
        return None


def _uncertainty_summary(value: Any) -> Optional[JsonMap]:
    if value is None:
        return None
    return {
        "source": getattr(value, "source", ""),
        "threshold": getattr(value, "threshold", None),
        "foreground_prob_shape": _shape_of(getattr(value, "foreground_prob", None)),
        "hard_mask_shape": _shape_of(getattr(value, "hard_mask", None)),
        "confidence_shape": _shape_of(getattr(value, "confidence", None)),
        "boundary_uncertainty_shape": _shape_of(
            getattr(value, "boundary_uncertainty", None)
        ),
        "diagnostics": _json_value(getattr(value, "diagnostics", {})),
    }


@dataclass(frozen=True)
class Bounds2D:
    """Exclusive 2D bounds in pixel or normalized coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_xyxy(cls, values: Sequence[float]) -> "Bounds2D":
        if len(values) != 4:
            raise ValueError("Bounds2D requires exactly four values")
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)

    def validate(self, *, allow_empty: bool = False) -> None:
        if allow_empty:
            if self.width < 0 or self.height < 0:
                raise ValueError("Bounds2D has negative extent")
        elif self.width <= 0 or self.height <= 0:
            raise ValueError("Bounds2D must have positive width and height")

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> JsonMap:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class Bounds3D:
    """World-space 3D bounds."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @classmethod
    def from_min_max(
        cls, bounds_min: Sequence[float], bounds_max: Sequence[float]
    ) -> "Bounds3D":
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            raise ValueError("Bounds3D min/max require three values each")
        return cls(
            float(bounds_min[0]),
            float(bounds_max[0]),
            float(bounds_min[1]),
            float(bounds_max[1]),
            float(bounds_min[2]),
            float(bounds_max[2]),
        )

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )

    def validate(self) -> None:
        if any(v <= 0 for v in self.size):
            raise ValueError("Bounds3D must have positive extent on every axis")

    def to_min_max(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (
            (self.min_x, self.min_y, self.min_z),
            (self.max_x, self.max_y, self.max_z),
        )

    def to_dict(self) -> JsonMap:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
        }


@dataclass(frozen=True)
class OrthographicCameraSpec:
    """Minimal orthographic camera description for reconstruction constraints."""

    view_name: str
    axis: str
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    roll_deg: float = 0.0
    resolution: Tuple[int, int] = (512, 512)
    bounds: Optional[Bounds2D] = None

    def to_dict(self) -> JsonMap:
        return {
            "view_name": self.view_name,
            "axis": self.axis,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "roll_deg": self.roll_deg,
            "resolution": list(self.resolution),
            "bounds": _json_value(self.bounds),
        }


@dataclass(frozen=True)
class ViewConstraint:
    """A silhouette/camera pair used by reconstruction backends."""

    view: str
    mask: Any
    camera: OrthographicCameraSpec
    bbox: Optional[Bounds2D] = None
    uncertainty: Optional[Any] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    valid_mask: Any = None

    def to_dict(self) -> JsonMap:
        return {
            "view": self.view,
            "camera": self.camera.to_dict(),
            "bbox": _json_value(self.bbox),
            "diagnostics": _json_value(self.diagnostics),
            "has_valid_mask": self.valid_mask is not None,
            "has_uncertainty": self.uncertainty is not None,
            "uncertainty": _uncertainty_summary(self.uncertainty),
        }


@dataclass(frozen=True)
class ProfileIntervalPx:
    """Foreground interval for a sampled silhouette row."""

    x0: float
    x1: float
    confidence: float = 1.0
    source: str = "mask"

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) * 0.5

    def to_dict(self) -> JsonMap:
        return {
            "x0": self.x0,
            "x1": self.x1,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class ProfileBand:
    """One sampled silhouette row preserving intervals and holes."""

    t: float
    intervals: Tuple[ProfileIntervalPx, ...]
    center_x: Optional[float] = None
    width_px: float = 0.0
    holes: Tuple[ProfileIntervalPx, ...] = ()
    moments: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source_view: str = ""

    def dominant_interval(self) -> Optional[ProfileIntervalPx]:
        if not self.intervals:
            return None
        return max(self.intervals, key=lambda interval: interval.width)

    def to_dict(self) -> JsonMap:
        return {
            "t": self.t,
            "intervals": [_json_value(v) for v in self.intervals],
            "center_x": self.center_x,
            "width_px": self.width_px,
            "holes": [_json_value(v) for v in self.holes],
            "moments": dict(self.moments),
            "confidence": self.confidence,
            "source_view": self.source_view,
        }


@dataclass(frozen=True)
class UncertainProfileBand(ProfileBand):
    center_std: float = 0.0
    width_std: float = 0.0

    def to_dict(self) -> JsonMap:
        data = super().to_dict()
        data.update({"center_std": self.center_std, "width_std": self.width_std})
        return data


@dataclass(frozen=True)
class ReconstructionTarget:
    """All preprocessed inputs needed by any reconstruction backend."""

    constraints: Tuple[ViewConstraint, ...] = ()
    profile_bands: Mapping[str, Tuple[ProfileBand, ...]] = field(default_factory=dict)
    bounds: Optional[Bounds3D] = None
    config_hash: str = ""
    constraint_hash: str = ""
    artifact_root: Optional[Path] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def views(self) -> Tuple[str, ...]:
        return tuple(constraint.view for constraint in self.constraints)

    def to_dict(self) -> JsonMap:
        return {
            "constraints": [_json_value(v) for v in self.constraints],
            "profile_bands": {
                view: [_json_value(band) for band in bands]
                for view, bands in self.profile_bands.items()
            },
            "bounds": _json_value(self.bounds),
            "config_hash": self.config_hash,
            "constraint_hash": self.constraint_hash,
            "artifact_root": str(self.artifact_root) if self.artifact_root else None,
            "extras": _json_value(self.extras),
        }


@dataclass(frozen=True)
class MeshQualityReport:
    """Portable mesh-quality summary for Blender and non-Blender backends."""

    object_name: str = ""
    vertices: int = 0
    edges: int = 0
    faces: int = 0
    loose_vertices: int = 0
    non_manifold_edges: int = 0
    boundary_edges: int = 0
    connected_components: int = 0
    degenerate_faces: int = 0
    zero_area_faces: int = 0
    bbox: Optional[Bounds3D] = None
    warnings: Tuple[str, ...] = ()

    @property
    def is_topology_clean(self) -> bool:
        return (
            self.loose_vertices == 0
            and self.non_manifold_edges == 0
            and self.degenerate_faces == 0
            and self.zero_area_faces == 0
        )

    def topology_score(self) -> float:
        score = 1.0
        score -= min(1.0, self.loose_vertices * 0.05)
        score -= min(1.0, self.non_manifold_edges * 0.03)
        score -= min(1.0, self.degenerate_faces * 0.03)
        score -= min(1.0, max(0, self.connected_components - 1) * 0.1)
        return max(0.0, score)

    def to_dict(self) -> JsonMap:
        return {
            "object_name": self.object_name,
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "loose_vertices": self.loose_vertices,
            "non_manifold_edges": self.non_manifold_edges,
            "boundary_edges": self.boundary_edges,
            "connected_components": self.connected_components,
            "degenerate_faces": self.degenerate_faces,
            "zero_area_faces": self.zero_area_faces,
            "bbox": _json_value(self.bbox),
            "warnings": list(self.warnings),
            "topology_score": self.topology_score(),
        }


@dataclass(frozen=True)
class MeshData:
    """Backend-neutral mesh arrays or mesh artifact references."""

    vertices: Tuple[Tuple[float, float, float], ...] = ()
    faces: Tuple[Tuple[int, ...], ...] = ()
    path: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return {
            "vertices": [list(v) for v in self.vertices],
            "faces": [list(f) for f in self.faces],
            "path": str(self.path) if self.path else None,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class MeshBuildResult:
    mesh: Optional[MeshData] = None
    quality: Optional[MeshQualityReport] = None
    status: str = "success"
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> JsonMap:
        return {
            "mesh": _json_value(self.mesh),
            "quality": _json_value(self.quality),
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class JoinAttempt:
    mode: str
    success: bool
    elapsed_s: float = 0.0
    warnings: Tuple[str, ...] = ()
    quality: Optional[MeshQualityReport] = None

    def to_dict(self) -> JsonMap:
        return {
            "mode": self.mode,
            "success": self.success,
            "elapsed_s": self.elapsed_s,
            "warnings": list(self.warnings),
            "quality": _json_value(self.quality),
        }


@dataclass(frozen=True)
class JoinResult:
    object: Any = None
    attempts: Tuple[JoinAttempt, ...] = ()
    selected_mode: Optional[str] = None
    degraded: bool = False
    fatal_error: Optional[str] = None

    def to_dict(self) -> JsonMap:
        return {
            "attempts": [_json_value(v) for v in self.attempts],
            "selected_mode": self.selected_mode,
            "degraded": self.degraded,
            "fatal_error": self.fatal_error,
            "has_object": self.object is not None,
        }


@dataclass(frozen=True)
class CandidateBudget:
    timeout_s: Optional[float] = None
    memory_budget_mb: Optional[int] = None
    max_vertices: Optional[int] = None
    max_faces: Optional[int] = None

    def to_dict(self) -> JsonMap:
        return {
            "timeout_s": self.timeout_s,
            "memory_budget_mb": self.memory_budget_mb,
            "max_vertices": self.max_vertices,
            "max_faces": self.max_faces,
        }


@dataclass(frozen=True)
class CandidateRequest:
    candidate_id: str
    backend_name: str
    target: ReconstructionTarget
    config: Mapping[str, Any] = field(default_factory=dict)
    budget: CandidateBudget = field(default_factory=CandidateBudget)
    artifact_root: Optional[Path] = None
    context: Any = None

    def candidate_artifact_root(self) -> Optional[Path]:
        if self.artifact_root is None:
            return None
        raw_candidate_path = Path(str(self.candidate_id))
        if raw_candidate_path.is_absolute() or any(
            part == ".." for part in raw_candidate_path.parts
        ):
            raise ValueError(
                f"candidate_id escapes artifact root: {self.candidate_id!r}"
            )
        artifact_root = self.artifact_root.resolve(strict=False)
        candidate_root = (
            artifact_root
            / compact_path_segment(
                self.candidate_id,
                max_length=36,
                fallback="candidate",
            )
        ).resolve(strict=False)
        try:
            candidate_root.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"candidate_id escapes artifact root: {self.candidate_id!r}"
            ) from exc
        return candidate_root

    def to_dict(self) -> JsonMap:
        return {
            "candidate_id": self.candidate_id,
            "backend_name": self.backend_name,
            "target": self.target.to_dict(),
            "config": _json_value(self.config),
            "budget": self.budget.to_dict(),
            "artifact_root": str(self.artifact_root) if self.artifact_root else None,
        }


@dataclass(frozen=True)
class CandidateMetrics:
    per_view: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    area_iou_min: float = 0.0
    area_iou_mean: float = 0.0
    boundary_iou_mean: float = 0.0
    topology_score: float = 0.0
    topology_penalty: float = 0.0
    uncertainty_consistency: float = 0.0
    constraint_score: float = 0.0
    constraint_penalty: float = 0.0
    constraint_report: Mapping[str, Any] = field(default_factory=dict)
    editability_score: float = 0.0
    complexity_penalty: float = 0.0
    elapsed_s: Optional[float] = None
    mesh_quality: Optional[MeshQualityReport] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        per_view = _normalize_per_view_metrics(self.per_view)
        object.__setattr__(self, "per_view", per_view)

        area_values = [
            value
            for value in (_optional_float(item.get("area_iou")) for item in per_view.values())
            if value is not None
        ]
        if area_values and self.area_iou_min == 0.0:
            object.__setattr__(self, "area_iou_min", min(area_values))
        if area_values and self.area_iou_mean == 0.0:
            object.__setattr__(self, "area_iou_mean", _mean(area_values))

        boundary_values = [
            value
            for value in (
                _optional_float(item.get("boundary_iou")) for item in per_view.values()
            )
            if value is not None
        ]
        if boundary_values and self.boundary_iou_mean == 0.0:
            object.__setattr__(self, "boundary_iou_mean", _mean(boundary_values))

        extras = dict(self.extras) if isinstance(self.extras, Mapping) else {}
        topology = extras.get("topology")
        if isinstance(topology, Mapping):
            topology_score = _optional_float(topology.get("topology_score"))
            if topology_score is not None and self.topology_score == 0.0:
                object.__setattr__(self, "topology_score", topology_score)
            topology_penalty = _optional_float(topology.get("penalty"))
            if topology_penalty is None and topology_score is not None:
                topology_penalty = max(0.0, 1.0 - topology_score)
            if topology_penalty is not None and self.topology_penalty == 0.0:
                object.__setattr__(self, "topology_penalty", topology_penalty)
        extra_topology_penalty = _optional_float(extras.get("topology_penalty"))
        if extra_topology_penalty is not None and self.topology_penalty == 0.0:
            object.__setattr__(self, "topology_penalty", extra_topology_penalty)

        uncertainty = extras.get("uncertainty_report", extras.get("uncertainty"))
        if isinstance(uncertainty, Mapping) and self.uncertainty_consistency == 0.0:
            uncertainty_score = _optional_float(
                uncertainty.get("consistency_score", uncertainty.get("confidence_score"))
            )
            if uncertainty_score is not None:
                object.__setattr__(self, "uncertainty_consistency", uncertainty_score)

        report = (
            dict(self.constraint_report)
            if isinstance(self.constraint_report, Mapping)
            else {}
        )
        extra_report = extras.get("constraint_report")
        if not report and isinstance(extra_report, Mapping):
            report = dict(extra_report)
            object.__setattr__(self, "constraint_report", report)
        report_score = _optional_float(report.get("score"))
        if report_score is not None and self.constraint_score == 0.0:
            object.__setattr__(self, "constraint_score", report_score)
        report_penalty = _optional_float(
            report.get("constraint_penalty", report.get("penalty"))
        )
        if report_penalty is not None and self.constraint_penalty == 0.0:
            object.__setattr__(self, "constraint_penalty", report_penalty)

    def to_dict(self) -> JsonMap:
        return {
            "per_view": _json_value(self.per_view),
            "area_iou_min": self.area_iou_min,
            "area_iou_mean": self.area_iou_mean,
            "boundary_iou_mean": self.boundary_iou_mean,
            "topology_score": self.topology_score,
            "topology_penalty": self.topology_penalty,
            "uncertainty_consistency": self.uncertainty_consistency,
            "constraint_score": self.constraint_score,
            "constraint_penalty": self.constraint_penalty,
            "constraint_report": _json_value(self.constraint_report),
            "editability_score": self.editability_score,
            "complexity_penalty": self.complexity_penalty,
            "elapsed_s": self.elapsed_s,
            "mesh_quality": _json_value(self.mesh_quality),
            "extras": _json_value(self.extras),
        }


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    backend_name: str
    status: str
    mesh_path: Optional[Path] = None
    primitive_path: Optional[Path] = None
    volume_path: Optional[Path] = None
    render_paths: Mapping[str, Path] = field(default_factory=dict)
    metric_result: CandidateMetrics = field(default_factory=CandidateMetrics)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()
    degraded: bool = False
    payload: Any = None
    geometry: Any = field(default=None, compare=False, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.status in {"success", "degraded"} and not self.errors

    def to_dict(self) -> JsonMap:
        return {
            "candidate_id": self.candidate_id,
            "backend_name": self.backend_name,
            "status": self.status,
            "mesh_path": str(self.mesh_path) if self.mesh_path else None,
            "primitive_path": str(self.primitive_path) if self.primitive_path else None,
            "volume_path": str(self.volume_path) if self.volume_path else None,
            "render_paths": {k: str(v) for k, v in self.render_paths.items()},
            "metric_result": self.metric_result.to_dict(),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "degraded": self.degraded,
        }

    def to_evaluation_bundle(
        self,
        *,
        target: Any = None,
        suite: str = "",
        run_id: str = "",
        repo: str | None = None,
    ) -> Any:
        """Create a source-level EvaluationBundle without importing at module load."""
        try:
            from blender_blocking.evaluation import bundle_from_candidate
        except Exception:  # pragma: no cover - script-style imports
            from evaluation import bundle_from_candidate

        return bundle_from_candidate(
            result=self,
            target=target,
            suite=suite,
            run_id=run_id,
            repo=repo,
        )


@dataclass(frozen=True)
class CandidateScoreTerm:
    name: str
    value: float
    weight: float = 1.0
    description: str = ""

    @property
    def weighted(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> JsonMap:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "weighted": self.weighted,
            "description": self.description,
        }


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    total: float
    terms: Tuple[CandidateScoreTerm, ...]
    policy: str = "best_score"

    def to_dict(self) -> JsonMap:
        return {
            "candidate_id": self.candidate_id,
            "total": self.total,
            "policy": self.policy,
            "terms": [term.to_dict() for term in self.terms],
        }


@dataclass(frozen=True)
class VolumeMetadata:
    value_type: str
    bounds: Optional[Bounds3D] = None
    voxel_size: Optional[float] = None
    resolution: Optional[Tuple[int, int, int]] = None
    default_value: float = 0.0
    dtype: str = "float32"
    source_candidate_id: str = ""

    def to_dict(self) -> JsonMap:
        return {
            "value_type": self.value_type,
            "bounds": _json_value(self.bounds),
            "voxel_size": self.voxel_size,
            "resolution": list(self.resolution) if self.resolution else None,
            "default_value": self.default_value,
            "dtype": self.dtype,
            "source_candidate_id": self.source_candidate_id,
        }


@dataclass(frozen=True)
class MeshExtractionResult:
    status: str
    mesh: Optional[MeshData] = None
    quality: Optional[MeshQualityReport] = None
    method: str = ""
    elapsed_s: float = 0.0
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> JsonMap:
        return {
            "status": self.status,
            "mesh": _json_value(self.mesh),
            "quality": _json_value(self.quality),
            "method": self.method,
            "elapsed_s": self.elapsed_s,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }
