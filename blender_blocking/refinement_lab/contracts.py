"""Typed contracts for reconstruction refinement experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


JsonMap = dict[str, Any]

_VALUE_TYPES = {"string", "int", "float", "bool", "csv", "json"}
_SCALES = {"linear", "log"}
_VALIDATION_MODES = {"auto", "render-iou", "backend-status"}
_CASE_SOURCES = {"builtin_sample", "custom_images", "synthetic"}
_RESULT_STATUSES = {"pass", "fail", "error", "skip"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert common values into JSON-safe structures."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {
            key: json_safe(getattr(value, key))
            for key in getattr(value, "__dataclass_fields__", {})
        }
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def stable_hash(payload: Any, *, length: int = 12) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def safe_slug(value: str, *, fallback: str = "unnamed") -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or fallback


def _tuple(value: Sequence[Any] | None) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _path_or_none(value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(str(value))


def _coerce_path_map(value: Mapping[str, Any] | None) -> dict[str, Path]:
    return {str(key): Path(str(item)) for key, item in (value or {}).items()}


def _coerce_json_path_map(value: Mapping[str, Any] | None) -> dict[str, str]:
    return {str(key): str(item) for key, item in (value or {}).items()}


def _is_json_safe(value: Any) -> bool:
    try:
        json.dumps(json_safe(value), sort_keys=True, default=str)
    except TypeError:
        return False
    return True


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    cli_arg: Optional[str] = None
    config_path: Optional[str] = None
    value_type: str = "string"
    choices: tuple[Any, ...] = ()
    values: tuple[Any, ...] = ()
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    scale: str = "linear"
    default: Any = None
    group: str = ""
    description: str = ""
    requires: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    lab_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", _tuple(self.choices))
        object.__setattr__(self, "values", _tuple(self.values))
        object.__setattr__(self, "requires", tuple(str(v) for v in self.requires))
        object.__setattr__(self, "excludes", tuple(str(v) for v in self.excludes))
        self.validate()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("parameter name is required")
        if not (self.cli_arg or self.config_path or self.lab_only):
            raise ValueError(
                f"parameter {self.name!r} must define cli_arg, config_path, or lab_only"
            )
        if self.value_type not in _VALUE_TYPES:
            raise ValueError(f"unsupported parameter value_type: {self.value_type}")
        if self.scale not in _SCALES:
            raise ValueError(f"unsupported parameter scale: {self.scale}")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value")
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be > 0")
        if self.scale == "log":
            if self.min_value is not None and self.min_value <= 0:
                raise ValueError("log-scale min_value must be positive")
            if self.max_value is not None and self.max_value <= 0:
                raise ValueError("log-scale max_value must be positive")
        for value in self.values + self.choices:
            self._coerce_value(value)

    def _coerce_value(self, value: Any) -> Any:
        if self.value_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}:
                return value.lower() in {"true", "1"}
            raise ValueError(f"{self.name} expects bool values")
        if self.value_type == "int":
            if isinstance(value, bool):
                raise ValueError(f"{self.name} expects int values")
            return int(value)
        if self.value_type == "float":
            if isinstance(value, bool):
                raise ValueError(f"{self.name} expects float values")
            return float(value)
        if self.value_type == "csv":
            if isinstance(value, (list, tuple)):
                return tuple(str(item) for item in value)
            return tuple(part for part in str(value).split(",") if part)
        if self.value_type == "json":
            if not _is_json_safe(value):
                raise ValueError(f"{self.name} expects JSON-safe values")
            return value
        return str(value)

    def cli_tokens_for_value(self, value: Any) -> tuple[str, ...]:
        if not self.cli_arg:
            return ()
        coerced = self._coerce_value(value)
        if self.value_type == "bool":
            flag = self.cli_arg
            if coerced:
                return (flag,)
            if flag.startswith("--"):
                return (f"--no-{flag[2:]}",)
            return (flag, "false")
        if self.value_type == "csv":
            return (self.cli_arg, ",".join(str(item) for item in coerced))
        if self.value_type == "json":
            return (
                self.cli_arg,
                json.dumps(json_safe(coerced), sort_keys=True, separators=(",", ":")),
            )
        return (self.cli_arg, str(coerced))

    def to_dict(self) -> JsonMap:
        return {
            "name": self.name,
            "cli_arg": self.cli_arg,
            "config_path": self.config_path,
            "value_type": self.value_type,
            "choices": list(json_safe(self.choices)),
            "values": list(json_safe(self.values)),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "scale": self.scale,
            "default": json_safe(self.default),
            "group": self.group,
            "description": self.description,
            "requires": list(self.requires),
            "excludes": list(self.excludes),
            "lab_only": self.lab_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterSpec":
        payload = dict(data)
        for key in ("choices", "values", "requires", "excludes"):
            if key in payload:
                payload[key] = tuple(payload[key] or ())
        return cls(**payload)


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    label: str
    mode: str
    validation_mode: str = "render-iou"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    cli_args: tuple[str, ...] = ()
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    expected_artifacts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    parent_variant_id: str = ""
    stage: str = ""
    diagnostic_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cli_args", tuple(str(v) for v in self.cli_args))
        object.__setattr__(self, "expected_artifacts", tuple(self.expected_artifacts))
        object.__setattr__(self, "tags", tuple(str(v) for v in self.tags))
        self.validate()

    def validate(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id is required")
        if not self.label:
            raise ValueError("variant label is required")
        if not self.mode:
            raise ValueError("variant mode is required")
        if self.validation_mode not in _VALIDATION_MODES:
            raise ValueError("invalid validation_mode")
        if not _is_json_safe(self.parameters):
            raise ValueError("variant parameters must be JSON-safe")
        if not _is_json_safe(self.config_overrides):
            raise ValueError("variant config_overrides must be JSON-safe")

    def variant_hash(self) -> str:
        return stable_hash(
            {
                "mode": self.mode,
                "validation_mode": self.validation_mode,
                "parameters": self.parameters,
                "config_overrides": self.config_overrides,
                "cli_args": self.cli_args,
            }
        )

    def to_dict(self) -> JsonMap:
        return {
            "variant_id": self.variant_id,
            "label": self.label,
            "mode": self.mode,
            "validation_mode": self.validation_mode,
            "parameters": json_safe(self.parameters),
            "cli_args": list(self.cli_args),
            "config_overrides": json_safe(self.config_overrides),
            "expected_artifacts": list(self.expected_artifacts),
            "tags": list(self.tags),
            "parent_variant_id": self.parent_variant_id,
            "stage": self.stage,
            "diagnostic_only": self.diagnostic_only,
            "variant_hash": self.variant_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentVariant":
        payload = dict(data)
        payload.pop("variant_hash", None)
        for key in ("cli_args", "expected_artifacts", "tags"):
            if key in payload:
                payload[key] = tuple(payload[key] or ())
        return cls(**payload)


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    suite: str
    source: str
    reference_paths: Mapping[str, Path] = field(default_factory=dict)
    synthetic_shape_id: str = ""
    synthetic_definition: str = ""
    expected_targets: Mapping[str, Any] = field(default_factory=dict)
    known_ambiguity_notes: tuple[str, ...] = ()
    required_views: tuple[str, ...] = ("front", "side", "top")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_paths", _coerce_path_map(self.reference_paths))
        object.__setattr__(
            self,
            "known_ambiguity_notes",
            tuple(str(v) for v in self.known_ambiguity_notes),
        )
        object.__setattr__(self, "required_views", tuple(str(v) for v in self.required_views))
        self.validate()

    def validate(self) -> None:
        if not self.case_id:
            raise ValueError("case_id is required")
        if not self.suite:
            raise ValueError("suite is required")
        if self.source not in _CASE_SOURCES:
            raise ValueError(f"invalid case source: {self.source}")
        if self.source in {"builtin_sample", "custom_images"}:
            missing = [view for view in self.required_views if view not in self.reference_paths]
            if missing:
                raise ValueError(f"case {self.case_id} missing reference views: {missing}")

    def to_dict(self) -> JsonMap:
        return {
            "case_id": self.case_id,
            "suite": self.suite,
            "source": self.source,
            "reference_paths": _coerce_json_path_map(self.reference_paths),
            "synthetic_shape_id": self.synthetic_shape_id,
            "synthetic_definition": self.synthetic_definition,
            "expected_targets": json_safe(self.expected_targets),
            "known_ambiguity_notes": list(self.known_ambiguity_notes),
            "required_views": list(self.required_views),
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentCase":
        payload = dict(data)
        payload["reference_paths"] = _coerce_path_map(payload.get("reference_paths"))
        for key in ("known_ambiguity_notes", "required_views"):
            if key in payload:
                payload[key] = tuple(payload[key] or ())
        return cls(**payload)


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    suite: str
    track: str
    search: str
    objective: str
    output_root: Path
    run_id: str
    cases: tuple[ExperimentCase, ...]
    variants: tuple[ExperimentVariant, ...]
    max_runs: Optional[int] = None
    top_k: int = 10
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self,
            "cases",
            tuple(
                case if isinstance(case, ExperimentCase) else ExperimentCase.from_dict(case)
                for case in self.cases
            ),
        )
        object.__setattr__(
            self,
            "variants",
            tuple(
                variant
                if isinstance(variant, ExperimentVariant)
                else ExperimentVariant.from_dict(variant)
                for variant in self.variants
            ),
        )
        self.validate()

    def validate(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if not self.cases:
            raise ValueError("experiment plan requires at least one case")
        if not self.variants:
            raise ValueError("experiment plan requires at least one variant")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be >= 1 when provided")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate case ids in plan")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("duplicate variant ids in plan")

    def to_dict(self) -> JsonMap:
        return {
            "schema_version": "refinement_plan_v1",
            "plan_id": self.plan_id,
            "suite": self.suite,
            "track": self.track,
            "search": self.search,
            "objective": self.objective,
            "output_root": self.output_root.as_posix(),
            "run_id": self.run_id,
            "cases": [case.to_dict() for case in self.cases],
            "variants": [variant.to_dict() for variant in self.variants],
            "max_runs": self.max_runs,
            "top_k": self.top_k,
            "seed": self.seed,
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentPlan":
        payload = dict(data)
        payload.pop("schema_version", None)
        payload["output_root"] = Path(payload["output_root"])
        payload["cases"] = tuple(ExperimentCase.from_dict(item) for item in payload["cases"])
        payload["variants"] = tuple(
            ExperimentVariant.from_dict(item) for item in payload["variants"]
        )
        return cls(**payload)

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, path: Path) -> "ExperimentPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    case_id: str
    variant_id: str
    mode: str
    status: str
    exit_code: int
    started_utc: str
    finished_utc: str
    elapsed_s: float
    command: tuple[str, ...] = ()
    result_json: Optional[Path] = None
    render_paths: Mapping[str, Path] = field(default_factory=dict)
    reference_paths: Mapping[str, Path] = field(default_factory=dict)
    backend_result: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    score: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    autopsy: Mapping[str, Any] = field(default_factory=dict)
    bounds_debug: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(str(v) for v in self.command))
        object.__setattr__(self, "result_json", _path_or_none(self.result_json))
        object.__setattr__(self, "render_paths", _coerce_path_map(self.render_paths))
        object.__setattr__(self, "reference_paths", _coerce_path_map(self.reference_paths))
        object.__setattr__(self, "artifacts", _coerce_path_map(self.artifacts))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(self, "errors", tuple(str(v) for v in self.errors))
        self.validate()

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def avg_iou(self) -> float:
        return float(self.metrics.get("average_iou", self.metrics.get("area_iou_mean", 0.0)) or 0.0)

    @property
    def min_iou(self) -> float:
        values = [
            self.view_iou(view)
            for view in ("front", "side", "top")
            if self.view_iou(view) is not None
        ]
        return min(values) if values else float(self.metrics.get("area_iou_min", 0.0) or 0.0)

    def view_iou(self, view: str) -> Optional[float]:
        key = f"{view}_iou"
        value = self.metrics.get(key)
        if value is None:
            views = self.metrics.get("views", {})
            if isinstance(views, Mapping):
                view_data = views.get(view, {})
                if isinstance(view_data, Mapping):
                    value = view_data.get("iou", view_data.get("area_iou"))
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def validate(self) -> None:
        if self.status not in _RESULT_STATUSES:
            raise ValueError(f"invalid experiment result status: {self.status}")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be >= 0")

    def to_dict(self) -> JsonMap:
        return {
            "schema_version": "refinement_result_v1",
            "run_id": self.run_id,
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "mode": self.mode,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_s": self.elapsed_s,
            "command": list(self.command),
            "result_json": self.result_json.as_posix() if self.result_json else None,
            "render_paths": _coerce_json_path_map(self.render_paths),
            "reference_paths": _coerce_json_path_map(self.reference_paths),
            "backend_result": json_safe(self.backend_result),
            "metrics": json_safe(self.metrics),
            "score": json_safe(self.score),
            "artifacts": _coerce_json_path_map(self.artifacts),
            "autopsy": json_safe(self.autopsy),
            "bounds_debug": json_safe(self.bounds_debug),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        payload = dict(data)
        payload.pop("schema_version", None)
        payload["result_json"] = _path_or_none(payload.get("result_json"))
        for key in ("render_paths", "reference_paths", "artifacts"):
            payload[key] = _coerce_path_map(payload.get(key))
        for key in ("command", "warnings", "errors"):
            payload[key] = tuple(payload.get(key) or ())
        return cls(**payload)


@dataclass(frozen=True)
class RefinementRunManifest:
    run_id: str
    created_utc: str
    suite: str
    track: str
    search: str
    objective: str
    output_root: Path
    git: Mapping[str, Any]
    environment: Mapping[str, Any]
    dependency_report: Mapping[str, Any]
    plan_path: Path
    index_path: Path
    html_report_path: Optional[Path] = None
    cases: tuple[ExperimentCase, ...] = ()
    variants: tuple[ExperimentVariant, ...] = ()
    result_counts: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = "refinement_run_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "plan_path", Path(self.plan_path))
        object.__setattr__(self, "index_path", Path(self.index_path))
        object.__setattr__(
            self,
            "html_report_path",
            _path_or_none(self.html_report_path),
        )

    def to_dict(self) -> JsonMap:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "suite": self.suite,
            "track": self.track,
            "search": self.search,
            "objective": self.objective,
            "output_root": self.output_root.as_posix(),
            "git": json_safe(self.git),
            "environment": json_safe(self.environment),
            "dependency_report": json_safe(self.dependency_report),
            "plan_path": self.plan_path.as_posix(),
            "index_path": self.index_path.as_posix(),
            "html_report_path": self.html_report_path.as_posix()
            if self.html_report_path
            else None,
            "cases": [case.to_dict() for case in self.cases],
            "variants": [variant.to_dict() for variant in self.variants],
            "result_counts": dict(self.result_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefinementRunManifest":
        payload = dict(data)
        payload["output_root"] = Path(payload["output_root"])
        payload["plan_path"] = Path(payload["plan_path"])
        payload["index_path"] = Path(payload["index_path"])
        payload["html_report_path"] = _path_or_none(payload.get("html_report_path"))
        payload["cases"] = tuple(
            ExperimentCase.from_dict(item) for item in payload.get("cases", ())
        )
        payload["variants"] = tuple(
            ExperimentVariant.from_dict(item) for item in payload.get("variants", ())
        )
        return cls(**payload)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
