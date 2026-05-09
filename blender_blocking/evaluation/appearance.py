"""Texture, UV, material, and appearance-attribution evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_PBR_CHANNELS = ("base_color", "roughness", "metallic", "normal")


@dataclass(frozen=True)
class TextureMaterialReport:
    """SOTA-style appearance report for editable Blender asset outputs.

    Geometry metrics answer whether the shape is correct.  This report answers
    whether texture and material evidence is valid, editable, and not being used
    to hide missing geometry.
    """

    has_uv_map: bool | None = None
    uv_valid: bool | None = None
    uv_island_count: int | None = None
    uv_overlap_ratio: float | None = None
    uv_out_of_bounds_ratio: float | None = None
    uv_stretch_mean: float | None = None
    texel_density_cv: float | None = None
    missing_uv_faces: int | None = None
    texture_width: int | None = None
    texture_height: int | None = None
    texture_file_count: int | None = None
    texture_memory_mb: float | None = None
    reprojection_metrics: Mapping[str, float] = field(default_factory=dict)
    seam_visibility_score: float | None = None
    material_slot_count: int | None = None
    named_material_ratio: float | None = None
    procedural_material_ratio: float | None = None
    bitmap_material_ratio: float | None = None
    pbr_channel_coverage: Mapping[str, bool] = field(default_factory=dict)
    duplicate_material_count: int | None = None
    orphan_texture_count: int | None = None
    appearance_attribution: Mapping[str, float | bool] = field(default_factory=dict)
    required: bool = False
    strict_uv: bool = False
    material_target: str = "pbr"
    max_texture_memory_mb: float | None = None
    source: str = "computed"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def pbr_channel_coverage_ratio(self) -> float | None:
        if not self.pbr_channel_coverage:
            return None
        return sum(1 for value in self.pbr_channel_coverage.values() if value) / len(
            self.pbr_channel_coverage
        )

    @property
    def texture_resolution(self) -> int | None:
        if self.texture_width is None or self.texture_height is None:
            return None
        return int(self.texture_width) * int(self.texture_height)

    @property
    def image_space_hallucination_warning(self) -> bool:
        explicit = _bool_or_none(
            self.appearance_attribution.get("image_space_hallucination_warning")
        )
        if explicit is not None:
            return explicit
        texture_detail = _float_or_none(
            self.appearance_attribution.get("texture_only_detail_score")
        )
        geometry_detail = _float_or_none(
            self.appearance_attribution.get("geometry_detail_score")
        )
        boundary = _float_or_none(
            self.appearance_attribution.get("boundary_geometry_fidelity")
        )
        if texture_detail is None or geometry_detail is None:
            return False
        return texture_detail >= geometry_detail + 0.25 and (
            boundary is None or boundary < 0.65
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "texture_material_report_v1",
            "source": self.source,
            "required": self.required,
            "strict_uv": self.strict_uv,
            "material_target": self.material_target,
            "has_uv_map": self.has_uv_map,
            "uv_valid": self.uv_valid,
            "uv_island_count": self.uv_island_count,
            "uv_overlap_ratio": self.uv_overlap_ratio,
            "uv_out_of_bounds_ratio": self.uv_out_of_bounds_ratio,
            "uv_stretch_mean": self.uv_stretch_mean,
            "texel_density_cv": self.texel_density_cv,
            "missing_uv_faces": self.missing_uv_faces,
            "texture_width": self.texture_width,
            "texture_height": self.texture_height,
            "texture_resolution": self.texture_resolution,
            "texture_file_count": self.texture_file_count,
            "texture_memory_mb": self.texture_memory_mb,
            "max_texture_memory_mb": self.max_texture_memory_mb,
            "reprojection_metrics": dict(self.reprojection_metrics),
            "seam_visibility_score": self.seam_visibility_score,
            "material_slot_count": self.material_slot_count,
            "named_material_ratio": self.named_material_ratio,
            "procedural_material_ratio": self.procedural_material_ratio,
            "bitmap_material_ratio": self.bitmap_material_ratio,
            "pbr_channel_coverage": dict(self.pbr_channel_coverage),
            "pbr_channel_coverage_ratio": self.pbr_channel_coverage_ratio,
            "duplicate_material_count": self.duplicate_material_count,
            "orphan_texture_count": self.orphan_texture_count,
            "appearance_attribution": dict(self.appearance_attribution),
            "image_space_hallucination_warning": self.image_space_hallucination_warning,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def report_from_mapping(payload: Mapping[str, Any]) -> TextureMaterialReport:
    """Normalize a nested appearance payload into a typed report."""

    uv = _mapping(payload.get("uv"))
    texture = _mapping(payload.get("texture"))
    materials = _mapping(payload.get("materials", payload.get("material")))
    attribution = _mapping(
        payload.get("appearance_attribution", payload.get("attribution"))
    )

    pbr_payload = _mapping(
        materials.get(
            "pbr_channel_coverage",
            payload.get("pbr_channel_coverage", payload.get("pbr_channels")),
        )
    )
    pbr_channels = _normalize_pbr_channels(pbr_payload)
    if not pbr_channels and str(payload.get("material_target", "pbr")) == "pbr":
        pbr_channels = {channel: False for channel in DEFAULT_PBR_CHANNELS}

    report = TextureMaterialReport(
        has_uv_map=_bool_or_none(
            _first_present(payload, uv, "has_uv_map", "has_uv", "uv_map")
        ),
        uv_valid=_bool_or_none(_first_present(payload, uv, "uv_valid", "valid")),
        uv_island_count=_int_or_none(
            _first_present(payload, uv, "uv_island_count", "island_count")
        ),
        uv_overlap_ratio=_float_or_none(
            _first_present(payload, uv, "uv_overlap_ratio", "overlap_ratio")
        ),
        uv_out_of_bounds_ratio=_float_or_none(
            _first_present(
                payload,
                uv,
                "uv_out_of_bounds_ratio",
                "out_of_bounds_ratio",
                "oob_ratio",
            )
        ),
        uv_stretch_mean=_float_or_none(
            _first_present(payload, uv, "uv_stretch_mean", "stretch_mean")
        ),
        texel_density_cv=_float_or_none(
            _first_present(payload, uv, "texel_density_cv", "texel_density_variance")
        ),
        missing_uv_faces=_int_or_none(
            _first_present(payload, uv, "missing_uv_faces", "faces_without_uv")
        ),
        texture_width=_int_or_none(
            _first_present(payload, texture, "texture_width", "width")
        ),
        texture_height=_int_or_none(
            _first_present(payload, texture, "texture_height", "height")
        ),
        texture_file_count=_int_or_none(
            _first_present(payload, texture, "texture_file_count", "file_count")
        ),
        texture_memory_mb=_float_or_none(
            _first_present(payload, texture, "texture_memory_mb", "memory_mb")
        ),
        reprojection_metrics=_float_mapping(
            _mapping(texture.get("reprojection_metrics", payload.get("reprojection_metrics")))
        ),
        seam_visibility_score=_float_or_none(
            _first_present(payload, texture, "seam_visibility_score", "seam_score")
        ),
        material_slot_count=_int_or_none(
            _first_present(
                payload, materials, "material_slot_count", "slot_count", "materials"
            )
        ),
        named_material_ratio=_float_or_none(
            _first_present(payload, materials, "named_material_ratio", "named_ratio")
        ),
        procedural_material_ratio=_float_or_none(
            _first_present(
                payload, materials, "procedural_material_ratio", "procedural_ratio"
            )
        ),
        bitmap_material_ratio=_float_or_none(
            _first_present(payload, materials, "bitmap_material_ratio", "bitmap_ratio")
        ),
        pbr_channel_coverage=pbr_channels,
        duplicate_material_count=_int_or_none(
            _first_present(
                payload, materials, "duplicate_material_count", "duplicate_count"
            )
        ),
        orphan_texture_count=_int_or_none(
            _first_present(payload, materials, "orphan_texture_count", "orphan_count")
        ),
        appearance_attribution=_attribution_mapping(attribution),
        required=bool(payload.get("required", False)),
        strict_uv=bool(payload.get("strict_uv", payload.get("uv_strict", False))),
        material_target=str(payload.get("material_target", "pbr") or "pbr"),
        max_texture_memory_mb=_float_or_none(payload.get("max_texture_memory_mb")),
        source=str(payload.get("source", "computed")),
        warnings=_string_tuple(payload.get("warnings", ())),
        errors=_string_tuple(payload.get("errors", ())),
    )
    return _with_derived_warnings(report)


def reports_from_payload(payload: Any) -> tuple[TextureMaterialReport, ...]:
    """Return one or more reports from common backend payload shapes."""

    if not payload:
        return ()
    if isinstance(payload, Mapping):
        if "reports" in payload:
            return reports_from_payload(payload.get("reports"))
        if "targets" in payload and isinstance(payload["targets"], Mapping):
            reports = []
            for name, item in payload["targets"].items():
                if not isinstance(item, Mapping):
                    continue
                merged = dict(item)
                merged.setdefault("source", str(name))
                reports.append(report_from_mapping(merged))
            return tuple(reports)
        return (report_from_mapping(payload),)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return tuple(
            report_from_mapping(item) for item in payload if isinstance(item, Mapping)
        )
    return ()


def _with_derived_warnings(report: TextureMaterialReport) -> TextureMaterialReport:
    warnings = list(report.warnings)
    errors = list(report.errors)

    if report.required and report.has_uv_map is False:
        errors.append("required_uv_map_missing")
    elif report.has_uv_map is False:
        warnings.append("uv_map_missing")
    if report.uv_valid is False:
        (errors if report.required or report.strict_uv else warnings).append("uv_invalid")
    if report.uv_overlap_ratio is not None and report.uv_overlap_ratio > 0.10:
        (errors if report.strict_uv else warnings).append("uv_overlap_high")
    if (
        report.uv_out_of_bounds_ratio is not None
        and report.uv_out_of_bounds_ratio > 0.05
    ):
        (errors if report.strict_uv else warnings).append("uv_out_of_bounds_high")
    if report.missing_uv_faces is not None and report.missing_uv_faces > 0:
        (errors if report.strict_uv else warnings).append("missing_uv_faces")
    if report.texel_density_cv is not None and report.texel_density_cv > 1.0:
        warnings.append("texel_density_variance_high")
    if report.texture_memory_mb is not None and report.max_texture_memory_mb is not None:
        if report.texture_memory_mb > report.max_texture_memory_mb:
            errors.append("texture_memory_budget_exceeded")
    if report.material_target == "pbr":
        coverage = report.pbr_channel_coverage_ratio
        if coverage is not None and coverage < 0.75:
            warnings.append("pbr_channel_coverage_low")
    if report.named_material_ratio is not None and report.named_material_ratio < 0.5:
        warnings.append("material_slots_not_named")
    if report.orphan_texture_count is not None and report.orphan_texture_count > 0:
        warnings.append("orphan_textures_present")
    if report.image_space_hallucination_warning:
        warnings.append("texture_detail_exceeds_geometry_detail")

    return TextureMaterialReport(
        **{
            **report.__dict__,
            "warnings": tuple(dict.fromkeys(warnings)),
            "errors": tuple(dict.fromkeys(errors)),
        }
    )


def _normalize_pbr_channels(payload: Mapping[str, Any]) -> dict[str, bool]:
    if not payload:
        return {}
    channels: dict[str, bool] = {}
    for key, value in payload.items():
        channels[_metric_fragment(key)] = bool(value)
    return channels


def _first_present(*sources_and_keys: Any) -> Any:
    sources = [item for item in sources_and_keys if isinstance(item, Mapping)]
    keys = [item for item in sources_and_keys if isinstance(item, str)]
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, item in value.items():
        parsed = _float_or_none(item)
        if parsed is not None:
            output[_metric_fragment(key)] = parsed
    return output


def _attribution_mapping(value: Mapping[str, Any]) -> dict[str, float | bool]:
    output: dict[str, float | bool] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            output[_metric_fragment(key)] = item
            continue
        parsed = _float_or_none(item)
        if parsed is not None:
            output[_metric_fragment(key)] = parsed
    return output


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value)


def _metric_fragment(value: object) -> str:
    text = str(value or "").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part) or "value"
