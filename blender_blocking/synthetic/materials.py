"""Deterministic material, UV, and texture challenge fixtures."""

from __future__ import annotations

import hashlib
import random
from typing import Mapping

from .specs import ChallengeTag, FailureModeTag


MATERIAL_FIXTURE_KINDS: tuple[str, ...] = (
    "flat_color",
    "region_labeled",
    "procedural_stripes",
    "uv_checker_distortion",
    "normal_map_detail",
    "texture_only_high_frequency",
)


def material_fixture_payload(kind: str, seed: int) -> dict[str, object]:
    """Return a JSON-safe appearance declaration for a synthetic shape.

    The payload is intentionally shaped like ``evaluation.appearance`` input:
    it contains ``uv``, ``texture``, ``materials``, and
    ``appearance_attribution`` blocks so synthetic fixtures can double as
    expected metric fixtures without requiring Blender execution.
    """

    if kind not in MATERIAL_FIXTURE_KINDS:
        raise ValueError(f"Unknown material fixture kind: {kind}")
    rng = _rng(kind, seed)
    palette = _palette(rng)

    if kind == "flat_color":
        slots = [
            _slot(
                "mat_body_flat",
                palette[0],
                roughness=0.82,
                metallic=0.0,
                channels=("base_color", "roughness", "metallic"),
            )
        ]
        return _payload(
            kind,
            slots,
            uv={"has_uv_map": False, "uv_valid": None},
            texture={"texture_file_count": 0, "texture_memory_mb": 0.0},
            material_target="simple",
            challenges=(ChallengeTag.ANALYTIC_EXACT,),
            expected_failures=(),
            attribution={
                "boundary_geometry_fidelity": 0.9,
                "geometry_detail_score": 0.85,
                "texture_only_detail_score": 0.0,
            },
        )

    if kind == "region_labeled":
        slots = [
            _slot("mat_region_front", palette[0], channels=("base_color", "roughness", "metallic")),
            _slot("mat_region_side", palette[1], channels=("base_color", "roughness", "metallic")),
            _slot("mat_region_top", palette[2], channels=("base_color", "roughness", "metallic")),
            _slot("mat_region_detail", palette[3], channels=("base_color", "roughness", "metallic")),
        ]
        return _payload(
            kind,
            slots,
            uv=_valid_uv(islands=6, stretch=0.22),
            texture={"texture_file_count": 0, "texture_memory_mb": 0.0},
            material_target="simple",
            challenges=(ChallengeTag.MATERIAL_REGIONS,),
            expected_failures=(),
            attribution={
                "boundary_geometry_fidelity": 0.86,
                "geometry_detail_score": 0.78,
                "texture_only_detail_score": 0.05,
            },
        )

    if kind == "procedural_stripes":
        slots = [
            _slot(
                "mat_procedural_stripes",
                palette[0],
                channels=("base_color", "roughness", "metallic", "normal"),
                procedural=True,
                texture_maps=(
                    _texture_map("stripe_albedo", "base_color", 512, "checker_stripes"),
                    _texture_map("stripe_normal", "normal", 512, "stripe_normals"),
                ),
            )
        ]
        return _payload(
            kind,
            slots,
            uv=_valid_uv(islands=3, stretch=0.38),
            texture={
                "texture_width": 512,
                "texture_height": 512,
                "texture_file_count": 2,
                "texture_memory_mb": 2.0,
                "seam_visibility_score": 0.12,
            },
            material_target="pbr",
            challenges=(ChallengeTag.PBR_CHANNELS,),
            expected_failures=(),
            attribution={
                "boundary_geometry_fidelity": 0.82,
                "geometry_detail_score": 0.72,
                "texture_only_detail_score": 0.22,
            },
        )

    if kind == "uv_checker_distortion":
        slots = [
            _slot(
                "mat_uv_checker_distortion",
                palette[0],
                channels=("base_color", "roughness", "metallic"),
                texture_maps=(_texture_map("distorted_checker", "base_color", 1024, "checker_grid"),),
            )
        ]
        return _payload(
            kind,
            slots,
            uv={
                "has_uv_map": True,
                "uv_valid": False,
                "uv_island_count": 1,
                "uv_overlap_ratio": 0.34,
                "uv_out_of_bounds_ratio": 0.18,
                "uv_stretch_mean": 2.45,
                "texel_density_cv": 1.72,
                "missing_uv_faces": 0,
                "distortion": "overlap_and_out_of_bounds",
            },
            texture={
                "texture_width": 1024,
                "texture_height": 1024,
                "texture_file_count": 1,
                "texture_memory_mb": 4.0,
                "seam_visibility_score": 0.72,
            },
            material_target="pbr",
            challenges=(ChallengeTag.UV_STRETCH,),
            expected_failures=(FailureModeTag.UV_INVALID,),
            attribution={
                "boundary_geometry_fidelity": 0.74,
                "geometry_detail_score": 0.66,
                "texture_only_detail_score": 0.3,
            },
            strict_uv=True,
        )

    if kind == "normal_map_detail":
        slots = [
            _slot(
                "mat_normal_map_detail",
                palette[0],
                channels=("base_color", "roughness", "metallic", "normal"),
                texture_maps=(
                    _texture_map("normal_albedo", "base_color", 1024, "soft_gradient"),
                    _texture_map("normal_detail", "normal", 1024, "micro_normal"),
                ),
            )
        ]
        return _payload(
            kind,
            slots,
            uv=_valid_uv(islands=4, stretch=0.43),
            texture={
                "texture_width": 1024,
                "texture_height": 1024,
                "texture_file_count": 2,
                "texture_memory_mb": 8.0,
                "seam_visibility_score": 0.18,
            },
            material_target="pbr",
            challenges=(ChallengeTag.PBR_CHANNELS,),
            expected_failures=(),
            attribution={
                "boundary_geometry_fidelity": 0.82,
                "geometry_detail_score": 0.76,
                "texture_only_detail_score": 0.38,
            },
        )

    slots = [
        _slot(
            "mat_texture_only_high_frequency",
            palette[0],
            channels=("base_color", "roughness", "metallic", "normal"),
            texture_maps=(
                _texture_map("painted_edges", "base_color", 2048, "painted_fake_geometry"),
                _texture_map("painted_normals", "normal", 2048, "painted_fake_normals"),
            ),
        )
    ]
    return _payload(
        kind,
        slots,
        uv=_valid_uv(islands=2, stretch=0.52),
        texture={
            "texture_width": 2048,
            "texture_height": 2048,
            "texture_file_count": 2,
            "texture_memory_mb": 32.0,
            "seam_visibility_score": 0.1,
        },
        material_target="pbr",
        challenges=(ChallengeTag.TEXTURE_ONLY_DETAIL, ChallengeTag.PBR_CHANNELS),
        expected_failures=(FailureModeTag.TEXTURE_ONLY_HALLUCINATION,),
        attribution={
            "boundary_geometry_fidelity": 0.48,
            "geometry_detail_score": 0.33,
            "texture_only_detail_score": 0.92,
            "image_space_hallucination_warning": True,
        },
    )


def appearance_payload_from_materials(materials: Mapping[str, object]) -> dict[str, object]:
    """Extract the metric-shaped appearance payload from a material declaration."""

    if not materials:
        return {}
    keys = (
        "required",
        "strict_uv",
        "material_target",
        "max_texture_memory_mb",
        "uv",
        "texture",
        "materials",
        "appearance_attribution",
        "source",
    )
    return {key: materials[key] for key in keys if key in materials}


def appearance_expectations_from_materials(materials: Mapping[str, object]) -> dict[str, object]:
    """Return quality-target thresholds implied by a material declaration."""

    payload = appearance_payload_from_materials(materials)
    if not payload:
        return {}
    uv = _mapping(payload.get("uv"))
    texture = _mapping(payload.get("texture"))
    material_metrics = _mapping(payload.get("materials"))
    attribution = _mapping(payload.get("appearance_attribution"))
    pbr = _mapping(material_metrics.get("pbr_channel_coverage"))
    pbr_ratio = _coverage_ratio(pbr)
    return {
        "report_schema": "texture_material_report_v1",
        "required": bool(payload.get("required", False)),
        "strict_uv": bool(payload.get("strict_uv", False)),
        "material_target": str(payload.get("material_target", "pbr") or "pbr"),
        "has_uv_map_expected": _optional_bool(uv.get("has_uv_map")),
        "uv_valid_expected": _optional_bool(uv.get("uv_valid")),
        "uv_overlap_ratio_max": _optional_float(uv.get("uv_overlap_ratio")),
        "uv_out_of_bounds_ratio_max": _optional_float(uv.get("uv_out_of_bounds_ratio")),
        "uv_stretch_mean_max": _optional_float(uv.get("uv_stretch_mean")),
        "texture_file_count_expected": _optional_int(texture.get("texture_file_count")),
        "texture_memory_mb_expected": _optional_float(texture.get("texture_memory_mb")),
        "max_texture_memory_mb": _optional_float(payload.get("max_texture_memory_mb")),
        "material_slot_count_min": _optional_int(material_metrics.get("material_slot_count")),
        "named_material_ratio_min": _optional_float(material_metrics.get("named_material_ratio")),
        "pbr_channel_coverage_min": pbr_ratio,
        "image_space_hallucination_expected": _optional_bool(
            attribution.get("image_space_hallucination_warning")
        ),
        "expected_failure_modes": list(_string_tuple(materials.get("expected_failure_modes"))),
    }


def _payload(
    kind: str,
    material_slots: list[dict[str, object]],
    *,
    uv: Mapping[str, object],
    texture: Mapping[str, object],
    material_target: str,
    challenges: tuple[ChallengeTag, ...],
    expected_failures: tuple[FailureModeTag, ...],
    attribution: Mapping[str, object],
    strict_uv: bool = False,
) -> dict[str, object]:
    pbr_channels = _aggregate_pbr_channels(material_slots)
    return {
        "source": "synthetic_shape_factory",
        "fixture_kind": kind,
        "required": True,
        "strict_uv": strict_uv,
        "material_target": material_target,
        "max_texture_memory_mb": 16.0 if kind != "texture_only_high_frequency" else 64.0,
        "challenge_tags": [value.value for value in challenges],
        "expected_failure_modes": [value.value for value in expected_failures],
        "uv": dict(uv),
        "texture": dict(texture),
        "materials": {
            "material_slot_count": len(material_slots),
            "named_material_ratio": 1.0 if material_slots else 0.0,
            "procedural_material_ratio": _ratio(material_slots, "procedural"),
            "bitmap_material_ratio": _ratio(material_slots, "bitmap"),
            "pbr_channel_coverage": pbr_channels,
            "duplicate_material_count": 0,
            "orphan_texture_count": 0,
        },
        "material_slots": material_slots,
        "appearance_attribution": dict(attribution),
    }


def _slot(
    name: str,
    color: tuple[float, float, float, float],
    *,
    roughness: float = 0.58,
    metallic: float = 0.0,
    channels: tuple[str, ...],
    procedural: bool = False,
    texture_maps: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    return {
        "name": name,
        "base_color": [round(value, 4) for value in color],
        "roughness": round(roughness, 4),
        "metallic": round(metallic, 4),
        "pbr_channels": list(channels),
        "procedural": procedural,
        "bitmap": bool(texture_maps),
        "texture_maps": list(texture_maps),
    }


def _texture_map(name: str, semantic: str, resolution: int, procedural: str) -> dict[str, object]:
    return {
        "name": name,
        "semantic": semantic,
        "resolution": [int(resolution), int(resolution)],
        "procedural": procedural,
        "bytes_per_pixel": 4,
    }


def _valid_uv(*, islands: int, stretch: float) -> dict[str, object]:
    return {
        "has_uv_map": True,
        "uv_valid": True,
        "uv_island_count": islands,
        "uv_overlap_ratio": 0.0,
        "uv_out_of_bounds_ratio": 0.0,
        "uv_stretch_mean": stretch,
        "texel_density_cv": 0.28,
        "missing_uv_faces": 0,
        "distortion": "none",
    }


def _aggregate_pbr_channels(slots: list[dict[str, object]]) -> dict[str, bool]:
    channels = {"base_color": False, "roughness": False, "metallic": False, "normal": False}
    for slot in slots:
        for channel in slot.get("pbr_channels", ()):
            key = str(channel)
            if key in channels:
                channels[key] = True
    return channels


def _palette(rng: random.Random) -> tuple[tuple[float, float, float, float], ...]:
    colors = []
    for index in range(4):
        hue = (rng.random() + index * 0.21) % 1.0
        sat = 0.45 + rng.random() * 0.35
        val = 0.48 + rng.random() * 0.32
        colors.append((*_hsv_to_rgb(hue, sat, val), 1.0))
    return tuple(colors)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    values = (
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    )
    return tuple(round(channel, 4) for channel in values[i % 6])


def _rng(kind: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"material|{kind}|{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _ratio(slots: list[dict[str, object]], key: str) -> float:
    if not slots:
        return 0.0
    return sum(1 for slot in slots if bool(slot.get(key))) / len(slots)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _coverage_ratio(value: Mapping[str, object]) -> float | None:
    if not value:
        return None
    return sum(1 for item in value.values() if bool(item)) / len(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()
