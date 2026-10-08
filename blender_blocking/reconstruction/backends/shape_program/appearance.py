from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...artifacts import write_json
from ...backend import BackendBudget, BackendCapabilities, BaseBackend
from ...types import CandidateMetrics, CandidateRequest, CandidateResult, ProfileBand, ReconstructionTarget

try:  # Package import when called as blender_blocking.*
    from blender_blocking.primitives.shape_program import (
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from blender_blocking.primitives.grammar import default_shape_program_grammar
    from blender_blocking.primitives.program_search import search_shape_program_candidates
    from blender_blocking.primitives.shape_dsl import program_to_dsl
except Exception:  # pragma: no cover - legacy script import path
    from primitives.shape_program import (  # type: ignore
        ResidualPatch,
        ShapeConstraint,
        ShapeNode,
        ShapeProgram,
        validate_shape_program,
    )
    from primitives.grammar import default_shape_program_grammar  # type: ignore
    from primitives.program_search import search_shape_program_candidates  # type: ignore
    from primitives.shape_dsl import program_to_dsl  # type: ignore

_ROOT_STRATEGIES = {"profile_lathe", "bounds_box", "hybrid_profile_bounds"}
_RESIDUAL_POLICIES = {"ignore", "report", "suggest_patches"}
from .editability import _float


def _compiled_appearance_summary(
    compiled: Any,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    required = bool(config.get("evaluate_texture_materials", False))
    if compiled is None:
        if not required:
            return None
        return {
            "source": "shape_program_compiler",
            "required": True,
            "strict_uv": bool(config.get("uv_strict", False)),
            "material_target": str(config.get("material_target", "pbr")),
            "texture_reference_dir": config.get("texture_reference_dir"),
            "has_uv_map": False,
            "uv_valid": False,
            "warnings": ["appearance evaluation requested but program was not compiled"],
            "errors": ["compiled_blender_asset_missing"],
        }

    objects = tuple(getattr(compiled, "objects", ()) or ())
    mesh_objects = tuple(obj for obj in objects if _object_is_mesh(obj))
    uv_meshes = tuple(obj for obj in mesh_objects if _object_has_uv_map(obj))
    material_slots = _material_slots(objects)
    materials = tuple(_slot_material(slot) for slot in material_slots)
    material_names = tuple(
        str(getattr(material, "name", "") or "")
        for material in materials
        if material is not None
    )
    named_material_count = sum(
        1 for name in material_names if name and not name.lower().startswith("material")
    )
    unique_names = {name for name in material_names if name}
    texture_images = _texture_images(materials)
    texture_memory_mb = sum(_image_memory_mb(image) for image in texture_images)
    polygon_count = sum(_object_polygon_count(obj) for obj in mesh_objects)
    missing_uv_faces = 0
    for obj in mesh_objects:
        if not _object_has_uv_map(obj):
            missing_uv_faces += _object_polygon_count(obj)
    has_uv_map = bool(mesh_objects) and len(uv_meshes) == len(mesh_objects)
    uv_valid = has_uv_map and missing_uv_faces == 0
    pbr_channels = _pbr_channel_coverage(materials)
    material_slot_count = len(material_slots)
    named_ratio = (
        named_material_count / material_slot_count if material_slot_count else None
    )
    appearance = {
        "source": "shape_program_compiler",
        "required": required,
        "strict_uv": bool(config.get("uv_strict", False)),
        "material_target": str(config.get("material_target", "pbr")),
        "texture_reference_dir": config.get("texture_reference_dir"),
        "max_texture_memory_mb": _float(config.get("max_texture_memory_mb"), None),
        "uv": {
            "has_uv_map": has_uv_map,
            "uv_valid": uv_valid,
            "missing_uv_faces": missing_uv_faces,
        },
        "texture": {
            "texture_file_count": len(texture_images),
            "texture_memory_mb": texture_memory_mb,
        },
        "materials": {
            "material_slot_count": material_slot_count,
            "named_material_ratio": named_ratio,
            "pbr_channel_coverage": pbr_channels,
            "duplicate_material_count": max(0, len(material_names) - len(unique_names)),
            "orphan_texture_count": 0,
        },
        "appearance_attribution": {
            "boundary_geometry_fidelity": 0.75 if mesh_objects else 0.0,
            "geometry_detail_score": 0.8 if mesh_objects else 0.0,
            "texture_only_detail_score": 0.1 if texture_images else 0.0,
        },
    }
    warnings: list[str] = []
    errors: list[str] = []
    if required and not mesh_objects:
        errors.append("compiled_shape_program_has_no_mesh_objects")
    if required and not has_uv_map:
        errors.append("compiled_shape_program_uv_missing")
    elif mesh_objects and not has_uv_map:
        warnings.append("compiled_shape_program_uv_missing")
    if material_slot_count == 0:
        warnings.append("compiled_shape_program_materials_missing")
    if warnings:
        appearance["warnings"] = warnings
    if errors:
        appearance["errors"] = errors
    return appearance

def _object_is_mesh(obj: Any) -> bool:
    if getattr(obj, "type", None) == "MESH":
        return True
    data = getattr(obj, "data", None)
    return hasattr(data, "polygons") or hasattr(data, "vertices")

def _object_has_uv_map(obj: Any) -> bool:
    data = getattr(obj, "data", None)
    uv_layers = getattr(data, "uv_layers", None)
    if uv_layers is None:
        return False
    try:
        return len(uv_layers) > 0
    except TypeError:
        return bool(uv_layers)

def _object_polygon_count(obj: Any) -> int:
    polygons = getattr(getattr(obj, "data", None), "polygons", None)
    if polygons is None:
        return 0
    try:
        return len(polygons)
    except TypeError:
        return 0

def _material_slots(objects: Sequence[Any]) -> tuple[Any, ...]:
    slots: list[Any] = []
    for obj in objects:
        obj_slots = getattr(obj, "material_slots", ()) or ()
        try:
            slots.extend(list(obj_slots))
        except TypeError:
            continue
    return tuple(slots)

def _slot_material(slot: Any) -> Any:
    if slot is None:
        return None
    return getattr(slot, "material", slot)

def _texture_images(materials: Sequence[Any]) -> tuple[Any, ...]:
    images: list[Any] = []
    seen: set[int] = set()
    for material in materials:
        if material is None:
            continue
        node_tree = getattr(material, "node_tree", None)
        nodes = getattr(node_tree, "nodes", ()) if node_tree is not None else ()
        try:
            iterable = list(nodes)
        except TypeError:
            iterable = ()
        for node in iterable:
            image = getattr(node, "image", None)
            if image is None:
                continue
            identity = id(image)
            if identity not in seen:
                seen.add(identity)
                images.append(image)
    return tuple(images)

def _image_memory_mb(image: Any) -> float:
    size = getattr(image, "size", None)
    if not isinstance(size, Sequence) or len(size) < 2:
        return 0.0
    try:
        width = int(size[0])
        height = int(size[1])
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, width * height * 4 / (1024.0 * 1024.0))

def _pbr_channel_coverage(materials: Sequence[Any]) -> dict[str, bool]:
    if not materials:
        return {
            "base_color": False,
            "roughness": False,
            "metallic": False,
            "normal": False,
        }
    channel_hits = {
        "base_color": False,
        "roughness": False,
        "metallic": False,
        "normal": False,
    }
    for material in materials:
        if material is None:
            continue
        if getattr(material, "diffuse_color", None) is not None:
            channel_hits["base_color"] = True
        for attr, channel in (
            ("roughness", "roughness"),
            ("metallic", "metallic"),
        ):
            if getattr(material, attr, None) is not None:
                channel_hits[channel] = True
        node_tree = getattr(material, "node_tree", None)
        nodes = getattr(node_tree, "nodes", ()) if node_tree is not None else ()
        try:
            iterable = list(nodes)
        except TypeError:
            iterable = ()
        for node in iterable:
            node_text = f"{getattr(node, 'type', '')} {getattr(node, 'name', '')}".lower()
            if "normal" in node_text:
                channel_hits["normal"] = True
            if "roughness" in node_text:
                channel_hits["roughness"] = True
            if "metallic" in node_text:
                channel_hits["metallic"] = True
            if "base" in node_text or "color" in node_text or getattr(node, "image", None) is not None:
                channel_hits["base_color"] = True
    return channel_hits
