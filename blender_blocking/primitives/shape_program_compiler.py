"""Compile editable shape programs into Blender objects.

The compiler intentionally keeps every generated object editable: primitives
remain primitives where possible, bevel/smooth behavior is represented with
modifiers, and residual patches become named markers instead of being fused into
opaque mesh blobs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .shape_program import ResidualPatch, ShapeNode, ShapeProgram

try:
    import bpy

    BLENDER_AVAILABLE = True
except Exception:  # pragma: no cover - non-Blender environments
    bpy = None  # type: ignore
    BLENDER_AVAILABLE = False


@dataclass(frozen=True)
class CompiledShapeProgram:
    program_id: str
    root_object: Any = None
    objects: tuple[Any, ...] = ()
    residual_markers: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()

    def object_names(self) -> tuple[str, ...]:
        return tuple(getattr(obj, "name", "") for obj in self.objects)

    def marker_names(self) -> tuple[str, ...]:
        return tuple(getattr(obj, "name", "") for obj in self.residual_markers)

    def to_dict(self) -> dict[str, object]:
        return {
            "program_id": self.program_id,
            "root_object": getattr(self.root_object, "name", None),
            "objects": list(self.object_names()),
            "residual_markers": list(self.marker_names()),
            "warnings": list(self.warnings),
        }


def compile_shape_program(
    program: ShapeProgram,
    *,
    collection_name: str | None = None,
    lathe_segments: int = 48,
    bevel_modifier: bool = True,
    weighted_normals: bool = True,
) -> CompiledShapeProgram:
    """Create editable Blender objects for a ShapeProgram.

    Raises:
        RuntimeError: if called outside Blender.
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("shape program compilation requires Blender")
    collection = _ensure_collection(collection_name or f"ShapeProgram_{program.program_id}")
    warnings: list[str] = []
    objects = []
    for node in program.root_nodes:
        obj, node_warnings = _compile_node(
            node,
            lathe_segments=lathe_segments,
            bevel_modifier=bevel_modifier,
            weighted_normals=weighted_normals,
        )
        warnings.extend(node_warnings)
        _link_to_collection(obj, collection)
        _tag_object(obj, program=program, node=node)
        objects.append(obj)

    residual_markers = []
    for patch in program.residual_patches:
        marker = _compile_residual_marker(patch, program=program)
        _link_to_collection(marker, collection)
        residual_markers.append(marker)

    root_object = _make_root_empty(program, objects, residual_markers, collection)
    return CompiledShapeProgram(
        program_id=program.program_id,
        root_object=root_object,
        objects=tuple(objects),
        residual_markers=tuple(residual_markers),
        warnings=tuple(warnings),
    )


def _compile_node(
    node: ShapeNode,
    *,
    lathe_segments: int,
    bevel_modifier: bool,
    weighted_normals: bool,
) -> tuple[Any, tuple[str, ...]]:
    primitive = node.primitive_type or "empty"
    params = node.parameters
    name = node.name or node.node_id
    warnings: list[str] = []
    if primitive in {"box", "rounded_box"}:
        obj = _cube(name=name, params=params)
        if primitive == "rounded_box" and bevel_modifier:
            _add_bevel(obj, _float(params, "corner_radius_world", 0.02))
    elif primitive in {"cylinder", "frustum", "cone", "superquadric"}:
        obj = _cone_or_cylinder(name=name, params=params, vertices=lathe_segments)
        if primitive == "superquadric":
            _add_subdivision(obj, levels=1)
            warnings.append("superquadric compiled as editable ellipsoid-like proxy")
    elif primitive in {"sphere", "ellipsoid"}:
        obj = _sphere(name=name, params=params, segments=lathe_segments)
    elif primitive in {"lathe_profile", "loft_profile"}:
        obj = _lathe_profile_proxy(name=name, params=params, vertices=lathe_segments)
        if _profile_curve_rows(params):
            warnings.append(f"{primitive} compiled from preserved profile-band curve")
        else:
            warnings.append(f"{primitive} compiled from profile summary, not full row curve")
    elif primitive == "torus":
        obj = _torus(name=name, params=params, segments=lathe_segments)
    elif primitive in {"plane_patch", "residual_mesh_patch"}:
        obj = _plane_patch(name=name, params=params)
    else:
        obj = _empty(name=name)
        warnings.append(f"unsupported primitive {primitive!r} compiled as empty")

    if weighted_normals and hasattr(obj, "modifiers") and primitive not in {"empty"}:
        _add_weighted_normals(obj)
    obj["blendslop_shape_node_operation"] = node.operation
    obj["blendslop_shape_node_editable"] = bool(node.editable)
    return obj, tuple(warnings)


def _cube(*, name: str, params: Mapping[str, Any]) -> Any:
    size = 1.0
    location = _location(params)
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (
        _float(params, "width_world", _float(params, "width", 1.0)),
        _float(params, "depth_world", _float(params, "depth", 1.0)),
        _float(params, "height_world", _float(params, "height", 1.0)),
    )
    return obj


def _cone_or_cylinder(*, name: str, params: Mapping[str, Any], vertices: int) -> Any:
    width = _float(params, "width_world", _float(params, "diameter", 1.0))
    depth = _float(params, "depth_world", width)
    radius = max(width, depth) * 0.5
    radius1 = _float(params, "radius_bottom", _float(params, "radius", radius))
    radius2 = _float(params, "radius_top", radius1)
    height = _float(params, "height_world", _float(params, "height", 1.0))
    bpy.ops.mesh.primitive_cone_add(
        vertices=max(8, int(vertices)),
        radius1=radius1,
        radius2=radius2,
        depth=height,
        location=_location(params),
    )
    obj = bpy.context.active_object
    obj.name = name
    if radius > 0:
        obj.scale.x = max(width * 0.5, 1e-6) / radius
        obj.scale.y = max(depth * 0.5, 1e-6) / radius
    return obj


def _sphere(*, name: str, params: Mapping[str, Any], segments: int) -> Any:
    width = _float(params, "width_world", _float(params, "diameter", 1.0))
    depth = _float(params, "depth_world", width)
    height = _float(params, "height_world", width)
    radius = max(width, depth, height) * 0.5
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=max(12, int(segments)),
        ring_count=max(6, int(segments // 2)),
        radius=max(radius, 1e-6),
        location=_location(params),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (
        max(width * 0.5, 1e-6) / max(radius, 1e-6),
        max(depth * 0.5, 1e-6) / max(radius, 1e-6),
        max(height * 0.5, 1e-6) / max(radius, 1e-6),
    )
    return obj


def _lathe_profile_proxy(
    *,
    name: str,
    params: Mapping[str, Any],
    vertices: int,
) -> Any:
    profile_curve = _profile_curve_rows(params)
    if profile_curve:
        return _lathe_profile_mesh(
            name=name,
            rows=profile_curve,
            vertices=vertices,
            params=params,
        )
    width = _float(params, "width_world", _float(params, "mean_width_px", 1.0))
    depth = _float(params, "depth_world", width)
    height = _float(params, "height_world", 1.0)
    radius = max(width, depth) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=max(12, int(vertices)),
        radius=max(radius, 1e-6),
        depth=max(height, 1e-6),
        location=_location(params),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale.x = max(width * 0.5, 1e-6) / max(radius, 1e-6)
    obj.scale.y = max(depth * 0.5, 1e-6) / max(radius, 1e-6)
    obj["blendslop_profile_band_count"] = int(_float(params, "band_count", 0.0))
    obj["blendslop_profile_confidence"] = _float(params, "confidence", 0.0)
    if bool(params.get("preserves_hole_hints")) or bool(
        params.get("preserves_multiple_intervals")
    ):
        _add_wire_overlay_modifier(obj)
    return obj


def _profile_curve_rows(params: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    curve = params.get("profile_curve")
    if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes, bytearray)):
        return ()
    rows = tuple(item for item in curve if isinstance(item, Mapping))
    return tuple(sorted(rows, key=lambda item: _float(item, "z_world", 0.0)))


def _lathe_profile_mesh(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    vertices: int,
    params: Mapping[str, Any],
) -> Any:
    segments = max(12, int(vertices))
    mesh_vertices: list[tuple[float, float, float]] = []
    mesh_faces: list[tuple[int, ...]] = []
    for row in rows:
        z = _float(row, "z_world", 0.0)
        radius_x = max(_float(row, "radius_x_world", 0.0), 1e-6)
        radius_y = max(_float(row, "radius_y_world", radius_x), 1e-6)
        center_offset = _float(row, "center_offset_world", 0.0)
        for index in range(segments):
            theta = (float(index) / float(segments)) * math.tau
            mesh_vertices.append(
                (
                    center_offset + math.cos(theta) * radius_x,
                    math.sin(theta) * radius_y,
                    z,
                )
            )

    ring_count = len(rows)
    for ring in range(max(0, ring_count - 1)):
        ring_start = ring * segments
        next_start = (ring + 1) * segments
        for index in range(segments):
            mesh_faces.append(
                (
                    ring_start + index,
                    ring_start + ((index + 1) % segments),
                    next_start + ((index + 1) % segments),
                    next_start + index,
                )
            )
    if ring_count:
        bottom_center = len(mesh_vertices)
        bottom_z = _float(rows[0], "z_world", 0.0)
        bottom_offset = _float(rows[0], "center_offset_world", 0.0)
        mesh_vertices.append((bottom_offset, 0.0, bottom_z))
        top_center = len(mesh_vertices)
        top_z = _float(rows[-1], "z_world", 0.0)
        top_offset = _float(rows[-1], "center_offset_world", 0.0)
        mesh_vertices.append((top_offset, 0.0, top_z))
        last_ring = (ring_count - 1) * segments
        for index in range(segments):
            mesh_faces.append(
                (
                    bottom_center,
                    (index + 1) % segments,
                    index,
                )
            )
            mesh_faces.append(
                (
                    top_center,
                    last_ring + index,
                    last_ring + ((index + 1) % segments),
                )
            )

    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(mesh_vertices, [], mesh_faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = _location(params)
    obj["blendslop_profile_curve_rows"] = len(rows)
    obj["blendslop_profile_curve_segments"] = segments
    obj["blendslop_profile_curve_confidence_mean"] = _mean(
        _float(row, "confidence", 0.0) for row in rows
    )
    return obj


def _torus(*, name: str, params: Mapping[str, Any], segments: int) -> Any:
    major = _float(params, "major_radius", _float(params, "width_world", 1.0) * 0.25)
    minor = _float(params, "minor_radius", max(major * 0.25, 1e-6))
    bpy.ops.mesh.primitive_torus_add(
        major_segments=max(12, int(segments)),
        minor_segments=max(6, int(segments // 4)),
        major_radius=major,
        minor_radius=minor,
        location=_location(params),
    )
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _plane_patch(*, name: str, params: Mapping[str, Any]) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=_location(params))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (
        _float(params, "width_world", 0.1),
        _float(params, "depth_world", 0.01),
        _float(params, "height_world", 0.1),
    )
    obj.display_type = "WIRE"
    return obj


def _compile_residual_marker(patch: ResidualPatch, *, program: ShapeProgram) -> Any:
    bpy.ops.object.empty_add(type="CUBE", location=(0.0, 0.0, 0.0))
    marker = bpy.context.active_object
    marker.name = f"Residual_{patch.patch_id}"
    marker.empty_display_size = 0.15 + max(0.0, min(1.0, patch.confidence)) * 0.25
    marker["blendslop_shape_program_id"] = program.program_id
    marker["blendslop_residual_patch_id"] = patch.patch_id
    marker["blendslop_residual_category"] = patch.category
    marker["blendslop_residual_source_view"] = patch.source_view
    marker["blendslop_residual_confidence"] = float(patch.confidence)
    marker["blendslop_residual_notes"] = "\n".join(patch.notes)
    return marker


def _make_root_empty(
    program: ShapeProgram,
    objects: Sequence[Any],
    residual_markers: Sequence[Any],
    collection: Any,
) -> Any:
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    root = bpy.context.active_object
    root.name = f"ShapeProgramRoot_{program.program_id}"
    root.empty_display_size = 0.5
    root["blendslop_shape_program_id"] = program.program_id
    root["blendslop_shape_program_schema"] = program.schema_version
    root["blendslop_shape_program_node_count"] = program.node_count()
    root["blendslop_shape_program_residual_patch_count"] = program.residual_patch_count()
    _link_to_collection(root, collection)
    for obj in tuple(objects) + tuple(residual_markers):
        obj.parent = root
    return root


def _ensure_collection(name: str) -> Any:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _link_to_collection(obj: Any, collection: Any) -> None:
    if obj.name not in collection.objects:
        try:
            collection.objects.link(obj)
        except RuntimeError:
            pass
    for old_collection in tuple(obj.users_collection):
        if old_collection != collection:
            try:
                old_collection.objects.unlink(obj)
            except RuntimeError:
                pass


def _tag_object(obj: Any, *, program: ShapeProgram, node: ShapeNode) -> None:
    obj["blendslop_shape_program_id"] = program.program_id
    obj["blendslop_shape_node_id"] = node.node_id
    obj["blendslop_shape_node_name"] = node.name or node.node_id
    obj["blendslop_shape_node_primitive_type"] = node.primitive_type or ""


def _add_bevel(obj: Any, amount: float) -> None:
    modifier = obj.modifiers.new("Blendslop editable bevel", "BEVEL")
    modifier.width = max(0.0, float(amount))
    modifier.segments = 3
    try:
        modifier.affect = "EDGES"
    except Exception:
        pass


def _add_weighted_normals(obj: Any) -> None:
    modifier = obj.modifiers.new("Blendslop weighted normals", "WEIGHTED_NORMAL")
    modifier.keep_sharp = True


def _add_subdivision(obj: Any, *, levels: int) -> None:
    modifier = obj.modifiers.new("Blendslop editable subdivision", "SUBSURF")
    modifier.levels = max(0, int(levels))
    modifier.render_levels = max(0, int(levels))


def _add_wire_overlay_modifier(obj: Any) -> None:
    modifier = obj.modifiers.new("Blendslop profile uncertainty wire", "WIREFRAME")
    modifier.thickness = 0.005
    modifier.use_even_offset = True


def _empty(*, name: str) -> Any:
    bpy.ops.object.empty_add(type="CUBE", location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _location(params: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        _float(params, "x", _float(params, "location_x", 0.0)),
        _float(params, "y", _float(params, "location_y", 0.0)),
        _float(params, "z", _float(params, "location_z", 0.0)),
    )


def _float(params: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    collected = tuple(float(value) for value in values)
    return sum(collected) / float(len(collected)) if collected else 0.0
