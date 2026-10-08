"""Lazy Blender mesh builders for synthetic shape specs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
from typing import Any

from .silhouette_render_jobs import canonical_view_jobs
from .specs import ShapeFamily, SyntheticShapeSpec


class BlenderUnavailableError(RuntimeError):
    """Raised when Blender-only generation is requested outside Blender."""


def require_bpy() -> Any:
    try:
        import bpy
    except ImportError as exc:
        raise BlenderUnavailableError(
            "Blender Python module 'bpy' is unavailable. Run this builder inside Blender "
            "with --background after all source edits are complete."
        ) from exc
    return bpy


def build_mesh_object(spec: SyntheticShapeSpec, name: str | None = None) -> Any:
    bpy = require_bpy()
    object_name = name or spec.shape_id
    if spec.family == ShapeFamily.ANALYTIC_PRIMITIVE.value:
        obj = _build_analytic(bpy, spec)
    elif spec.family == ShapeFamily.PROFILE_LATHE.value:
        obj = _build_lathe(bpy, spec)
    elif spec.family in (ShapeFamily.FURNITURE.value, ShapeFamily.VEHICLE_MECHANICAL.value):
        obj = _build_compound(bpy, spec)
    else:
        raise BlenderUnavailableError(f"Spec family {spec.family!r} is a pure mask family and has no Blender mesh.")
    obj.name = object_name
    _apply_transform(obj, spec)
    _apply_material_declaration(bpy, obj, spec)
    return obj


def export_mesh(spec: SyntheticShapeSpec, output_dir: Path) -> dict[str, Path]:
    bpy = require_bpy()
    obj = build_mesh_object(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    blend_path = output_dir / "source.blend"
    obj_path = output_dir / "ground_truth.obj"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    paths["source_blend"] = blend_path
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=str(obj_path), export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=str(obj_path), use_selection=True)
    paths["ground_truth_obj"] = obj_path
    return paths


def render_views(
    spec: SyntheticShapeSpec,
    output_dir: Path,
    resolution: tuple[int, int] = (256, 256),
    include_orbit: bool = False,
    orbit_angles: Sequence[float] | None = None,
) -> dict[str, Path]:
    bpy = require_bpy()
    try:
        from integration.blender_ops.camera_framing import compute_bounds_world
        from integration.blender_ops.silhouette_render import (
            render_silhouette_frame,
            set_camera_orbit,
            set_camera_top,
            silhouette_session,
        )
    except ImportError as exc:
        raise BlenderUnavailableError(
            "Blender silhouette render helpers are unavailable on sys.path."
        ) from exc

    obj = build_mesh_object(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds_min, bounds_max = compute_bounds_world([obj])
    center = (bounds_min + bounds_max) / 2.0
    width = bounds_max.x - bounds_min.x
    depth = bounds_max.y - bounds_min.y
    height = bounds_max.z - bounds_min.z
    max_dim = max(width, depth, height, 1e-3)
    distance = max_dim * 2.0
    paths: dict[str, Path] = {}

    with silhouette_session(
        target_objects=[obj],
        resolution=resolution,
        color_mode="BW",
        transparent_bg=False,
        engine="BLENDER_EEVEE",
        background_color=(1.0, 1.0, 1.0, 1.0),
        silhouette_color=(0.0, 0.0, 0.0, 1.0),
    ) as session:
        for view in canonical_view_jobs(
            spec,
            resolution=resolution,
            include_orbit=include_orbit,
            orbit_angles=orbit_angles,
        ):
            path = output_dir / f"{view.view_name}.png"
            if view.view_name in {"front", "side", "top"}:
                from integration.blender_ops.camera_framing import configure_ortho_camera_for_view
                configure_ortho_camera_for_view(session.camera, view.view_name, bounds_min, bounds_max,
                    margin_frac=view.padding, resolution=resolution)
                # Common image scale across canonical captures; different extents do not rescale the object.
                session.camera.data.ortho_scale = max_dim*(1.+2.*view.padding)
            else:
                import math

                scale = max_dim * (1.0 + 2.0 * view.padding)
                set_camera_orbit(session.camera, center, distance, math.radians(view.azimuth_deg), scale)
                session.camera.rotation_euler.rotate_axis("Z", math.radians(view.roll_deg))
            render_silhouette_frame(session, path)
            paths[view.view_name] = path
    from blender_blocking.evaluation.comparable_geometry import export_evaluated_object
    paths['ground_truth_obj'] = export_evaluated_object(obj, output_dir / 'ground_truth.obj')
    bpy.data.objects.remove(obj, do_unlink=True)
    return paths


def _build_analytic(bpy: Any, spec: SyntheticShapeSpec) -> Any:
    p = spec.parameters
    primitive = p["primitive"]
    if primitive == "box":
        dims = p["dimensions"]
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.dimensions = (float(dims["width"]), float(dims["depth"]), float(dims["height"]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj
    if primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=float(p["radius"]))
        return bpy.context.active_object
    if primitive == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
        obj = bpy.context.active_object
        obj.scale = tuple(float(x) for x in p["radii"])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj
    if primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=float(p["radius"]), depth=float(p["height"]))
        return bpy.context.active_object
    if primitive in ("frustum", "cone"):
        bpy.ops.mesh.primitive_cone_add(
            vertices=96,
            radius1=float(p.get("radius_bottom", p.get("radius", 0.5))),
            radius2=float(p.get("radius_top", 0.0)),
            depth=float(p["height"]),
        )
        return bpy.context.active_object
    if primitive == "capsule":
        return _capsule_mesh(bpy, float(p["radius"]), float(p["segment_height"]))
    if primitive == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_segments=128,
            minor_segments=24,
            major_radius=float(p["major_radius"]),
            minor_radius=float(p["minor_radius"]),
        )
        return bpy.context.active_object
    if primitive == "rounded_box":
        dims = p["dimensions"]
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.dimensions = (float(dims["width"]), float(dims["depth"]), float(dims["height"]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        modifier = obj.modifiers.new(name="synthetic_rounding", type="BEVEL")
        modifier.width = float(p["radius"])
        modifier.segments = 8
        obj.modifiers.new(name="synthetic_weighted_normals", type="WEIGHTED_NORMAL")
        return obj
    if primitive == "superquadric":
        return _superquadric_mesh(bpy, p)
    raise ValueError(f"Unknown analytic primitive {primitive!r}")


def _build_lathe(bpy: Any, spec: SyntheticShapeSpec) -> Any:
    import math

    profile = spec.parameters["profile"]
    height = float(spec.parameters["height"])
    segments = int(spec.parameters.get("segments", 96))
    verts = []
    faces = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        for row in profile:
            z = float(row[0]) * height / 2.0
            radius_value = row[1]
            radius = float(radius_value[0] if isinstance(radius_value, list) else radius_value)
            offset_x = float(row[2]) if len(row) > 2 else 0.0
            verts.append((offset_x + math.cos(theta) * radius, math.sin(theta) * radius, z))
    rows = len(profile)
    for i in range(segments):
        ni = (i + 1) % segments
        for j in range(rows - 1):
            faces.append((i * rows + j, ni * rows + j, ni * rows + j + 1, i * rows + j + 1))
    mesh = bpy.data.meshes.new(f"{spec.shape_id}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(spec.shape_id, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def _build_compound(bpy: Any, spec: SyntheticShapeSpec) -> Any:
    part_entries = []
    for part_spec in spec.parameters.get("parts", []):
        obj = _part_object(bpy, part_spec)
        part_entries.append((part_spec, obj))
    if not part_entries:
        raise ValueError(f"Compound spec {spec.shape_id} has no parts")
    root = part_entries[0][1]
    for part_spec, part_obj in part_entries[1:]:
        part_name = str(part_obj.name)
        if part_spec.get("boolean") == "subtract":
            modifier = root.modifiers.new(name=f"subtract_{part_name}", type="BOOLEAN")
            modifier.operation = "DIFFERENCE"
        else:
            modifier = root.modifiers.new(name=f"union_{part_name}", type="BOOLEAN")
            modifier.operation = "UNION"
        modifier.object = part_obj
        bpy.context.view_layer.objects.active = root
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(part_obj, do_unlink=True)
        except Exception:
            pass
    return root


def _part_object(bpy: Any, part: dict[str, Any]) -> Any:
    kind = part["type"]
    center = tuple(float(x) for x in part.get("center", (0, 0, 0)))
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
        obj = bpy.context.active_object
        obj.dimensions = tuple(float(x) for x in part["size"])
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=float(part["radius"]), location=center)
        obj = bpy.context.active_object
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=float(part["radius"]), depth=float(part["height"]), location=center)
        obj = bpy.context.active_object
    elif kind == "frustum":
        bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=float(part["radius_bottom"]), radius2=float(part["radius_top"]), depth=float(part["height"]), location=center)
        obj = bpy.context.active_object
    elif kind == "capsule":
        obj = _capsule_mesh(bpy, float(part["radius"]), float(part["segment_height"]))
        obj.location = center
    elif kind == "torus_section":
        bpy.ops.mesh.primitive_torus_add(major_radius=float(part["major_radius"]), minor_radius=float(part["minor_radius"]), location=center)
        obj = bpy.context.active_object
    elif kind == "gear":
        obj = _gear_mesh(bpy, part)
        obj.location = center
    else:
        raise ValueError(f"Unknown compound part type {kind!r}")
    obj.name = str(part.get("name", kind))
    if "rotation_deg" in part:
        obj.rotation_euler = tuple(_deg_to_rad(float(x)) for x in part["rotation_deg"])
    return obj


def _capsule_mesh(bpy: Any, radius: float, segment_height: float) -> Any:
    from blender_blocking.primitives.capsule import CapsulePrimitive
    data = CapsulePrimitive(radius=radius, segment_height=segment_height).to_mesh_data(48)
    mesh = bpy.data.meshes.new("synthetic_capsule")
    mesh.from_pydata(data.vertices.tolist(), [], data.faces)
    mesh.update()
    obj = bpy.data.objects.new("synthetic_capsule", mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def _superquadric_mesh(bpy: Any, params: dict[str, Any]) -> Any:
    import math

    radii = [float(x) for x in params["radii"]]
    exponents = [float(x) for x in params["exponents"]]
    taper = float(params.get("taper", 0.0))
    u_steps = 48
    v_steps = 24
    verts = []
    faces = []
    for vi in range(v_steps + 1):
        v = -math.pi / 2 + math.pi * vi / v_steps
        cv = _signed_pow(math.cos(v), exponents[1])
        sv = _signed_pow(math.sin(v), exponents[1])
        for ui in range(u_steps):
            u = -math.pi + 2 * math.pi * ui / u_steps
            cu = _signed_pow(math.cos(u), exponents[0])
            su = _signed_pow(math.sin(u), exponents[0])
            scale = max(0.15, 1.0 + taper * sv)
            verts.append((radii[0] * scale * cv * cu, radii[1] * scale * cv * su, radii[2] * sv))
    for vi in range(v_steps):
        for ui in range(u_steps):
            n_ui = (ui + 1) % u_steps
            faces.append((vi * u_steps + ui, vi * u_steps + n_ui, (vi + 1) * u_steps + n_ui, (vi + 1) * u_steps + ui))
    mesh = bpy.data.meshes.new("synthetic_superquadric")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("synthetic_superquadric", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def _gear_mesh(bpy: Any, part: dict[str, Any]) -> Any:
    import math

    teeth = int(part["teeth"])
    root = float(part["root_radius"])
    tip = float(part["tip_radius"])
    height = float(part["height"])
    hole_radius = float(part.get("hole_radius", 0.0))
    center = tuple(float(value) for value in part.get("center", (0.0, 0.0, 0.0)))
    inner = max(0.0, min(hole_radius, root * 0.9))
    n = teeth * 2
    verts = []
    for z in (-height / 2.0, height / 2.0):
        for i in range(n):
            radius = tip if i % 2 == 0 else root
            theta = 2.0 * math.pi * i / (teeth * 2)
            verts.append(
                (
                    center[0] + math.cos(theta) * radius,
                    center[1] + math.sin(theta) * radius,
                    center[2] + z,
                )
            )
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            verts.append(
                (
                    center[0] + math.cos(theta) * inner,
                    center[1] + math.sin(theta) * inner,
                    center[2] + z,
                )
            )
    faces = []
    bottom_outer = 0
    bottom_inner = n
    top_outer = n * 2
    top_inner = n * 3
    for i in range(n):
        j = (i + 1) % n
        faces.append((bottom_outer + i, bottom_outer + j, top_outer + j, top_outer + i))
        if inner > 0.0:
            faces.append((bottom_inner + j, bottom_inner + i, top_inner + i, top_inner + j))
            faces.append((top_outer + i, top_outer + j, top_inner + j, top_inner + i))
            faces.append((bottom_outer + j, bottom_outer + i, bottom_inner + i, bottom_inner + j))
        else:
            faces.append((top_outer + i, top_outer + j, top_inner + j))
            faces.append((bottom_outer + j, bottom_outer + i, bottom_inner + i))
    mesh = bpy.data.meshes.new("synthetic_gear")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("synthetic_gear", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def _apply_transform(obj: Any, spec: SyntheticShapeSpec) -> None:
    transform = spec.transforms
    if "position" in transform:
        obj.location = tuple(float(x) for x in transform["position"])
    if "rotation_deg" in transform:
        obj.rotation_euler = tuple(_deg_to_rad(float(x)) for x in transform["rotation_deg"])
    if "scale" in transform:
        obj.scale = tuple(float(x) for x in transform["scale"])


def _apply_material_declaration(bpy: Any, obj: Any, spec: SyntheticShapeSpec) -> None:
    materials = spec.materials
    if not materials:
        return
    data = getattr(obj, "data", None)
    if data is None:
        return

    material_slots = materials.get("material_slots")
    if isinstance(material_slots, list) and material_slots:
        _clear_mesh_materials(data)
        created = [_create_declared_material(bpy, dict(slot)) for slot in material_slots if isinstance(slot, dict)]
        for material in created:
            data.materials.append(material)
        _assign_declared_material_indices(data, max(1, len(created)))
    elif "base_color" in materials:
        _clear_mesh_materials(data)
        data.materials.append(
            _create_declared_material(
                bpy,
                {
                    "name": f"{spec.shape_id}_material",
                    "base_color": materials.get("base_color"),
                    "roughness": materials.get("roughness", 0.55),
                    "metallic": materials.get("metallic", 0.0),
                    "pbr_channels": ["base_color", "roughness", "metallic"],
                },
            )
        )

    uv_decl = materials.get("uv")
    if isinstance(uv_decl, dict) and bool(uv_decl.get("has_uv_map", False)):
        _ensure_declared_uvs(data, str(uv_decl.get("distortion", "none")))


def _clear_mesh_materials(data: Any) -> None:
    try:
        data.materials.clear()
        return
    except Exception:
        pass
    try:
        while len(data.materials):
            data.materials.pop(index=0)
    except Exception:
        return


def _create_declared_material(bpy: Any, declaration: dict[str, Any]) -> Any:
    name = str(declaration.get("name") or "synthetic_material")
    material = bpy.data.materials.new(name=name)
    color = _rgba_tuple(declaration.get("base_color"), default=(0.35, 0.35, 0.35, 1.0))
    material.diffuse_color = color
    for attr in ("roughness", "metallic"):
        try:
            setattr(material, attr, float(declaration.get(attr, 0.0 if attr == "metallic" else 0.55)))
        except Exception:
            pass

    try:
        material.use_nodes = True
        nodes = material.node_tree.nodes
        principled = _find_principled_node(nodes)
        if principled is not None:
            _set_node_input(principled, "Base Color", color)
            _set_node_input(principled, "Roughness", float(declaration.get("roughness", 0.55)))
            _set_node_input(principled, "Metallic", float(declaration.get("metallic", 0.0)))
        for texture in declaration.get("texture_maps", ()) or ():
            if isinstance(texture, dict):
                _add_texture_node(bpy, material, texture)
    except Exception:
        pass
    return material


def _find_principled_node(nodes: Any) -> Any | None:
    try:
        for node in nodes:
            node_type = str(getattr(node, "type", "")).upper()
            node_name = str(getattr(node, "name", "")).lower()
            if node_type == "BSDF_PRINCIPLED" or "principled" in node_name:
                return node
    except Exception:
        return None
    return None


def _set_node_input(node: Any, name: str, value: Any) -> None:
    try:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
    except Exception:
        return


def _add_texture_node(bpy: Any, material: Any, texture: dict[str, Any]) -> None:
    nodes = material.node_tree.nodes
    node = nodes.new(type="ShaderNodeTexImage")
    semantic = str(texture.get("semantic", "base_color"))
    node.name = f"synthetic_{semantic}_{texture.get('name', 'texture')}"
    resolution = texture.get("resolution", [512, 512])
    width = max(1, int(resolution[0]))  # type: ignore[index]
    height = max(1, int(resolution[1]))  # type: ignore[index]
    image = bpy.data.images.new(
        name=str(texture.get("name", "synthetic_texture")),
        width=width,
        height=height,
        alpha=True,
    )
    node.image = image
    if semantic == "normal":
        normal_node = nodes.new(type="ShaderNodeNormalMap")
        normal_node.name = f"synthetic_normal_{texture.get('name', 'texture')}"


def _assign_declared_material_indices(data: Any, material_count: int) -> None:
    polygons = getattr(data, "polygons", None)
    if polygons is None or material_count <= 1:
        return
    try:
        for index, polygon in enumerate(polygons):
            polygon.material_index = index % material_count
    except Exception:
        return


def _ensure_declared_uvs(data: Any, distortion: str) -> None:
    uv_layers = getattr(data, "uv_layers", None)
    loops = getattr(data, "loops", None)
    vertices = getattr(data, "vertices", None)
    if uv_layers is None or loops is None or vertices is None:
        return
    try:
        layer = uv_layers.active or uv_layers.new(name="synthetic_uv")
    except Exception:
        return
    coords = [vertex.co for vertex in vertices]
    if not coords:
        return
    min_x = min(float(co.x) for co in coords)
    max_x = max(float(co.x) for co in coords)
    min_y = min(float(co.y) for co in coords)
    max_y = max(float(co.y) for co in coords)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    try:
        for loop_index, loop in enumerate(loops):
            co = vertices[loop.vertex_index].co
            u = (float(co.x) - min_x) / span_x
            v = (float(co.y) - min_y) / span_y
            if distortion == "overlap_and_out_of_bounds":
                if loop_index % 5 == 0:
                    u += 1.25
                if loop_index % 7 == 0:
                    v -= 0.35
                if loop_index % 11 == 0:
                    u, v = 0.5, 0.5
            layer.data[loop_index].uv = (u, v)
    except Exception:
        return


def _rgba_tuple(value: Any, *, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return default
    values = [float(item) for item in value[:4]]
    if len(values) == 3:
        values.append(1.0)
    return tuple(values[:4])  # type: ignore[return-value]


def _signed_pow(value: float, exponent: float) -> float:
    return (1.0 if value >= 0 else -1.0) * (abs(value) ** exponent)


def _deg_to_rad(value: float) -> float:
    import math

    return value * math.pi / 180.0
