"""Compile editable shape programs into Blender objects.

The compiler intentionally keeps every generated object editable: primitives
remain primitives where possible, bevel/smooth behavior is represented with
modifiers, and residual patches remain named, inspectable markers even when the
program also contains editable suggested patch nodes for them.
"""

from __future__ import annotations

import math
import numpy as np
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
    union_report: Any = None
    output_contract: str = "live_editable_program"
    final_qualification: Any = None

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
            "native_union": self.union_report,
            "output_contract": self.output_contract,
            "final_output_qualification": self.final_qualification,
        }


def compile_shape_program(
    program: ShapeProgram,
    *,
    collection_name: str | None = None,
    lathe_segments: int = 48,
    bevel_modifier: bool = True,
    weighted_normals: bool = True,
    csg_options: Mapping[str, Any] | None = None,
    executor=None,
    timeout_s: float | None = None,
) -> CompiledShapeProgram:
    """Create editable Blender objects for a ShapeProgram.

    Raises:
        RuntimeError: if called outside Blender.
    """
    from .shape_program import validate_compilable_program
    errors = validate_compilable_program(program)
    if errors:
        raise ValueError("shape program cannot compile: " + "; ".join(errors))
    if not BLENDER_AVAILABLE:
        raise RuntimeError("shape program compilation requires Blender")
    collection = _ensure_collection(collection_name or f"ShapeProgram_{program.program_id}")
    warnings: list[str] = []
    csg_options = csg_options or {}
    union_report = None
    output_contract = "live_editable_program"
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

    operations = [(node,obj) for node,obj in zip(program.root_nodes,objects) if obj.type == "MESH"]
    if csg_options.get("native_union_execution", False) or any(node.operation in {"subtract","intersect","difference","intersection"} for node,obj in operations):
        from blender_blocking.reconstruction.native_geometry import NativeOwnedGeometry, evaluated_arrays, GeometryArrays
        from blender_blocking.reconstruction.native_csg import boolean_mesh
        positives = [evaluated_arrays(obj) for node,obj in operations if node.operation in {"add","union"}]
        if not positives:
            raise ValueError("CSG program requires a positive base")
        from blender_blocking.reconstruction.grouped_solids import production_union
        # A primitive box's minimum dimension is a declared feature scale. It is
        # unavailable for arbitrary profiles/convex meshes; no guessed SDF bound.
        positive_nodes = [node for node, obj in operations if node.operation in {'add', 'union'}]
        thickness = None
        if all(node.primitive_type == 'box' for node in positive_nodes) and not bevel_modifier:
            thickness = min(float(node.parameters.get(key, 1.)) for node in positive_nodes
                            for key in ('width_world', 'depth_world', 'height_world'))
        data, union_report = production_union(positives, csg_options, executor=executor,
                                             timeout_s=timeout_s, feature_thickness=thickness)
        for node,obj in operations:
            if node.operation in {"subtract","intersect","difference","intersection"}:
                data, _ = boolean_mesh(data,evaluated_arrays(obj),operation="DIFFERENCE" if node.operation in {"subtract", "difference"} else "INTERSECT")
        # Retain a live native modifier tree. Changing an editable positive or
        # negative node must change the evaluated output without a recompile.
        positive_objects = [obj for node,obj in operations if node.operation in {"add", "union"}]
        intermediates = []
        def live_sources(sources, name):
            output = bpy.data.objects.new(name, bpy.data.meshes.new(name+"Mesh"))
            _link_to_collection(output, collection)
            graph = bpy.data.node_groups.new(name+"Graph", "GeometryNodeTree")
            graph.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            join = graph.nodes.new("GeometryNodeJoinGeometry")
            for source in sources:
                info = graph.nodes.new("GeometryNodeObjectInfo")
                info.transform_space = "RELATIVE"
                info.inputs["Object"].default_value = source
                info.inputs["As Instance"].default_value = False
                graph.links.new(info.outputs["Geometry"], join.inputs["Geometry"])
            final = graph.nodes.new("NodeGroupOutput")
            graph.links.new(join.outputs["Geometry"], final.inputs["Geometry"])
            output.modifiers.new("EditableBaseGeometry", "NODES").node_group = graph
            return output

        # The live tree follows the same spatial grouping and balanced pairing
        # as numeric CSG. Source nodes remain editable; disconnected groups join.
        from blender_blocking.reconstruction.grouped_solids import overlap_groups
        source_by_identity = {id(part): obj for part, obj in zip(positives, positive_objects)}
        group_outputs = []
        for group in overlap_groups(positives):
            level = [source_by_identity[id(part)] for part in group]
            while len(level) > 1:
                next_level = []
                for index in range(0, len(level)-1, 2):
                    branch = live_sources([level[index]], "EditableUnionBranch")
                    modifier = branch.modifiers.new("EditableUnion", "BOOLEAN")
                    modifier.solver = "EXACT"
                    modifier.operation = "UNION"
                    modifier.object = level[index+1]
                    intermediates.append(branch)
                    next_level.append(branch)
                level = next_level+(level[-1:] if len(level)%2 else [])
            group_outputs.extend(level)
        output = live_sources(group_outputs, "ShapeProgramCSGOutput")
        for node, operand in operations:
            if node.operation not in {"subtract", "difference", "intersect", "intersection"}:
                continue
            modifier = output.modifiers.new("Editable_"+node.node_id, "BOOLEAN")
            modifier.solver = "EXACT"
            modifier.operation = "DIFFERENCE" if node.operation in {"subtract", "difference"} else "INTERSECT"
            modifier.object = operand
        for obj in intermediates:
            obj["blendslop_export_exclude"] = True
            obj.hide_render = True
            obj.hide_set(True)
        objects.extend(intermediates)
        for node,obj in operations:
            obj["blendslop_export_exclude"] = True
            obj.hide_render = True
            obj.hide_set(True)
        objects.append(output)
        if csg_options.get('native_union_solver', 'EXACT') != 'EXACT' or csg_options.get('native_sdf_fallback', False):
            # Live modifiers keep Exact because artist edits invalidate numeric
            # qualification. The admitted numeric experiment is the rendered output.
            output['blendslop_export_exclude'] = True
            output.hide_render = True
            output.hide_set(True)
            baked = NativeOwnedGeometry(data, 'QualifiedProgramOutput').attach()
            _link_to_collection(baked, collection)
            baked.use_fake_user = False
            baked['blendslop_output_contract'] = 'baked_candidate_with_live_exact_artist_sources'
            objects.append(baked)
            output_contract = 'baked_candidate_with_live_exact_artist_sources; recompile after edits'
    residual_markers = []
    for patch in program.residual_patches:
        marker = _compile_residual_marker(patch, program=program)
        _link_to_collection(marker, collection)
        residual_markers.append(marker)

    root_object = _make_root_empty(program, objects, residual_markers, collection)
    final_qualification = None
    if union_report is not None:
        from blender_blocking.reconstruction.native_geometry import evaluated_arrays
        from blender_blocking.reconstruction.output_qualification import qualify_retained_output
        # The live Exact tree may differ from a numeric MANIFOLD/SDF proposal.
        # Qualify the actually retained tree/baked output after every signed edit.
        final_qualification = qualify_retained_output(evaluated_arrays(root_object), csg_options)
        if final_qualification.get("boundary_qualified"):
            output_contract += "; content-scoped boundary qualification passed"
        else:
            output_contract += "; final boundary qualification not established"
            warnings.append("final CSG output is an unchecked candidate; operand checks do not qualify it")
    return CompiledShapeProgram(
        program_id=program.program_id,
        root_object=root_object,
        objects=tuple(objects),
        residual_markers=tuple(residual_markers),
        warnings=tuple(warnings),
        union_report=union_report,
        output_contract=output_contract,
        final_qualification=final_qualification,
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
    if primitive == "capsule":
        from .capsule import CapsulePrimitive
        data = CapsulePrimitive.from_program_parameters(params, world=False).to_mesh_data(max(8, lathe_segments))
        mesh = bpy.data.meshes.new(name + "Mesh")
        mesh.from_pydata(data.vertices.tolist(), [], data.faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        obj.location = _location(params)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    elif primitive == "rounded_triangle":
        from .rounded_triangle import RoundedTrianglePrimitive
        part = RoundedTrianglePrimitive.from_program_parameters(params, world=False)
        data = part.to_mesh_data()
        mesh = bpy.data.meshes.new(name + "Mesh")
        mesh.from_pydata(data.vertices.tolist(), [], data.faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        obj.location = _location(params)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        obj["blendslop_triangle_field_semantics"] = "signed zero-set field; not Euclidean distance"
    elif primitive == 'generalized_sweep':
        from .generalized_sweep import GeneralizedSweepPrimitive
        part = GeneralizedSweepPrimitive.from_program_parameters(params,world=False)
        data = part.to_mesh_data(max(12,lathe_segments))
        mesh = bpy.data.meshes.new(name+'Mesh')
        mesh.from_pydata(data.vertices.tolist(),[],data.faces);mesh.update()
        obj = bpy.data.objects.new(name,mesh);obj.location = _location(params)
        obj['blendslop_sweep_section_count'] = len(part.section_knots_normalized)
        obj['blendslop_sweep_field_semantics'] = 'signed zero-set field; not Euclidean distance'
    elif primitive == 'deformed_superquadric':
        from .deformed_superquadric import DeformedSuperquadricPrimitive
        part = DeformedSuperquadricPrimitive.from_program_parameters(params, world=False)
        data = part.to_mesh_data(max(12, lathe_segments))
        mesh = bpy.data.meshes.new(name+'Mesh')
        mesh.from_pydata(data.vertices.tolist(), [], data.faces); mesh.update()
        obj = bpy.data.objects.new(name, mesh); obj.location = _location(params)
        obj['blendslop_deformation_contract'] = part.deformation_report()['deformation']
        obj['blendslop_deformed_field_semantics'] = 'inverse-mapped signed zero-set proxy; not Euclidean distance'
        obj['blendslop_continuous_jacobian_lower_bound'] = part.deformation_report()['jacobian_determinant_lower_bound']
    elif primitive == "polygon_extrusion":
        from .polygon_extrusion import PolygonExtrusionPrimitive
        outline = np.asarray(params["outer"],dtype=float)
        spans = np.ptp(outline,axis=0)
        scale = np.array([_float(params,"width_world",spans[0]),_float(params,"depth_world",spans[1])])/spans
        part = PolygonExtrusionPrimitive(outline,params.get("holes",()),
            height=_float(params,"height_world",.1),scale_xy=scale)
        data = part.to_mesh_data()
        mesh = bpy.data.meshes.new(name+"Mesh")
        mesh.from_pydata(data.vertices.tolist(),[],data.faces);mesh.update()
        obj = bpy.data.objects.new(name,mesh);obj.location = _location(params)
    elif primitive == "convex_hull":
        from blender_blocking.reconstruction.convex_proxy import native_convex_mesh
        from blender_blocking.reconstruction.native_geometry import NativeOwnedGeometry
        owner = NativeOwnedGeometry(native_convex_mesh(params["points_world"]), name)
        obj = owner.attach(); obj.use_fake_user = False
    elif primitive in {"box", "rounded_box"}:
        obj = _cube(name=name, params=params)
        if primitive == "rounded_box" and bevel_modifier:
            _add_bevel(obj, _float(params, "corner_radius_world", 0.02))
    elif primitive == "superquadric":
        from .analytic_primitives import SuperquadricPrimitive
        size = [_float(params, key, 1.)*.5 for key in ("width_world","depth_world","height_world")]
        part = SuperquadricPrimitive(radii=size, epsilon1=_float(params,"epsilon1",1.), epsilon2=_float(params,"epsilon2",1.))
        data = part.to_mesh_data(max(12, lathe_segments))
        mesh = bpy.data.meshes.new(name+"Mesh")
        mesh.from_pydata(data.vertices.tolist(), [], data.faces); mesh.update()
        obj = bpy.data.objects.new(name,mesh); obj.location = _location(params)
    elif primitive in {"cylinder", "frustum", "cone"}:
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
        raise ValueError(f"unsupported compiler primitive {primitive!r}")

    if weighted_normals and hasattr(obj, "modifiers") and primitive not in {"empty"}:
        _add_weighted_normals(obj)
    if any(key in params for key in ('rotation','rotation_row_major','rotation_euler')):
        from mathutils import Matrix
        from reconstruction.program_transforms import rotation_matrix
        obj.rotation_euler = Matrix(rotation_matrix(params).tolist()).to_euler()
    if "offset_x_normalized" in params:
        obj.location.x += float(params["offset_x_normalized"]) * _float(params,"assembly_width_world",1.)
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
    width = _float(params, "width_world", _float(params, "radius_x", .5)*2)
    depth = _float(params, "depth_world", _float(params, "radius_y", width*.5)*2)
    height = _float(params, "height_world", _float(params, "radius_z", width*.5)*2)
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
                    _float(row, "center_y_offset_world", 0.) + math.sin(theta) * radius_y,
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
        mesh_vertices.append((bottom_offset, _float(rows[0],"center_y_offset_world",0.), bottom_z))
        top_center = len(mesh_vertices)
        top_z = _float(rows[-1], "z_world", 0.0)
        top_offset = _float(rows[-1], "center_offset_world", 0.0)
        mesh_vertices.append((top_offset, _float(rows[-1],"center_y_offset_world",0.), top_z))
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
    if patch.suggested_node is not None:
        marker["blendslop_residual_suggested_node_id"] = patch.suggested_node.node_id
        marker["blendslop_residual_suggested_operation"] = patch.suggested_node.operation
        marker["blendslop_residual_suggested_primitive"] = (
            patch.suggested_node.primitive_type or ""
        )
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
