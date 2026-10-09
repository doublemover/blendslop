"""Explicit editable shared-plane construction for three observed box leaves.

This is a new model proposal, never an automatic proximity snap or a shading
repair. Historical independent-box programs continue to use their old compiler.
"""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
import math
import numpy as np

from .multipart_family import retained_multipart_program

PROTOCOL = "multipart_shared_far_y_plane_v1"
_DIMENSIONS = ("width_world", "depth_world", "height_world")
_CENTER = ("x", "y", "z")


def _boxes(program, *, require_flat_style=True):
    boxes = {}
    for node in program.root_nodes:
        p = node.parameters
        values = [p.get(key) for key in (*_CENTER, *_DIMENSIONS)]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
               for value in values):
            raise ValueError("shared-plane model requires explicit numeric box centers and dimensions")
        center, size = np.asarray(values[:3], float), np.asarray(values[3:], float)
        if not np.isfinite(center).all() or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("shared-plane model requires finite positive box dimensions")
        rotation = np.asarray(p.get("rotation", np.eye(3)), float)
        if (rotation.shape != (3, 3) or not np.array_equal(rotation, np.eye(3))
                or any(key in p for key in ("rotation_euler", "rotation_row_major"))):
            raise ValueError("shared-plane model requires axis-aligned box leaves")
        if require_flat_style and p.get("weighted_normals", False):
            raise ValueError("shared-plane repair preserves flat authored shading")
        if node.node_id in boxes:
            raise ValueError("shared-plane model requires unique node identities")
        boxes[node.node_id] = (node, center - size / 2, center + size / 2)
    return boxes


def shared_far_plane_program(wire, *, base_node_id="observed_base", arm_node_id="observed_tall_arm"):
    """Declare one far-Y relation, preserving the arm near plane and other leaves.

    Calling this function is the explicit model choice. No tolerance, reference
    arrays, camera pixels, fitting or acceptance criterion enters this interface.
    """
    program = retained_multipart_program(wire)
    if "shared_far_plane" in program.metadata:
        raise ValueError("shared-plane proposal must start from an independent-box recipe")
    boxes = _boxes(program)
    if base_node_id == arm_node_id or base_node_id not in boxes or arm_node_id not in boxes:
        raise ValueError("shared-plane relation needs distinct existing base and arm nodes")
    base, base_lo, base_hi = boxes[base_node_id]
    arm, arm_lo, arm_hi = boxes[arm_node_id]
    if (not base_lo[1] <= arm_lo[1] < base_hi[1]
            or not base_lo[0] < arm_lo[0] < arm_hi[0] < base_hi[0]
            or not base_lo[2] < arm_lo[2] < base_hi[2] < arm_hi[2]):
        raise ValueError("shared-plane model needs an overlapping raised arm inside the base footprint")
    near, far = float(arm_lo[1]), float(base_hi[1])
    parameters = {**arm.parameters, "y": (near + far) / 2, "depth_world": far - near}
    declaration = {"protocol": PROTOCOL, "base_node_id": base.node_id, "arm_node_id": arm.node_id,
                   "axis": "y", "side": "far", "original_arm_far_world": float(arm_hi[1]),
                   "base_far_world": far, "preserved_arm_near_world": near,
                   "original_gap_world": far - float(arm_hi[1]),
                   "scope": "explicit shared-plane model; near fixed, other leaves and X/Z controls unchanged",
                   "native_construction": "arm Y embedded in base native Y frame; far local coordinate exactly 0.5"}
    roots = tuple(replace(node, parameters=parameters) if node.node_id == arm.node_id else node
                  for node in program.root_nodes)
    return replace(program, root_nodes=roots, metadata={**program.metadata, "shared_far_plane": declaration})


def _relation(program):
    if (len(program.root_nodes) != 3 or program.constraints or program.residual_patches
            or any(node.operation != "add" or node.primitive_type != "box" or node.children
                   for node in program.root_nodes)):
        raise ValueError("declared shared plane supports three unconstrained additive box leaves only")
    declaration = program.metadata.get("shared_far_plane", {})
    if not isinstance(declaration, Mapping):
        raise ValueError("shared-plane declaration must be a mapping")
    boxes = _boxes(program, require_flat_style=False)
    if (declaration.get("protocol") != PROTOCOL or declaration.get("axis") != "y"
            or declaration.get("side") != "far"):
        raise ValueError("compilation requires an explicit supported shared-plane declaration")
    base_id, arm_id = declaration.get("base_node_id"), declaration.get("arm_node_id")
    if base_id == arm_id or base_id not in boxes or arm_id not in boxes:
        raise ValueError("shared-plane declaration does not bind distinct existing box leaves")
    _, base_lo, base_hi = boxes[base_id]
    _, arm_lo, arm_hi = boxes[arm_id]
    for key in ("base_far_world", "preserved_arm_near_world"):
        value = declaration.get(key)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))):
            raise ValueError("shared-plane declaration requires finite numeric " + key)
    # Exact recipe consistency; native construction is guarded separately.
    if (float(base_hi[1]) != declaration.get("base_far_world")
            or float(arm_lo[1]) != declaration.get("preserved_arm_near_world")
            or float(arm_hi[1]) != float(base_hi[1]) or base_lo[1] > arm_lo[1]):
        raise ValueError("shared-plane declaration differs from its exact recipe controls")
    # Blender source poses are binary32. Reject unrepresentable source/frame
    # controls before allocating any collection or primitive.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for node, _, _ in boxes.values():
            controls = np.asarray([node.parameters[key] for key in (*_CENTER, *_DIMENSIONS)], np.float32)
            if not np.isfinite(controls).all() or (controls[3:] <= 0).any():
                raise ValueError("shared-plane source pose/dimensions are not native-representable")
        base = boxes[base_id][0].parameters
        center, depth = float(np.float32(base["y"])), float(np.float32(base["depth_world"]))
        unit = np.asarray([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], np.float32)
        shared_y_coordinates(unit, near_world=float(arm_lo[1]), base_center_y=center, base_depth_world=depth)
    return base_id, arm_id, float(arm_lo[1])


def shared_y_coordinates(coordinates, *, near_world, base_center_y, base_depth_world):
    """Keep X/Z coordinates and use the base's exact native Y frame."""
    values = (near_world, base_center_y, base_depth_world)
    if any(isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) for value in values):
        raise ValueError("shared-plane frame requires finite numeric values")
    if base_depth_world <= 0:
        raise ValueError("shared-plane frame requires positive base depth")
    source = np.asarray(coordinates)
    if (source.shape != (8, 3) or not np.isfinite(source).all()
            or not np.array_equal(np.unique(source[:, 1]), [-.5, .5])):
        raise ValueError("shared-plane embedding needs an unchanged eight-vertex unit box")
    near_local = np.float32((near_world - base_center_y) / base_depth_world)
    if not np.isfinite(near_local) or near_local >= .5:
        raise ValueError("shared-plane arm must have positive native depth")
    result = np.asarray(source, dtype=np.float32).copy()
    result[:, 1] = np.where(source[:, 1] < 0, near_local, np.float32(.5))
    return result


def shared_depth_coordinates(coordinates, *, depth_world, frame_depth_world):
    """Change only arm depth toward its near side, keeping the shared far plane."""
    if (isinstance(depth_world, (bool, np.bool_)) or isinstance(frame_depth_world, (bool, np.bool_))
            or not math.isfinite(float(depth_world)) or not math.isfinite(float(frame_depth_world))
            or depth_world <= 0 or frame_depth_world <= 0):
        raise ValueError("shared-plane depth edit requires finite positive world dimensions")
    source = np.asarray(coordinates)
    if (source.shape != (8, 3) or not np.isfinite(source).all()
            or len(np.unique(source[:, 1])) != 2 or float(source[:, 1].max()) != .5):
        raise ValueError("depth edit requires the declared two-plane native arm")
    near_local = np.float32(.5 - depth_world / frame_depth_world)
    if not np.isfinite(near_local) or near_local >= .5:
        raise ValueError("depth edit cannot represent a positive native arm")
    result = np.asarray(source, dtype=np.float32).copy()
    result[source[:, 1] < .5, 1] = near_local
    return result



def _compile_live_union(program):
    """Compile three production box leaves with two retained live Exact unions."""
    if (len(program.root_nodes) != 3 or any(node.operation != "add" or node.primitive_type != "box"
            or node.children for node in program.root_nodes)):
        raise ValueError("shared-plane live union requires exactly three additive box leaves")
    import bpy
    from primitives.shape_program_compiler import compile_shape_program
    compiled = compile_shape_program(program, bevel_modifier=False, weighted_normals=False)
    by_node = {part.get("blendslop_shape_node_id"): part for part in compiled.objects}
    parts = [by_node[node.node_id] for node in program.root_nodes]
    obj = parts[0]
    for index, operand in enumerate(parts[1:]):
        modifier = obj.modifiers.new("ObservedExactUnion" + str(index + 1), "BOOLEAN")
        modifier.operation, modifier.solver, modifier.object = "UNION", "EXACT", operand
        operand["blendslop_export_exclude"] = True
        operand.hide_render = True
        operand.hide_set(True)
    bpy.context.view_layer.update()
    return obj, parts


def shared_plane_receipt(program, parts):
    """Observe the declared source relation without applying it a second time."""
    base_id, arm_id, near = _relation(program)
    by_id = {part.get("blendslop_shape_node_id"): part for part in parts}
    base, arm = by_id[base_id], by_id[arm_id]
    if (arm.get("blendslop_shared_far_plane_protocol") != PROTOCOL
            or arm.get("blendslop_shared_far_base_node") != base_id):
        raise ValueError("declared shared-plane compiler embedding is unavailable")
    vertices = np.empty(len(arm.data.vertices) * 3, dtype=np.float32)
    arm.data.vertices.foreach_get("co", vertices)
    coordinates = vertices.reshape(-1, 3)
    if float(coordinates[:, 1].max()) != .5:
        raise ValueError("declared shared-plane native far coordinate changed")
    actual_far = float(base.location.y) + float(base.scale.y) * .5
    actual_near = float(base.location.y) + float(base.scale.y) * float(coordinates[:, 1].min())
    receipt = {"protocol": PROTOCOL, "base_node_id": base_id, "arm_node_id": arm_id,
               "exact_common_y_frame": arm.location.y == base.location.y and arm.scale.y == base.scale.y,
               "far_local": .5, "shared_far_world": actual_far, "near_world": actual_near,
               "declared_near_world": near, "native_near_rounding_world": actual_near - near,
               "flat_shading_preserved": all(not p.use_smooth for part in parts for p in part.data.polygons),
               "edit_scope": "arm near-side depth from mesh Y coordinates; far plane fixed; other X/Z source controls remain live",
               "scope": "construction observation only; final shape, surface, boundary and edit gates remain independent"}
    if not receipt["exact_common_y_frame"] or not receipt["flat_shading_preserved"]:
        raise ValueError("shared-plane native construction changed its frame or flat style")
    return receipt


def embed_shared_plane_sources(program, parts):
    """Apply the declared relation exactly once before any union evaluation.

    This geometry-only hook preserves caller collection/style options. Existing
    historical metadata is unrestricted; constraints/residuals are unsupported
    only when this explicit three-box relation is declared.
    """
    base_id, arm_id, near = _relation(program)
    import bpy
    by_id = {part.get("blendslop_shape_node_id"): part for part in parts}
    if by_id[arm_id].get("blendslop_shared_far_plane_protocol") is not None:
        raise ValueError("declared shared-plane source embedding already applied")
    base, arm = by_id[base_id], by_id[arm_id]
    vertices = np.empty(len(arm.data.vertices) * 3, dtype=np.float32)
    arm.data.vertices.foreach_get("co", vertices)
    coordinates = shared_y_coordinates(vertices.reshape(-1, 3), near_world=near,
        base_center_y=float(base.location.y), base_depth_world=float(base.scale.y))
    arm.data.vertices.foreach_set("co", coordinates.ravel())
    arm.location.y, arm.scale.y = base.location.y, base.scale.y
    arm["blendslop_shared_far_base_node"] = base_id
    arm["blendslop_shared_far_plane_protocol"] = PROTOCOL
    arm.data.update()
    bpy.context.view_layer.update()
    return shared_plane_receipt(program, parts)


def compile_shared_plane_multipart(program):
    """Use generic declared embedding once, then retain two live Exact unions."""
    _relation(program)  # Fail before scene allocation.
    obj, parts = _compile_live_union(program)
    return obj, parts, shared_plane_receipt(program, parts)
