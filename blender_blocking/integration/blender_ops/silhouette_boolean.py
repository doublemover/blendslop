"""Shared helpers for silhouette boolean reconstruction."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def split_contours(
    contours: List[np.ndarray],
    hierarchy: Optional[np.ndarray],
) -> Tuple[List[int], Dict[int, List[int]]]:
    if not contours:
        return [], {}
    if hierarchy is None or len(hierarchy) == 0:
        outer = list(range(len(contours)))
        return outer, {}
    outer = [index for index, item in enumerate(hierarchy[0]) if item[3] == -1]
    holes: Dict[int, List[int]] = {index: [] for index in outer}
    for index, item in enumerate(hierarchy[0]):
        parent = item[3]
        if parent != -1 and parent in holes:
            holes[parent].append(index)
    if not outer:
        outer = list(range(len(contours)))
    return outer, holes


def mesh_counts(obj: object) -> Tuple[int, int]:
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0, 0
    return len(obj.data.vertices), len(obj.data.polygons)


def apply_transforms(
    obj: object,
    *,
    bpy_module: object,
    triangulate_object: Callable[[object], None],
    clean_mesh_for_boolean: Callable[[object], None],
) -> None:
    bpy_module.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy_module.context.view_layer.objects.active = obj
    bpy_module.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy_module.ops.object.mode_set(mode="EDIT")
    bpy_module.ops.mesh.normals_make_consistent(inside=False)
    bpy_module.ops.object.mode_set(mode="OBJECT")
    triangulate_object(obj)
    clean_mesh_for_boolean(obj)
    obj.select_set(False)


def apply_boolean(
    base: object,
    other: object,
    operation: str,
    solver: str,
    *,
    bpy_module: object,
) -> bool:
    if base is None or other is None:
        return False
    modifier = base.modifiers.new(name=f"{operation}_Op", type="BOOLEAN")
    modifier.operation = operation
    modifier.object = other
    modifier.solver = solver
    bpy_module.context.view_layer.objects.active = base
    bpy_module.ops.object.modifier_apply(modifier=modifier.name)
    bpy_module.data.objects.remove(other, do_unlink=True)
    verts, faces = mesh_counts(base)
    return verts > 0 and faces > 0
