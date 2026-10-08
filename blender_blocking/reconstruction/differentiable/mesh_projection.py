from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from primitives.primitive_protocol import MeshData
    from primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from primitives.superfrustum import SuperFrustum
    from reconstruction.artifacts import write_json
except ImportError:  # pragma: no cover - package import path.
    from ...primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ...primitives.primitive_protocol import MeshData
    from ...primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ...primitives.superfrustum import SuperFrustum
    from ...reconstruction.artifacts import write_json
from .contracts import CameraSpec, RenderableScene


def _target_cameras_and_masks(target: object) -> tuple[tuple[CameraSpec, ...], dict[str, np.ndarray]]:
    from reconstruction.projection_contract import pixel_cell_viewport

    cameras = []
    silhouettes = {}
    for constraint in getattr(target, "constraints", ()):
        mask = np.asarray(getattr(constraint.mask, "mask", constraint.mask), dtype=np.float32)
        if mask.ndim != 2:
            continue
        height, width = mask.shape
        axes, world_bounds = pixel_cell_viewport(target, constraint)
        cameras.append(
            CameraSpec(
                name=constraint.view,
                axes=axes,
                image_size=(int(width), int(height)),
                world_bounds=world_bounds,
            )
        )
        silhouettes[constraint.view] = mask
    return tuple(cameras), silhouettes

def _scene_mesh_arrays(scene: RenderableScene) -> tuple[np.ndarray, np.ndarray]:
    vertices_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0
    if scene.mesh is not None:
        vertices, faces = _mesh_arrays(scene.mesh)
        vertices_blocks.append(vertices)
        face_blocks.append(_triangulated_mesh_faces(faces))
        offset += len(vertices)
    for renderable in scene.primitives:
        if renderable.mesh_proxy is None:
            continue
        vertices, faces = _mesh_arrays(renderable.mesh_proxy)
        triangles = _triangulated_mesh_faces(faces)
        vertices_blocks.append(vertices)
        if len(triangles):
            face_blocks.append(triangles + offset)
        offset += len(vertices)
    if not vertices_blocks:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    vertices_all = np.vstack(vertices_blocks).astype(np.float32, copy=False)
    faces_all = (
        np.vstack(face_blocks).astype(np.int32, copy=False)
        if face_blocks
        else np.empty((0, 3), dtype=np.int32)
    )
    return vertices_all, faces_all

def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float32)
    faces = tuple(tuple(int(index) for index in face) for face in getattr(mesh, "faces", ()))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"mesh vertices must have shape (N, 3), got {vertices.shape}")
    return vertices, faces

def _triangulated_mesh_faces(faces: Sequence[Sequence[int]]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        first = int(face[0])
        for index in range(1, len(face) - 1):
            tri = (first, int(face[index]), int(face[index + 1]))
            if len(set(tri)) == 3:
                triangles.append(tri)
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)

def _project_vertices_to_clip(vertices: np.ndarray, camera: CameraSpec) -> np.ndarray:
    vertex_array = np.asarray(vertices, dtype=np.float32)
    axes = tuple(int(axis) for axis in camera.axes)
    if len(axes) != 2 or axes[0] == axes[1] or any(axis not in (0, 1, 2) for axis in axes):
        raise RuntimeError(f"invalid camera axes for nvdiffrast: {camera.axes!r}")
    depth_axis = next(axis for axis in (0, 1, 2) if axis not in axes)
    xmin, xmax, ymin, ymax = (float(value) for value in camera.world_bounds)
    xden = max(abs(xmax - xmin), 1.0e-12)
    yden = max(abs(ymax - ymin), 1.0e-12)
    x_clip = ((vertex_array[:, axes[0]] - xmin) / xden) * 2.0 - 1.0
    y_clip = ((vertex_array[:, axes[1]] - ymin) / yden) * 2.0 - 1.0
    depth_values = vertex_array[:, depth_axis]
    depth_min = float(np.min(depth_values)) if len(depth_values) else 0.0
    depth_max = float(np.max(depth_values)) if len(depth_values) else 1.0
    depth_den = max(abs(depth_max - depth_min), 1.0e-12)
    z_clip = ((depth_values - depth_min) / depth_den) * 2.0 - 1.0
    w_clip = np.ones_like(z_clip, dtype=np.float32)
    return np.column_stack((x_clip, y_clip, z_clip, w_clip)).astype(
        np.float32,
        copy=False,
    )
