"""Immutable geometry and explicitly owned Blender meshes; no bpy at import time."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import numpy as np

from blender_blocking.metrics.topology_receipt import (
    TOPOLOGY_EVALUATOR, connectivity_hash, topology_for_geometry,
)


@dataclass(frozen=True)
class GeometryArrays:
    vertices: np.ndarray
    faces: np.ndarray
    content_hash: str
    connectivity_hash: str

    @classmethod
    def capture(cls, vertices, faces):
        v = np.array(vertices, dtype=np.float64, order="C", copy=True)
        f = np.array(faces, dtype=np.int64, order="C", copy=True)
        if (v.ndim != 2 or v.shape[1] != 3 or not len(v) or
                f.ndim != 2 or f.shape[1] != 3 or not len(f)):
            raise ValueError("candidate requires vertices and triangle faces")
        if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
            raise ValueError("invalid evaluated candidate geometry")
        connectivity = connectivity_hash(len(v), f)
        content = hashlib.sha256(v.tobytes() + connectivity.encode()).hexdigest()
        v.flags.writeable = False; f.flags.writeable = False
        return cls(v, f, content, connectivity)

    def __reduce__(self):
        # NumPy pickle restores writable arrays. Rebuild through capture so
        # transported geometry is validated and its hashes match its buffers.
        return (type(self).capture, (self.vertices, self.faces))

    @property
    def nbytes(self):
        return self.vertices.nbytes + self.faces.nbytes


class GeometryCache:
    def __init__(self):
        self.topology = {}
        self.bvh = {}
        self.hits = 0
        self.misses = 0
        self.receipt_reuses = 0

    def topology_report(self, data, receipt=None):
        # The current index-only report ignores coordinates. A geometric
        # degeneracy evaluator must use content_hash and a new version.
        key = (TOPOLOGY_EVALUATOR, data.connectivity_hash)
        if key in self.topology:
            self.hits += 1
        else:
            report, reused = topology_for_geometry(data, receipt)
            self.topology[key] = report
            self.receipt_reuses += int(reused)
            self.misses += 1
        return dict(self.topology[key])

    def scalar_bvh(self, data):
        key = ("BVHTree_triangles_v1", data.content_hash)
        if key not in self.bvh:
            from mathutils.bvhtree import BVHTree
            self.bvh[key] = BVHTree.FromPolygons(data.vertices.tolist(), data.faces.tolist(), all_triangles=True)
        return self.bvh[key]


def evaluated_arrays(obj, recorder=None):
    """Copy bulk numeric buffers, then release every temporary evaluated mesh."""
    import bpy
    vertices, faces = [], []
    offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    from .output_targets import output_mesh_targets
    for part in output_mesh_targets([obj]):
        evaluated = part.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            v = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            f = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.vertices.foreach_get("co", v)
            mesh.loop_triangles.foreach_get("vertices", f)
            matrix = np.array(evaluated.matrix_world, dtype=float)
            world = v.reshape(-1, 3).astype(float) @ matrix[:3, :3].T + matrix[:3, 3]
            vertices.append(world); faces.append(f.reshape(-1, 3).astype(np.int64) + offset)
            offset += len(world)
            if recorder is not None:
                recorder.count("native_to_python_bytes", v.nbytes + f.nbytes)
        finally:
            evaluated.to_mesh_clear()
    if not vertices:
        raise ValueError("render target has no evaluated mesh geometry")
    return GeometryArrays.capture(np.concatenate(vertices), np.concatenate(faces))


class NativeOwnedGeometry:
    """Own one original mesh/object, detached while other candidates mutate scenes."""
    def __init__(self, data, name="CandidateEvidence", recorder=None):
        import bpy
        self.data = data
        mesh = bpy.data.meshes.new(name)
        self.obj = bpy.data.objects.new(name, mesh)
        self.obj.use_fake_user = True
        self.obj["blendslop_owned_candidate"] = True
        try:
            mesh.vertices.add(len(data.vertices))
            mesh.loops.add(data.faces.size)
            mesh.polygons.add(len(data.faces))
            coordinates = np.asarray(data.vertices, np.float32).ravel()
            indices = np.asarray(data.faces, np.int32).ravel()
            mesh.vertices.foreach_set("co", coordinates)
            mesh.loops.foreach_set("vertex_index", indices)
            mesh.polygons.foreach_set("loop_start", np.arange(len(data.faces), dtype=np.int32) * 3)
            mesh.polygons.foreach_set("loop_total", np.full(len(data.faces), 3, dtype=np.int32))
            mesh.update(calc_edges=True)
            if recorder is not None:
                recorder.count("native_mesh_builds")
                recorder.count("python_to_native_bytes", coordinates.nbytes + indices.nbytes + len(data.faces) * 8)
        except BaseException:
            self.release()
            raise

    def update(self, data):
        """Retain the target object; bulk coordinates or a changed-topology mesh."""
        if data.content_hash == self.data.content_hash:
            return 'unchanged'
        import bpy
        if data.connectivity_hash == self.data.connectivity_hash:
            self.obj.data.vertices.foreach_set('co', np.asarray(data.vertices, np.float32).ravel())
            self.obj.data.update()
            action = 'coordinates_updated'
        else:
            replacement = NativeOwnedGeometry(data, self.obj.name+'Topology')
            old_mesh = self.obj.data
            self.obj.data = replacement.obj.data
            replacement.release()
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
            action = 'topology_rebuilt'
        self.data = data
        self.obj.update_tag()
        return action

    def attach(self):
        import bpy
        if not self.obj.users_collection:
            bpy.context.collection.objects.link(self.obj)
        self.obj.hide_render = False
        return self.obj

    def detach(self):
        for collection in list(self.obj.users_collection):
            collection.objects.unlink(self.obj)

    @contextmanager
    def rendered(self):
        self.attach()
        try:
            yield self.obj
        finally:
            self.detach()

    def release(self):
        import bpy
        obj = getattr(self, "obj", None)
        if obj is None:
            return
        try:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        except ReferenceError:
            pass
        self.obj = None


def geometry_arrays(value):
    return value.data if isinstance(value, NativeOwnedGeometry) else value
