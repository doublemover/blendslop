"""Read-only Boolean hierarchy with extraction confined to boundary tiles."""
from itertools import product
import time

import numpy as np

from .contracts import Chunk, ChunkKey, MeshExtractionResult, VolumeStats, VoxelTransform
from .grid import chunk_slices


class HierarchicalOccupancyGrid:
    """Accepted index boxes from deterministic, midpoint octree partitions.

    The boxes are disjoint occupied leaves. Empty leaves are implicit. Regular
    chunks are generated on demand for the existing VolumeGrid interchange.
    """

    backend = "hierarchical_occupancy"
    value_type = "occupancy_bool"
    default_value = False
    dtype = np.dtype(bool)

    def __init__(self, shape, bounds, accepted_boxes, *, transform=None, chunk_size=16):
        raw_shape = np.asarray(shape)
        if (raw_shape.shape != (3,) or raw_shape.dtype.kind not in "iuf"
                or not np.all(np.isfinite(raw_shape))
                or np.any(raw_shape != np.floor(raw_shape))):
            raise ValueError("hierarchy shape must contain three finite integers")
        self.shape = tuple(map(int, shape))
        self.bounds = bounds
        self.transform = transform or VoxelTransform.from_bounds_shape(bounds, self.shape)
        if not isinstance(chunk_size, (int, np.integer)):
            raise ValueError("hierarchy chunk size must be an integer")
        self.chunk_size = int(chunk_size)
        if (min(self.shape) <= 0 or self.chunk_size <= 0
                or tuple(self.transform.shape) != self.shape):
            raise ValueError("hierarchy shape/transform/chunk size are incompatible")
        raw_boxes = np.asarray(accepted_boxes)
        if raw_boxes.size == 0:
            raw_boxes = np.empty((0, 6), np.int64)
        if (raw_boxes.ndim != 2 or raw_boxes.shape[1] != 6
                or raw_boxes.dtype.kind not in "iuf"
                or not np.all(np.isfinite(raw_boxes))
                or np.any(raw_boxes != np.floor(raw_boxes))):
            raise ValueError("hierarchy boxes must be Nx6 finite integer intervals")
        self.accepted_boxes = raw_boxes.astype(np.int64).copy()
        if (np.any(self.accepted_boxes[:, :3] < 0)
                or np.any(self.accepted_boxes[:, 3:] > self.shape)
                or np.any(self.accepted_boxes[:, 3:] <= self.accepted_boxes[:, :3])):
            raise ValueError("hierarchy boxes must be positive intervals inside the grid")
        self.accepted_boxes.setflags(write=False)
        self.root = self._build_tree(
            np.zeros(3, int), np.array(self.shape), np.arange(len(self.accepted_boxes)))
        self.adaptive_report = {
            "storage": "accepted hierarchy boxes", "stored_boxes": len(self.accepted_boxes)}

    def _build_tree(self, start, end, ids):
        if not len(ids):
            return start, end, False, ()
        if (len(ids) == 1
                and np.array_equal(self.accepted_boxes[ids[0]], np.r_[start, end])):
            return start, end, True, ()
        intervals = [([(a, b)] if b-a == 1 else [(a, (a+b)//2), ((a+b)//2, b)])
                     for a, b in zip(start, end)]
        if all(len(interval) == 1 for interval in intervals):
            raise ValueError("duplicate or nonhierarchical occupancy boxes")
        children = []
        assigned = 0
        for axes in product(*intervals):
            lo = np.array([row[0] for row in axes])
            hi = np.array([row[1] for row in axes])
            keep = (np.all(self.accepted_boxes[ids, :3] >= lo, axis=1)
                    & np.all(self.accepted_boxes[ids, 3:] <= hi, axis=1))
            assigned += int(keep.sum())
            children.append(self._build_tree(lo, hi, ids[keep]))
        if assigned != len(ids):
            raise ValueError("occupancy boxes cross a hierarchy partition or overlap")
        return start, end, None, tuple(children)

    def sample_indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("hierarchy index queries must be Nx3")
        result = np.zeros(len(indices), bool)
        stack = [(self.root, np.arange(len(indices)))]
        while stack:
            node, ids = stack.pop()
            lo, hi, value, children = node
            ids = ids[np.all(indices[ids] >= lo, axis=1)
                      & np.all(indices[ids] < hi, axis=1)]
            if not len(ids):
                continue
            if value is not None:
                result[ids] = value
            else:
                stack.extend((child, ids) for child in children)
        return result

    def sample_world(self, points):
        indices = self.transform.world_to_nearest_index(
            np.asarray(points, float).reshape(-1, 3))
        return self.sample_indices(indices)

    def active_voxel_count(self):
        extents = self.accepted_boxes[:, 3:] - self.accepted_boxes[:, :3]
        return int(np.prod(extents, axis=1).sum())

    def _active_keys(self, size):
        keys = set()
        for box in self.accepted_boxes:
            ranges = [range(int(box[a]//size), int((box[a+3]-1)//size)+1)
                      for a in range(3)]
            keys.update(product(*ranges))
        return sorted(keys)

    def get_chunk(self, key):
        _, origin, valid_shape = chunk_slices(key, self.shape, self.chunk_size)
        output = np.zeros((self.chunk_size,)*3, bool)
        if min(valid_shape) <= 0:
            return output
        axes = [np.arange(origin[a], origin[a]+valid_shape[a]) for a in range(3)]
        indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        output[tuple(slice(0, n) for n in valid_shape)] = self.sample_indices(
            indices.reshape(-1, 3)).reshape(valid_shape)
        return output

    def iter_active_chunks(self):
        for indices in self._active_keys(self.chunk_size):
            key = ChunkKey(*indices)
            _, origin, valid_shape = chunk_slices(key, self.shape, self.chunk_size)
            yield Chunk(key, self.get_chunk(key), origin, valid_shape)

    def stats(self):
        # No regular chunks are stored; occupied chunks are generated on demand.
        # The actual stored-box count lives in adaptive_report.
        count = len(self._active_keys(self.chunk_size))
        return VolumeStats(int(np.prod(self.shape)), self.active_voxel_count(),
                           count, 0, self.shape, self.chunk_size, "bool",
                           self.value_type, self.default_value)

    def to_dense(self, max_voxels=None):
        if max_voxels is not None and np.prod(self.shape) > max_voxels:
            raise ValueError("hierarchy dense expansion exceeds voxel allowance")
        data = np.zeros(self.shape, bool)
        for box in self.accepted_boxes:
            data[tuple(slice(a, b) for a, b in zip(box[:3], box[3:]))] = True
        return data

    def boundary_tile_keys(self, size=16):
        """Only tiles touching an accepted-block face can contain a mixed cell."""
        keys = set()
        maximum = np.array(self.shape)
        for box in self.accepted_boxes:
            for axis in range(3):
                for face in (box[axis]-1, box[axis+3]-1):
                    ranges = []
                    for other in range(3):
                        if other == axis:
                            lo = hi = int(face)
                        else:
                            lo, hi = int(box[other]-1), int(box[other+3]-1)
                        lo = max(-1, lo)
                        hi = min(int(maximum[other]-1), hi)
                        ranges.append(range((lo+1)//size, (hi+1)//size+1))
                    keys.update(product(*ranges))
        return sorted(keys)

    def _tile_indices(self, key):
        start = np.array(key)*self.chunk_size-1
        end = np.minimum(start+self.chunk_size, self.shape)
        axes = [np.arange(start[a], end[a]+1) for a in range(3)]
        return start, np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def surface_indices(self, *, max_voxels=None, maximum_tiles=65536):
        # Preserve the public logical-voxel allowance. Store occupied boundary
        # nodes only, never the global occupancy array.
        if max_voxels is not None and np.prod(self.shape) > max_voxels:
            raise ValueError("hierarchy surface query exceeds voxel allowance")
        keys = self.boundary_tile_keys(self.chunk_size)
        if len(keys) > maximum_tiles:
            raise ValueError("hierarchy surface-tile allowance exceeded")
        result = set()
        started = time.perf_counter()
        offsets = np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]
        for count, key in enumerate(keys):
            if count % 64 == 0:
                print(f"hierarchy surface tiles={count}/{len(keys)} points={len(result)} "
                      f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
            _, indices = self._tile_indices(key)
            indices = indices.reshape(-1, 3)
            indices = indices[self.sample_indices(indices)]
            if not len(indices):
                continue
            enclosed = np.ones(len(indices), bool)
            for offset in offsets:
                enclosed &= self.sample_indices(indices+offset)
            result.update(map(tuple, indices[~enclosed]))
        return np.asarray(sorted(result), np.int64).reshape(-1, 3)

    def extract_selective_mesh(self, *, method="marching_cubes", skimage_method="lewiner",
                               maximum_tiles=65536, recorder=None):
        from skimage.measure import marching_cubes
        from .meshing import _mesh_topology_summary
        from blender_blocking.metrics.topology_receipt import ConnectivityTopologyReceipt

        keys = self.boundary_tile_keys(self.chunk_size)
        if len(keys) > maximum_tiles:
            return MeshExtractionResult.unavailable(
                method, "hierarchy boundary-tile allowance exceeded")
        started = time.perf_counter()
        vertices, faces, lookup = [], [], {}
        sampled = mixed = 0
        for count, key in enumerate(keys):
            if count % 64 == 0:
                print(f"hierarchy extraction tiles={count}/{len(keys)} sampled_nodes={sampled} "
                      f"elapsed={time.perf_counter()-started:.2f}s", flush=True)
            start, indices = self._tile_indices(key)
            data = self.sample_indices(indices.reshape(-1, 3)).reshape(indices.shape[:-1])
            sampled += data.size
            if not data.any() or data.all():
                continue
            mixed += 1
            local, triangles, _, _ = marching_cubes(
                data.astype(np.float32), level=.5, method=skimage_method,
                gradient_direction="ascent", allow_degenerate=False)
            ids = []
            for point in np.asarray(local, float)+start:
                # Same grid-edge intersection, exact coordinates only. No
                # approximate welding, tolerance changes, or point movement.
                identity = tuple(point)
                if identity not in lookup:
                    lookup[identity] = len(vertices)
                    vertices.append(point)
                ids.append(lookup[identity])
            faces.extend(np.asarray(ids)[triangles].tolist())
        points = self.transform.index_to_world(np.asarray(vertices, float).reshape(-1, 3))
        triangles = np.asarray(faces, np.int64).reshape(-1, 3)
        receipt = ConnectivityTopologyReceipt.capture(points, triangles) if len(triangles) else None
        topology = receipt.summary() if receipt is not None else _mesh_topology_summary(points, triangles)
        return MeshExtractionResult(
            "ok" if len(triangles) else "skipped", method, points, triangles,
            requested_method=method, topology=topology, topology_receipt=receipt, metrics={
                "storage": self.backend, "candidate_boundary_tiles": len(keys),
                "mixed_tiles": mixed, "sampled_nodes": sampled,
                "global_dense_expansion": False,
                "shared_vertex_rule": "exact identical global grid coordinates only",
                "boundary_qualification": False, "skimage_method": skimage_method,
                "elapsed_s": time.perf_counter()-started})
