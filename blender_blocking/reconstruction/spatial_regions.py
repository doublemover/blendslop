"""Deterministic neighborhood partitions, local PCA and adjacency merging."""
from __future__ import annotations
import heapq
import numpy as np
from scipy.spatial import cKDTree


def connected_point_regions(points, *, minimum=8, max_regions=8, radius=None):
    points = np.asarray(points, float)
    if len(points) < minimum:
        return []
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(7, len(points)))
    positive = distances[:, 1:][distances[:, 1:] > 1e-12]
    radius = float(radius if radius is not None else (np.median(positive)*2.5 if len(positive) else 1e-6))
    remaining = set(range(len(points))); groups = []
    while remaining:
        seed = min(remaining); remaining.remove(seed); queue = [seed]; group = []
        while queue:
            i = queue.pop(); group.append(i)
            for j in tree.query_ball_point(points[i], radius):
                if j in remaining:
                    remaining.remove(j); queue.append(j)
        if len(group) >= minimum:
            groups.append(points[np.array(sorted(group))])
    return sorted(groups, key=lambda p: (-len(p), tuple(p.mean(0))))[:max_regions]


def pca_box(points):
    points = np.asarray(points, float)
    center = points.mean(0)
    _, basis = np.linalg.eigh(np.cov(points-center, rowvar=False)+np.eye(3)*1e-12)
    basis = basis[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    local = (points-center)@basis
    lo, hi = local.min(0), local.max(0)
    radii = np.maximum((hi-lo)/2., np.ptp(points, axis=0).max()*1e-4)
    return center+((lo+hi)/2.)@basis.T, radii, basis


def cuboid_levels(points, *, fine_parts=24, levels=(12, 6, 3)):
    """Split locally, then merge only neighboring leaves using a lazy priority queue."""
    points = np.asarray(points, float)
    if len(points) < 8:
        return []
    groups = {0: np.arange(len(points))}
    serial = 1
    while len(groups) < fine_parts:
        key = max(groups, key=lambda k: (len(groups[k]), -k))
        ids = groups[key]
        if len(ids) < 16:
            break
        center, _, basis = pca_box(points[ids])
        order = ids[np.argsort((points[ids]-center)@basis[:, 0], kind='stable')]
        del groups[key]
        groups[serial], groups[serial+1] = order[:len(order)//2], order[len(order)//2:]
        serial += 2
    def volume(ids):
        return float(np.prod(pca_box(points[ids])[1]*2))
    # Point-neighborhood adjacency is retained through each merge.
    tree = cKDTree(points)
    _, nearest = tree.query(points, k=min(7, len(points)))
    adjacency = set()
    owners = np.empty(len(points), int)
    for key, ids in groups.items(): owners[ids] = key
    for i, row in enumerate(nearest):
        for j in row:
            a, b = sorted((int(owners[i]), int(owners[j])))
            if a != b: adjacency.add((a, b))
    heap = []
    def enqueue(a, b):
        if a in groups and b in groups:
            merged = np.concatenate([groups[a], groups[b]])
            heapq.heappush(heap, (volume(merged)-volume(groups[a])-volume(groups[b]), a, b))
    for a, b in sorted(adjacency): enqueue(a, b)
    snapshots = [tuple(points[ids] for _, ids in sorted(groups.items()))]
    wanted = set(int(n) for n in levels)
    while heap and len(groups) > min(wanted, default=1):
        _, a, b = heapq.heappop(heap)
        if a not in groups or b not in groups: continue
        neighbors = {v for edge in adjacency for v in edge if (a in edge or b in edge) and v not in {a, b}}
        merged = np.concatenate([groups.pop(a), groups.pop(b)])
        groups[serial] = merged
        adjacency = {edge for edge in adjacency if a not in edge and b not in edge}
        for other in sorted(neighbors):
            edge = tuple(sorted((serial, other))); adjacency.add(edge); enqueue(*edge)
        serial += 1
        if len(groups) in wanted:
            snapshots.append(tuple(points[ids] for _, ids in sorted(groups.items())))
    return snapshots
