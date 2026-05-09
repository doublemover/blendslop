"""Topology-aware mesh quality metrics.

The functions here are Blender-free and operate on plain vertex/face arrays.
Blender-side collectors can feed richer counts, but these helpers give pure
backends and tests the same scoring vocabulary.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


Face = Tuple[int, ...]


@dataclass(frozen=True)
class TopologyReport:
    """Portable topology diagnostics for a triangle/quad mesh."""

    vertex_count: int
    face_count: int
    edge_count: int
    boundary_edges: int
    non_manifold_edges: int
    degenerate_faces: int
    loose_vertices: int
    connected_components: int
    euler_characteristic: int
    watertight: bool

    @property
    def topology_score(self) -> float:
        score = 1.0
        score -= min(0.35, self.non_manifold_edges * 0.02)
        score -= min(0.25, self.boundary_edges * 0.01)
        score -= min(0.20, self.degenerate_faces * 0.03)
        score -= min(0.10, self.loose_vertices * 0.01)
        score -= min(0.20, max(0, self.connected_components - 1) * 0.05)
        return max(0.0, score)

    @property
    def required(self) -> bool:
        return True

    @property
    def passed(self) -> bool:
        return (
            self.watertight
            and self.non_manifold_edges == 0
            and self.degenerate_faces == 0
            and self.loose_vertices == 0
        )

    @property
    def reason(self) -> str:
        if self.passed:
            return ""
        reasons = []
        if self.boundary_edges:
            reasons.append(f"{self.boundary_edges} boundary edges")
        if self.non_manifold_edges:
            reasons.append(f"{self.non_manifold_edges} non-manifold edges")
        if self.degenerate_faces:
            reasons.append(f"{self.degenerate_faces} degenerate faces")
        if self.loose_vertices:
            reasons.append(f"{self.loose_vertices} loose vertices")
        if self.connected_components > 1:
            reasons.append(f"{self.connected_components} connected components")
        return "; ".join(reasons)

    @property
    def penalty(self) -> float:
        return max(0.0, 1.0 - self.topology_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "boundary_edges": self.boundary_edges,
            "non_manifold_edges": self.non_manifold_edges,
            "degenerate_faces": self.degenerate_faces,
            "loose_vertices": self.loose_vertices,
            "connected_components": self.connected_components,
            "euler_characteristic": self.euler_characteristic,
            "watertight": self.watertight,
            "topology_score": self.topology_score,
            "penalty": self.penalty,
            "required": self.required,
            "passed": self.passed,
            "pass": self.passed,
            "reason": self.reason,
        }


def normalize_faces(faces: Iterable[Sequence[int]]) -> tuple[Face, ...]:
    """Normalize faces to unique integer tuples, dropping repeated tail noise."""
    normalized = []
    for face in faces:
        clean = tuple(int(v) for v in face)
        normalized.append(clean)
    return tuple(normalized)


def face_edges(face: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return undirected canonical edges for one polygon face."""
    if len(face) < 2:
        return ()
    edges = []
    for idx, vertex in enumerate(face):
        other = face[(idx + 1) % len(face)]
        if vertex == other:
            continue
        edges.append((min(vertex, other), max(vertex, other)))
    return tuple(edges)


def mesh_topology_report(
    vertices: np.ndarray | Sequence[Sequence[float]],
    faces: Iterable[Sequence[int]],
) -> TopologyReport:
    """Compute Blender-free topology diagnostics from mesh arrays."""
    vertex_array = np.asarray(vertices, dtype=float)
    if vertex_array.ndim != 2 or (vertex_array.size and vertex_array.shape[1] != 3):
        raise ValueError("vertices must have shape (N, 3)")
    face_tuple = normalize_faces(faces)

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    used_vertices: set[int] = set()
    degenerate_faces = 0
    for face_index, face in enumerate(face_tuple):
        if len(face) < 3 or len(set(face)) < 3:
            degenerate_faces += 1
            continue
        if any(vertex < 0 or vertex >= len(vertex_array) for vertex in face):
            degenerate_faces += 1
            continue
        used_vertices.update(face)
        for edge in face_edges(face):
            edge_to_faces[edge].append(face_index)

    boundary_edges = sum(1 for owners in edge_to_faces.values() if len(owners) == 1)
    non_manifold_edges = sum(1 for owners in edge_to_faces.values() if len(owners) > 2)
    loose_vertices = len(vertex_array) - len(used_vertices)
    components = _connected_components(len(vertex_array), edge_to_faces.keys(), used_vertices)
    euler = len(vertex_array) - len(edge_to_faces) + len(face_tuple)
    watertight = boundary_edges == 0 and non_manifold_edges == 0 and degenerate_faces == 0
    return TopologyReport(
        vertex_count=int(len(vertex_array)),
        face_count=int(len(face_tuple)),
        edge_count=int(len(edge_to_faces)),
        boundary_edges=int(boundary_edges),
        non_manifold_edges=int(non_manifold_edges),
        degenerate_faces=int(degenerate_faces),
        loose_vertices=int(loose_vertices),
        connected_components=int(components),
        euler_characteristic=int(euler),
        watertight=bool(watertight),
    )


def topology_penalty(report: TopologyReport, weights: Mapping[str, float] | None = None) -> float:
    """Return a scalar penalty suitable for objective functions."""
    w = {
        "non_manifold_edges": 1.0,
        "boundary_edges": 0.25,
        "degenerate_faces": 1.0,
        "loose_vertices": 0.2,
        "connected_components": 0.5,
    }
    if weights:
        w.update({str(k): float(v) for k, v in weights.items()})
    return float(
        report.non_manifold_edges * w["non_manifold_edges"]
        + report.boundary_edges * w["boundary_edges"]
        + report.degenerate_faces * w["degenerate_faces"]
        + report.loose_vertices * w["loose_vertices"]
        + max(0, report.connected_components - 1) * w["connected_components"]
    )


def _connected_components(
    vertex_count: int,
    edges: Iterable[tuple[int, int]],
    used_vertices: set[int],
) -> int:
    if vertex_count == 0 or not used_vertices:
        return 0
    graph: dict[int, set[int]] = {vertex: set() for vertex in used_vertices}
    for a, b in edges:
        if a in graph and b in graph:
            graph[a].add(b)
            graph[b].add(a)
    remaining = set(graph)
    components = 0
    while remaining:
        components += 1
        start = remaining.pop()
        queue: deque[int] = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in graph[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components
