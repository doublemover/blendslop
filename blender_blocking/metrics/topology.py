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


@dataclass(frozen=True)
class TopologyRepairResult:
    """Result of a conservative pure-Python mesh topology repair pass."""

    vertices: np.ndarray
    faces: tuple[Face, ...]
    before: TopologyReport
    after: TopologyReport
    operations: tuple[Mapping[str, object], ...]
    changed: bool

    @property
    def improved(self) -> bool:
        return self.after.topology_score >= self.before.topology_score and (
            self.after.topology_score > self.before.topology_score
            or self.after.penalty < self.before.penalty
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "improved": self.improved,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "operations": [dict(operation) for operation in self.operations],
            "vertex_count": int(len(self.vertices)),
            "face_count": int(len(self.faces)),
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


def safe_topology_repair(
    vertices: np.ndarray | Sequence[Sequence[float]],
    faces: Iterable[Sequence[int]],
    *,
    keep_largest_component: bool = True,
) -> TopologyRepairResult:
    """Perform conservative topology cleanup without inventing new surfaces.

    This pass is intentionally safe for reconstruction QA loops: it removes
    provably invalid elements, duplicate faces, loose vertices, and optionally
    disconnected components outside the largest face-connected component. It
    does not fill holes, remesh, smooth, or move vertices, so any quality change
    is traceable and reversible.
    """
    vertex_array = _as_vertex_array(vertices)
    original_faces = normalize_faces(faces)
    before = mesh_topology_report(vertex_array, original_faces)
    operations: list[Mapping[str, object]] = []

    clean_faces, dropped_invalid = _valid_unique_faces(original_faces, len(vertex_array))
    if dropped_invalid:
        operations.append(
            {
                "operation": "drop_invalid_or_degenerate_faces",
                "count": dropped_invalid,
            }
        )

    duplicate_count = len(original_faces) - dropped_invalid - len(clean_faces)
    if duplicate_count > 0:
        operations.append({"operation": "drop_duplicate_faces", "count": duplicate_count})

    if keep_largest_component and clean_faces:
        clean_faces, dropped_components = _keep_largest_face_component(clean_faces)
        if dropped_components:
            operations.append(
                {
                    "operation": "drop_non_largest_components",
                    "count": dropped_components,
                }
            )

    compact_vertices, compact_faces, removed_vertices = _compact_vertices(
        vertex_array,
        clean_faces,
    )
    if removed_vertices:
        operations.append({"operation": "drop_loose_vertices", "count": removed_vertices})

    after = mesh_topology_report(compact_vertices, compact_faces)
    changed = bool(operations) or len(compact_vertices) != len(vertex_array)
    return TopologyRepairResult(
        vertices=compact_vertices,
        faces=compact_faces,
        before=before,
        after=after,
        operations=tuple(operations),
        changed=changed,
    )


def topology_repair_plan(report: TopologyReport | Mapping[str, object]) -> dict[str, object]:
    """Return a deterministic plan for the safest next topology repair steps."""
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    steps: list[dict[str, object]] = []
    if int(data.get("degenerate_faces", 0) or 0) > 0:
        steps.append(
            {
                "operation": "drop_invalid_or_degenerate_faces",
                "reason": "degenerate faces cannot contribute stable editable topology",
                "risk": "low",
            }
        )
    if int(data.get("loose_vertices", 0) or 0) > 0:
        steps.append(
            {
                "operation": "drop_loose_vertices",
                "reason": "loose vertices are not referenced by any surface face",
                "risk": "low",
            }
        )
    if int(data.get("connected_components", 0) or 0) > 1:
        steps.append(
            {
                "operation": "drop_or_label_small_components",
                "reason": "extra components may be floating artifacts or separate parts",
                "risk": "medium",
            }
        )
    if int(data.get("non_manifold_edges", 0) or 0) > 0:
        steps.append(
            {
                "operation": "split_or_remove_non_manifold_faces",
                "reason": "non-manifold edges block reliable boolean/edit operations",
                "risk": "medium",
            }
        )
    if int(data.get("boundary_edges", 0) or 0) > 0:
        steps.append(
            {
                "operation": "hole_fill_or_remesh_required",
                "reason": "boundary edges require surface synthesis, not safe deletion only",
                "risk": "high",
            }
        )
    return {
        "status": "clean" if not steps else "repair_recommended",
        "safe_automatic": all(step["risk"] == "low" for step in steps),
        "steps": steps,
    }


def _as_vertex_array(vertices: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    vertex_array = np.asarray(vertices, dtype=float)
    if vertex_array.size == 0:
        return np.empty((0, 3), dtype=float)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    return vertex_array


def _valid_unique_faces(
    faces: Sequence[Face],
    vertex_count: int,
) -> tuple[tuple[Face, ...], int]:
    clean: list[Face] = []
    seen: set[tuple[int, ...]] = set()
    dropped = 0
    for face in faces:
        if len(face) < 3 or len(set(face)) < 3:
            dropped += 1
            continue
        if any(vertex < 0 or vertex >= vertex_count for vertex in face):
            dropped += 1
            continue
        key = tuple(sorted(face))
        if key in seen:
            continue
        seen.add(key)
        clean.append(face)
    return tuple(clean), dropped


def _keep_largest_face_component(faces: Sequence[Face]) -> tuple[tuple[Face, ...], int]:
    vertex_to_faces: dict[int, set[int]] = defaultdict(set)
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].add(face_index)

    remaining = set(range(len(faces)))
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            face_index = queue.popleft()
            for vertex in faces[face_index]:
                for neighbor in vertex_to_faces[vertex]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
        components.append(component)

    if len(components) <= 1:
        return tuple(faces), 0
    largest = max(components, key=len)
    kept = tuple(face for index, face in enumerate(faces) if index in largest)
    return kept, len(faces) - len(kept)


def _compact_vertices(
    vertices: np.ndarray,
    faces: Sequence[Face],
) -> tuple[np.ndarray, tuple[Face, ...], int]:
    used = sorted({vertex for face in faces for vertex in face})
    if not used:
        return np.empty((0, 3), dtype=float), (), int(len(vertices))
    remap = {old: new for new, old in enumerate(used)}
    compact_faces = tuple(tuple(remap[vertex] for vertex in face) for face in faces)
    compact_vertices = np.asarray(vertices, dtype=float)[used]
    removed = int(len(vertices) - len(compact_vertices))
    return compact_vertices, compact_faces, removed


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
