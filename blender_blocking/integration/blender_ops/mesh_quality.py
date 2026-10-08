"""Mesh topology and bounds quality reporting for Blender mesh objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned world-space 3D bounds."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @classmethod
    def empty(cls) -> "Bounds3D":
        """Return a zero-sized bounds record for empty meshes."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def size(self) -> Tuple[float, float, float]:
        """Return the extent along each axis."""
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the center point."""
        return (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "size": list(self.size),
            "center": list(self.center),
        }


@dataclass(frozen=True)
class MeshQualityReport:
    """Summary of mesh topology, degeneracy, and world-space bounds."""

    object_name: str
    vertices: int
    edges: int
    faces: int
    loose_vertices: int
    non_manifold_edges: int
    boundary_edges: int
    self_intersection_warnings: Tuple[str, ...]
    bbox: Bounds3D
    connected_components: int = 0
    internal_disconnected_shells: int = 0
    watertight: bool = False
    face_normal_consistent: bool = True
    zero_area_faces: int = 0
    duplicate_vertices: int = 0
    degenerate_faces: int = 0
    bbox_sane: bool = True
    volume: Optional[float] = None
    surface_area: Optional[float] = None
    genus_estimate: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "object_name": self.object_name,
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "loose_vertices": self.loose_vertices,
            "non_manifold_edges": self.non_manifold_edges,
            "boundary_edges": self.boundary_edges,
            "self_intersection_warnings": list(self.self_intersection_warnings),
            "bbox": self.bbox.to_dict(),
            "connected_components": self.connected_components,
            "internal_disconnected_shells": self.internal_disconnected_shells,
            "watertight": self.watertight,
            "face_normal_consistent": self.face_normal_consistent,
            "zero_area_faces": self.zero_area_faces,
            "duplicate_vertices": self.duplicate_vertices,
            "degenerate_faces": self.degenerate_faces,
            "bbox_sane": self.bbox_sane,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "genus_estimate": self.genus_estimate,
        }


def _vector_to_tuple(value: Any) -> Tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _world_vertex_tuple(obj: Any, vertex: Any) -> Tuple[float, float, float]:
    co = getattr(vertex, "co", vertex)
    matrix_world = getattr(obj, "matrix_world", None)
    if matrix_world is not None:
        try:
            return _vector_to_tuple(matrix_world @ co)
        except Exception:
            pass
    return _vector_to_tuple(co)


def _compute_bbox(coords: Sequence[Tuple[float, float, float]]) -> Bounds3D:
    if not coords:
        return Bounds3D.empty()
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    return Bounds3D(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _count_duplicate_vertices(
    coords: Sequence[Tuple[float, float, float]], threshold: float
) -> int:
    if not coords:
        return 0
    threshold = max(float(threshold), 1e-12)
    seen = set()
    duplicates = 0
    for coord in coords:
        key = tuple(round(component / threshold) for component in coord)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _connected_components(
    vertex_count: int, edges: Sequence[Tuple[int, int]]
) -> int:
    if vertex_count == 0:
        return 0

    adjacency: List[List[int]] = [[] for _ in range(vertex_count)]
    for a, b in edges:
        if 0 <= a < vertex_count and 0 <= b < vertex_count:
            adjacency[a].append(b)
            adjacency[b].append(a)

    seen = [False] * vertex_count
    components = 0
    for start in range(vertex_count):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return components


def _mesh_edges_as_indices(mesh: Any) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for edge in getattr(mesh, "edges", []):
        vertices = getattr(edge, "vertices", None)
        if vertices is None:
            continue
        edges.append((int(vertices[0]), int(vertices[1])))
    return edges


def _basic_report(
    obj: Any,
    *,
    warnings: Tuple[str, ...] = (),
    duplicate_threshold: float = 1e-6,
) -> MeshQualityReport:
    mesh = getattr(obj, "data", None)
    vertices = list(getattr(mesh, "vertices", [])) if mesh is not None else []
    edges = list(getattr(mesh, "edges", [])) if mesh is not None else []
    polygons = list(getattr(mesh, "polygons", [])) if mesh is not None else []
    coords = [_world_vertex_tuple(obj, vertex) for vertex in vertices]
    edge_indices = _mesh_edges_as_indices(mesh) if mesh is not None else []
    components = _connected_components(len(vertices), edge_indices)
    duplicate_vertices = _count_duplicate_vertices(coords, duplicate_threshold)

    return MeshQualityReport(
        object_name=str(getattr(obj, "name", "")),
        vertices=len(vertices),
        edges=len(edges),
        faces=len(polygons),
        loose_vertices=0,
        non_manifold_edges=0,
        boundary_edges=0,
        self_intersection_warnings=warnings,
        bbox=_compute_bbox(coords),
        connected_components=components,
        internal_disconnected_shells=max(0, components - 1),
        watertight=False,
        zero_area_faces=0,
        duplicate_vertices=duplicate_vertices,
        degenerate_faces=0,
        bbox_sane=bool(vertices),
        surface_area=None,
        volume=None,
        genus_estimate=None,
    )


def collect_mesh_quality(
    obj: Any,
    *,
    duplicate_threshold: float = 1e-6,
    zero_area_threshold: float = 1e-12,
) -> MeshQualityReport:
    """Collect topology and bounds metrics for a Blender mesh object.

    Blender and bmesh are imported lazily so this module remains importable in
    pure-Python test environments.
    """
    if obj is None:
        return MeshQualityReport(
            object_name="",
            vertices=0,
            edges=0,
            faces=0,
            loose_vertices=0,
            non_manifold_edges=0,
            boundary_edges=0,
            self_intersection_warnings=("object_missing",),
            bbox=Bounds3D.empty(),
            bbox_sane=False,
        )

    mesh = getattr(obj, "data", None)
    if mesh is None:
        return MeshQualityReport(
            object_name=str(getattr(obj, "name", "")),
            vertices=0,
            edges=0,
            faces=0,
            loose_vertices=0,
            non_manifold_edges=0,
            boundary_edges=0,
            self_intersection_warnings=("mesh_data_missing",),
            bbox=Bounds3D.empty(),
            bbox_sane=False,
        )

    try:
        import bmesh  # type: ignore
    except ImportError:
        return _basic_report(
            obj,
            warnings=("bmesh_unavailable_topology_metrics_partial",),
            duplicate_threshold=duplicate_threshold,
        )

    vertices = list(getattr(mesh, "vertices", []))
    coords = [_world_vertex_tuple(obj, vertex) for vertex in vertices]
    bbox = _compute_bbox(coords)

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()

        loose_vertices = sum(1 for vert in bm.verts if not vert.link_edges)
        boundary_edges = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
        non_manifold_edges = sum(1 for edge in bm.edges if not edge.is_manifold)
        zero_area_faces = sum(
            1 for face in bm.faces if face.calc_area() <= zero_area_threshold
        )
        degenerate_faces = zero_area_faces + sum(
            1
            for face in bm.faces
            if len({int(vert.index) for vert in face.verts}) < 3
        )
        surface_area = sum(face.calc_area() for face in bm.faces)

        edge_indices = []
        for edge in bm.edges:
            edge_indices.append((int(edge.verts[0].index), int(edge.verts[1].index)))
        components = _connected_components(len(bm.verts), edge_indices)
        watertight = (
            len(bm.verts) > 0
            and len(bm.faces) > 0
            and loose_vertices == 0
            and boundary_edges == 0
            and non_manifold_edges == 0
            and zero_area_faces == 0
        )

        volume = None
        try:
            volume = float(abs(bm.calc_volume(signed=True)))
        except Exception:
            volume = None

        genus_estimate = None
        if watertight and components > 0:
            chi = len(bm.verts) - len(bm.edges) + len(bm.faces)
            genus_estimate = max(0, int(round((2 * components - chi) / 2.0)))

        return MeshQualityReport(
            object_name=str(getattr(obj, "name", "")),
            vertices=len(bm.verts),
            edges=len(bm.edges),
            faces=len(bm.faces),
            loose_vertices=loose_vertices,
            non_manifold_edges=non_manifold_edges,
            boundary_edges=boundary_edges,
            self_intersection_warnings=(),
            bbox=bbox,
            connected_components=components,
            internal_disconnected_shells=max(0, components - 1),
            watertight=watertight,
            face_normal_consistent=True,
            zero_area_faces=zero_area_faces,
            duplicate_vertices=_count_duplicate_vertices(coords, duplicate_threshold),
            degenerate_faces=degenerate_faces,
            bbox_sane=bool(vertices) and all(size >= 0.0 for size in bbox.size),
            volume=volume,
            surface_area=surface_area,
            genus_estimate=genus_estimate,
        )
    finally:
        bm.free()
