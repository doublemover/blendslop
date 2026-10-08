"""Loft mesh generation from elliptical profile slices using bmesh."""

from __future__ import annotations

import math
import json
from typing import List, Optional, Sequence, Tuple

from geometry.profile_models import EllipticalSlice
from geometry.loft_surface import prepare_loft_surface
from integration.blender_ops.mesh_quality import collect_mesh_quality

try:
    import bpy
    import bmesh

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def _is_degenerate_slice(
    slice_data: EllipticalSlice,
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> bool:
    if not weld_degenerate_rings:
        return False
    return float(slice_data.rx) <= min_radius_u or float(slice_data.ry) <= min_radius_u


def _collapse_degenerate_slice_run(run: Sequence[EllipticalSlice]) -> EllipticalSlice:
    count = float(len(run))
    cx = sum(float(item.cx) if item.cx is not None else 0.0 for item in run) / count
    cy = sum(float(item.cy) if item.cy is not None else 0.0 for item in run) / count
    z = sum(float(item.z) for item in run) / count
    return EllipticalSlice(z=z, rx=0.0, ry=0.0, cx=cx, cy=cy)


def _prepare_slices(
    slices: Sequence[EllipticalSlice],
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> Tuple[List[EllipticalSlice], int]:
    prepared: List[EllipticalSlice] = []
    degenerate_count = 0
    run: List[EllipticalSlice] = []

    def flush_run() -> None:
        nonlocal run
        if run:
            prepared.append(_collapse_degenerate_slice_run(run))
            run = []

    for slice_data in slices:
        if _is_degenerate_slice(slice_data, min_radius_u, weld_degenerate_rings):
            degenerate_count += 1
            run.append(slice_data)
            continue
        flush_run()
        prepared.append(slice_data)
    flush_run()

    if not any(
        not _is_degenerate_slice(slice_data, min_radius_u, weld_degenerate_rings)
        for slice_data in prepared
    ):
        raise ValueError("At least one non-degenerate slice is required")

    for idx, slice_data in enumerate(prepared):
        if not _is_degenerate_slice(slice_data, min_radius_u, weld_degenerate_rings):
            continue
        if idx not in {0, len(prepared) - 1}:
            raise ValueError(
                "Interior zero-radius slices are not supported in a single loft lobe"
            )

    return prepared, degenerate_count


def _resolve_radial_segments(
    slices: Sequence[EllipticalSlice],
    radial_segments: int,
    adaptive_radial_segments: bool,
    target_edge_error_u: Optional[float],
) -> Tuple[int, Tuple[str, ...]]:
    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")

    warnings: List[str] = []
    resolved = radial_segments
    if radial_segments < 16:
        warnings.append("radial_segments_below_recommended_minimum:16")

    if adaptive_radial_segments:
        max_radius = max(
            max(abs(float(slice_data.rx)), abs(float(slice_data.ry)))
            for slice_data in slices
        )
        edge_error = (
            float(target_edge_error_u)
            if target_edge_error_u is not None and target_edge_error_u > 0
            else max(max_radius / 12.0, 1e-6)
        )
        adaptive_segments = int(math.ceil((2.0 * math.pi * max_radius) / edge_error))
        resolved = max(resolved, min(max(adaptive_segments, 3), 256))

    return resolved, tuple(warnings)


def _ring_vertices(
    bm: "bmesh.types.BMesh",
    slice_data: EllipticalSlice,
    radial_segments: int,
    min_radius_u: float,
    weld_degenerate_rings: bool,
) -> Tuple[List["bmesh.types.BMVert"], bool]:
    rx = float(slice_data.rx)
    ry = float(slice_data.ry)
    center_x = float(slice_data.cx) if slice_data.cx is not None else 0.0
    center_y = float(slice_data.cy) if slice_data.cy is not None else 0.0

    if weld_degenerate_rings and (rx <= min_radius_u or ry <= min_radius_u):
        vert = bm.verts.new((center_x, center_y, slice_data.z))
        return [vert], True

    verts = []
    for i in range(radial_segments):
        theta = (2.0 * math.pi * i) / radial_segments
        x = center_x + rx * math.cos(theta)
        y = center_y + ry * math.sin(theta)
        verts.append(bm.verts.new((x, y, slice_data.z)))

    return verts, False


def _bridge_rings(
    bm: "bmesh.types.BMesh",
    ring_a: List["bmesh.types.BMVert"],
    ring_b: List["bmesh.types.BMVert"],
) -> None:
    if len(ring_a) == 1 and len(ring_b) == 1:
        return

    if len(ring_a) == 1:
        center = ring_a[0]
        for i in range(len(ring_b)):
            v1 = ring_b[i]
            v2 = ring_b[(i + 1) % len(ring_b)]
            bm.faces.new((center, v1, v2))
        return

    if len(ring_b) == 1:
        center = ring_b[0]
        for i in range(len(ring_a)):
            v1 = ring_a[i]
            v2 = ring_a[(i + 1) % len(ring_a)]
            bm.faces.new((v1, v2, center))
        return

    count = min(len(ring_a), len(ring_b))
    for i in range(count):
        v1 = ring_a[i]
        v2 = ring_a[(i + 1) % count]
        v3 = ring_b[(i + 1) % count]
        v4 = ring_b[i]
        try:
            bm.faces.new((v1, v2, v3, v4))
        except ValueError:
            continue


def _cap_ring(
    bm: "bmesh.types.BMesh",
    ring: List["bmesh.types.BMVert"],
    cap_mode: str,
) -> None:
    if len(ring) < 3:
        return

    if cap_mode == "fan":
        center_coords = [sum(v.co[i] for v in ring) / len(ring) for i in range(3)]
        center = bm.verts.new(center_coords)
        for i in range(len(ring)):
            v1 = ring[i]
            v2 = ring[(i + 1) % len(ring)]
            try:
                bm.faces.new((center, v1, v2))
            except ValueError:
                continue
        return

    if cap_mode == "ngon":
        try:
            bm.faces.new(ring)
        except ValueError:
            edges = []
            for i in range(len(ring)):
                v1 = ring[i]
                v2 = ring[(i + 1) % len(ring)]
                try:
                    edges.append(bm.edges.new((v1, v2)))
                except ValueError:
                    continue
            if edges:
                bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
        return

    if cap_mode != "none":
        raise ValueError(f"Unknown cap_mode: {cap_mode}")


def create_loft_mesh_from_slices(
    slices: Sequence[EllipticalSlice],
    *,
    name: str = "LoftMesh",
    radial_segments: int = 24,
    adaptive_radial_segments: bool = False,
    target_edge_error_u: Optional[float] = None,
    cap_mode: str = "fan",
    min_radius_u: float = 0.0,
    merge_threshold_u: float = 0.0,
    recalc_normals: bool = True,
    shade_smooth: bool = True,
    weld_degenerate_rings: bool = True,
    surface_mode: str = "smooth",
    surface_subdivisions: int = 4,
) -> Optional[object]:
    """Create a Blender mesh object lofted from elliptical slices."""
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return None

    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")

    if not slices:
        raise ValueError("slices must not be empty")

    radial_segments, radial_warnings = _resolve_radial_segments(
        slices, radial_segments, adaptive_radial_segments, target_edge_error_u
    )
    surface_slices = prepare_loft_surface(slices, surface_mode, surface_subdivisions)
    prepared_slices, degenerate_count = _prepare_slices(
        surface_slices, min_radius_u, weld_degenerate_rings
    )

    bm = bmesh.new()

    rings: List[List[bmesh.types.BMVert]] = []
    for slice_data in prepared_slices:
        ring, _ = _ring_vertices(
            bm,
            slice_data,
            radial_segments,
            min_radius_u,
            weld_degenerate_rings,
        )
        rings.append(ring)

    for ring_a, ring_b in zip(rings[:-1], rings[1:]):
        _bridge_rings(bm, ring_a, ring_b)

    if cap_mode != "none":
        _cap_ring(bm, rings[0], cap_mode)
        _cap_ring(bm, rings[-1], cap_mode)

    if merge_threshold_u > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_threshold_u)

    if recalc_normals:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    if shade_smooth and surface_mode != "sharp":
        for polygon in obj.data.polygons:
            # Caps and authored step shoulders stay planar. Blending their
            # normals into the wall makes a closed lip look melted.
            zs = [obj.data.vertices[i].co.z for i in polygon.vertices]
            polygon.use_smooth = max(zs) - min(zs) > 1e-9
        for edge in obj.data.edges:
            a, b = (obj.data.vertices[i].co for i in edge.vertices)
            if abs(a.z - b.z) <= 1e-9 and (
                surface_mode == "stepped" or
                abs(a.z - prepared_slices[0].z) <= 1e-9 or
                abs(a.z - prepared_slices[-1].z) <= 1e-9
            ):
                edge.use_edge_sharp = True

    quality = collect_mesh_quality(obj)
    metadata = {
        "surface_mode": surface_mode,
        "surface_subdivisions": surface_subdivisions,
        "radial_segments": radial_segments,
        "adaptive_radial_segments": adaptive_radial_segments,
        "target_edge_error_u": target_edge_error_u,
        "ring_count": len(rings),
        "input_slice_count": len(slices),
        "prepared_slice_count": len(prepared_slices),
        "degenerate_count": degenerate_count,
        "face_count": len(obj.data.polygons),
        "cap_mode": cap_mode,
        "weld_threshold": merge_threshold_u,
        "min_radius_u": min_radius_u,
        "warnings": list(radial_warnings),
        "quality": quality.to_dict(),
    }
    recipe = {
        "schema_version": 1,
        "slices": [{"z": float(s.z), "rx": float(s.rx), "ry": float(s.ry),
                    "cx": float(s.cx or 0.), "cy": float(s.cy or 0.)} for s in slices],
        "options": {"radial_segments": radial_segments,
                    "adaptive_radial_segments": adaptive_radial_segments,
                    "target_edge_error_u": target_edge_error_u, "cap_mode": cap_mode,
                    "min_radius_u": min_radius_u, "merge_threshold_u": merge_threshold_u,
                    "recalc_normals": recalc_normals, "shade_smooth": shade_smooth,
                    "weld_degenerate_rings": weld_degenerate_rings,
                    "surface_mode": surface_mode, "surface_subdivisions": surface_subdivisions},
    }
    obj["loft_recipe_json"] = json.dumps(recipe, sort_keys=True)
    obj["loft_metadata_json"] = json.dumps(metadata, sort_keys=True)
    obj["mesh_quality_json"] = json.dumps(quality.to_dict(), sort_keys=True)

    return obj



def rebuild_loft_mesh(obj, *, slices=None, **options):
    """Regenerate a saved loft after an artist edit, preserving object identity.

    Validation/build happens before replacing geometry. Existing transforms,
    tags, materials, parent, collection membership and modifiers stay attached.
    The receipt describes the base geometry; downstream modifiers remain live.
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("loft regeneration requires Blender")
    if getattr(obj, "type", None) != "MESH" or not obj.get("loft_recipe_json"):
        raise ValueError("object has no saved loft recipe")
    recipe = json.loads(obj["loft_recipe_json"])
    if recipe.get("schema_version") != 1:
        raise ValueError("unsupported loft recipe version")
    original_options = recipe["options"]
    unknown = set(options) - set(original_options)
    if unknown:
        raise ValueError("unknown loft edit options: " + ", ".join(sorted(unknown)))
    sections = ([EllipticalSlice(**row) for row in recipe["slices"]]
                if slices is None else list(slices))
    replacement = create_loft_mesh_from_slices(
        sections, name=obj.name + "_Regenerated", **{**original_options, **options})
    old_mesh = obj.data
    new_mesh = replacement.data
    try:
        for material in old_mesh.materials:
            new_mesh.materials.append(material)
        obj.data = new_mesh
        for key in ("loft_recipe_json", "loft_metadata_json", "mesh_quality_json"):
            obj[key] = replacement[key]
    finally:
        bpy.data.objects.remove(replacement, do_unlink=True)
    # Reclaim only the replaced generated datablock when it has no other users.
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return obj
