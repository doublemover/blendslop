"""Blender-only tests for loft mesh generation."""

from __future__ import annotations

import math
import json
import sys
import unittest
from pathlib import Path

import bpy
import bmesh

sys.path.insert(0, str(Path(__file__).parent))

from geometry.profile_models import EllipticalSlice
from integration.blender_ops.profile_loft_mesh import create_loft_mesh_from_slices


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _bounds_for_object(
    obj: bpy.types.Object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_x = min(v.x for v in coords)
    max_x = max(v.x for v in coords)
    min_y = min(v.y for v in coords)
    max_y = max(v.y for v in coords)
    min_z = min(v.z for v in coords)
    max_z = max(v.z for v in coords)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


class TestProfileLoftMesh(unittest.TestCase):
    def test_surface_modes_are_connected_and_keep_flat_caps(self) -> None:
        slices = [EllipticalSlice(z=z, rx=r, ry=r)
                  for z, r in ((0.0, 1.0), (1.0, .6), (2.0, 1.0))]
        for mode in ("smooth", "stepped", "sharp"):
            with self.subTest(mode=mode):
                _clear_scene()
                obj = create_loft_mesh_from_slices(slices, radial_segments=48,
                                                  surface_mode=mode)
                bm = bmesh.new()
                bm.from_mesh(obj.data)
                self.assertFalse([edge for edge in bm.edges if not edge.is_manifold])
                bm.verts.ensure_lookup_table()
                seen = set()
                pending = [bm.verts[0]]
                while pending:
                    vertex = pending.pop()
                    if vertex in seen:
                        continue
                    seen.add(vertex)
                    pending.extend(edge.other_vert(vertex) for edge in vertex.link_edges)
                self.assertEqual(len(seen), len(bm.verts))
                bm.free()
                planar = [p for p in obj.data.polygons if
                          max(obj.data.vertices[i].co.z for i in p.vertices) -
                          min(obj.data.vertices[i].co.z for i in p.vertices) <= 1e-9]
                self.assertTrue(planar)
                self.assertFalse(any(p.use_smooth for p in planar))
                if mode == "sharp":
                    self.assertFalse(any(p.use_smooth for p in obj.data.polygons))
                else:
                    self.assertTrue(any(p.use_smooth for p in obj.data.polygons))
                self.assertEqual(json.loads(obj["loft_metadata_json"])["surface_mode"], mode)

    def test_loft_cylinder_bounds(self) -> None:
        _clear_scene()
        slices = [
            EllipticalSlice(z=z, rx=1.0, ry=1.0) for z in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]
        obj = create_loft_mesh_from_slices(
            slices, name="TestCylinder", radial_segments=24, cap_mode="fan"
        )

        bounds_min, bounds_max = _bounds_for_object(obj)
        width = bounds_max[0] - bounds_min[0]
        depth = bounds_max[1] - bounds_min[1]
        height = bounds_max[2] - bounds_min[2]

        self.assertAlmostEqual(width, 2.0, delta=0.05)
        self.assertAlmostEqual(depth, 2.0, delta=0.05)
        self.assertAlmostEqual(height, 2.0, delta=0.05)

    def test_loft_elliptical_bounds(self) -> None:
        _clear_scene()
        slices = [EllipticalSlice(z=z, rx=2.0, ry=1.0) for z in [0.0, 1.5, 3.0]]
        obj = create_loft_mesh_from_slices(
            slices, name="TestElliptical", radial_segments=24, cap_mode="fan"
        )

        bounds_min, bounds_max = _bounds_for_object(obj)
        width = bounds_max[0] - bounds_min[0]
        depth = bounds_max[1] - bounds_min[1]
        height = bounds_max[2] - bounds_min[2]

        self.assertAlmostEqual(width, 4.0, delta=0.05)
        self.assertAlmostEqual(depth, 2.0, delta=0.05)
        self.assertAlmostEqual(height, 3.0, delta=0.05)

    def test_loft_manifoldness(self) -> None:
        _clear_scene()
        slices = [EllipticalSlice(z=z, rx=1.0, ry=1.0) for z in [0.0, 1.0, 2.0]]
        obj = create_loft_mesh_from_slices(
            slices, name="TestManifold", radial_segments=24, cap_mode="fan"
        )

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        non_manifold = [edge for edge in bm.edges if len(edge.link_faces) != 2]
        bm.free()

        self.assertEqual(len(non_manifold), 0)

    def test_loft_determinism(self) -> None:
        _clear_scene()
        slices = [EllipticalSlice(z=z, rx=1.5, ry=0.5) for z in [0.0, 1.0, 2.0]]
        obj_a = create_loft_mesh_from_slices(
            slices, name="TestDeterminismA", radial_segments=24, cap_mode="fan"
        )
        obj_b = create_loft_mesh_from_slices(
            slices, name="TestDeterminismB", radial_segments=24, cap_mode="fan"
        )

        coords_a = sorted(
            [
                (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6))
                for v in obj_a.data.vertices
            ]
        )
        coords_b = sorted(
            [
                (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6))
                for v in obj_b.data.vertices
            ]
        )

        self.assertEqual(coords_a, coords_b)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
