"""
Inspect Phase 2 Step 2 blend file to diagnose primitive size issues.

Visual inspection script - loads blend file and reports on mesh properties.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bpy
import numpy as np
from mathutils import Vector


def inspect_blend_file(blend_path: Path) -> None:
    """Inspect blend file and report mesh properties."""
    print("=" * 70)
    print("PHASE 2 BLEND FILE INSPECTION")
    print("=" * 70)
    print(f"\nFile: {blend_path}")

    # Load blend file
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    print(f"\nObjects in scene: {len(bpy.data.objects)}")

    # Find all meshes
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    print(f"Mesh objects: {len(meshes)}")

    print("\n" + "=" * 70)
    print("MESH DETAILS")
    print("=" * 70)

    for obj in meshes:
        print(f"\n{obj.name}:")
        print(f"  Type: {obj.type}")
        print(f"  Vertices: {len(obj.data.vertices):,}")
        print(f"  Faces: {len(obj.data.polygons):,}")
        print(f"  Location: {obj.location}")
        print(f"  Rotation: {obj.rotation_euler}")
        print(f"  Scale: {obj.scale}")

        # Get bounding box
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        bounds_min = Vector(
            (min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox))
        )
        bounds_max = Vector(
            (max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox))
        )

        size = bounds_max - bounds_min
        print(f"  Bounding box:")
        print(f"    Min: ({bounds_min.x:.3f}, {bounds_min.y:.3f}, {bounds_min.z:.3f})")
        print(f"    Max: ({bounds_max.x:.3f}, {bounds_max.y:.3f}, {bounds_max.z:.3f})")
        print(f"    Size: ({size.x:.3f}, {size.y:.3f}, {size.z:.3f})")
        print(f"    Volume estimate: {size.x * size.y * size.z:.3f}")

        # Sample some vertex positions
        if len(obj.data.vertices) > 0:
            sample_verts = [
                obj.data.vertices[i].co for i in range(min(5, len(obj.data.vertices)))
            ]
            print(f"  Sample vertices (first 5):")
            for i, v in enumerate(sample_verts):
                world_pos = obj.matrix_world @ v
                print(f"    {i}: local={v}, world={world_pos}")

    # Compare meshes
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if len(meshes) >= 2:
        # Assume first is Visual Hull, second is Primitives
        vh = (
            meshes[0]
            if "VisualHull" in meshes[0].name or meshes[0].name.startswith("Visual")
            else meshes[1]
        )
        prim = meshes[1] if vh == meshes[0] else meshes[0]

        # Recalculate bounds for comparison
        vh_bbox = [vh.matrix_world @ Vector(corner) for corner in vh.bound_box]
        vh_min = Vector(
            (
                min(v.x for v in vh_bbox),
                min(v.y for v in vh_bbox),
                min(v.z for v in vh_bbox),
            )
        )
        vh_max = Vector(
            (
                max(v.x for v in vh_bbox),
                max(v.y for v in vh_bbox),
                max(v.z for v in vh_bbox),
            )
        )
        vh_size = vh_max - vh_min

        prim_bbox = [prim.matrix_world @ Vector(corner) for corner in prim.bound_box]
        prim_min = Vector(
            (
                min(v.x for v in prim_bbox),
                min(v.y for v in prim_bbox),
                min(v.z for v in prim_bbox),
            )
        )
        prim_max = Vector(
            (
                max(v.x for v in prim_bbox),
                max(v.y for v in prim_bbox),
                max(v.z for v in prim_bbox),
            )
        )
        prim_size = prim_max - prim_min

        print(f"\nVisual Hull ({vh.name}):")
        print(f"  Size: ({vh_size.x:.3f}, {vh_size.y:.3f}, {vh_size.z:.3f})")
        print(f"  Volume: {vh_size.x * vh_size.y * vh_size.z:.3f}")

        print(f"\nPrimitives ({prim.name}):")
        print(f"  Size: ({prim_size.x:.3f}, {prim_size.y:.3f}, {prim_size.z:.3f})")
        print(f"  Volume: {prim_size.x * prim_size.y * prim_size.z:.3f}")

        print(f"\nSize Ratio (Primitives / Visual Hull):")
        print(f"  X: {prim_size.x / vh_size.x if vh_size.x > 0 else 'N/A':.2f}x")
        print(f"  Y: {prim_size.y / vh_size.y if vh_size.y > 0 else 'N/A':.2f}x")
        print(f"  Z: {prim_size.z / vh_size.z if vh_size.z > 0 else 'N/A':.2f}x")

        vol_ratio = (
            (prim_size.x * prim_size.y * prim_size.z)
            / (vh_size.x * vh_size.y * vh_size.z)
            if (vh_size.x * vh_size.y * vh_size.z) > 0
            else 0
        )
        print(f"  Volume: {vol_ratio:.2f}x")

        if vol_ratio > 10:
            print(
                f"\n  ✗ PROBLEM: Primitives are {vol_ratio:.1f}x larger than Visual Hull!"
            )
            print(f"    This explains the inflated radii and low IoU.")
        elif vol_ratio > 2:
            print(
                f"\n  ⚠ WARNING: Primitives are {vol_ratio:.1f}x larger than expected"
            )
        else:
            print(f"\n  ✓ Size ratio looks reasonable")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    blend_file = Path(__file__).resolve().parents[1] / "temp" / "phase2_step2.blend"
    if not blend_file.exists():
        print(f"Error: {blend_file} not found")
        sys.exit(1)

    inspect_blend_file(blend_file)
