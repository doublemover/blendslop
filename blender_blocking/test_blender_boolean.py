#!/usr/bin/env python3
"""
Blender-specific integration test for boolean operations.

This test MUST be run inside Blender to verify current boolean API correctness.

Usage:
    blender --background --python test_blender_boolean.py

The supported stable release provides EXACT, FLOAT and MANIFOLD solvers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_boolean_solver_enum() -> bool:
    """Test that boolean modifier uses valid solver enum."""
    print("\n" + "=" * 60)
    print("TEST: Current Boolean Solver Enum")
    print("=" * 60)

    try:
        import bpy
    except ImportError:
        print("❌ FAILED: Must run inside Blender")
        print("Usage: blender --background --python test_blender_boolean.py")
        return False

    print(f"✓ Running in Blender {bpy.app.version_string}")

    # Clean scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create two cubes to test boolean operation
    print("Creating test objects...")
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    cube1 = bpy.context.active_object
    cube1.name = "Cube1"

    bpy.ops.mesh.primitive_cube_add(location=(0.5, 0, 0))
    cube2 = bpy.context.active_object
    cube2.name = "Cube2"

    # Test boolean modifier with EXACT solver
    print("Testing boolean modifier with EXACT solver...")
    try:
        modifier = cube1.modifiers.new(name="BoolTest", type="BOOLEAN")
        modifier.operation = "UNION"
        modifier.object = cube2
        modifier.solver = "EXACT"  # Should work in the current stable release
        print("✓ EXACT solver accepted")
    except TypeError as e:
        print(f"❌ FAILED: EXACT solver not valid: {e}")
        return False

    # Try to apply the modifier
    print("Applying boolean modifier...")
    try:
        bpy.context.view_layer.objects.active = cube1
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        print("✓ Boolean modifier applied successfully")
    except Exception as e:
        print(f"❌ FAILED: Could not apply modifier: {e}")
        return False

    # Verify the result
    if cube1.data.vertices:
        print(f"✓ Result has {len(cube1.data.vertices)} vertices")
    else:
        print("❌ FAILED: Result has no vertices")
        return False

    print("\n" + "=" * 60)
    print("✓ Boolean solver test PASSED")
    print("=" * 60)
    return True


def test_mesh_joiner() -> bool:
    """Test the MeshJoiner class boolean operations."""
    print("\n" + "=" * 60)
    print("TEST: MeshJoiner Boolean Operations")
    print("=" * 60)

    try:
        import bpy
    except ImportError:
        print("❌ FAILED: Must run inside Blender")
        return False

    from blender_blocking.placement.primitive_placement import MeshJoiner

    # Clean scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Create test objects
    print("Creating test primitives...")
    bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0))
    obj1 = bpy.context.active_object

    bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 1))
    obj2 = bpy.context.active_object

    bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 2))
    obj3 = bpy.context.active_object

    objects = [obj1, obj2, obj3]

    # Test MeshJoiner
    print("Testing MeshJoiner.join_with_boolean_union...")
    try:
        joiner = MeshJoiner()
        result = joiner.join_with_boolean_union(objects, target_name="Test_Result")
        print(f"✓ Boolean union successful: {result.name}")
        print(f"  Vertices: {len(result.data.vertices)}")
        print(f"  Faces: {len(result.data.polygons)}")
    except Exception as e:
        print(f"❌ FAILED: MeshJoiner error: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ MeshJoiner test PASSED")
    print("=" * 60)
    return True


def run_all_blender_tests() -> int:
    """Run all Blender-specific tests."""
    print("\n" + "=" * 70)
    print("BLENDER INTEGRATION TEST SUITE")
    print("=" * 70)

    try:
        import bpy

        print(f"Blender version: {bpy.app.version_string}")
        print(f"Python version: {sys.version}")
    except ImportError:
        print("\n❌ CRITICAL: Must run inside Blender")
        print("\nUsage:")
        print("  blender --background --python test_blender_boolean.py")
        print("\nOr from Blender's Python console:")
        print("  exec(open('test_blender_boolean.py').read())")
        sys.exit(1)

    results = {}

    # Test 1: Boolean solver enum
    results["boolean_solver"] = test_boolean_solver_enum()

    # Test 2: MeshJoiner integration
    results["mesh_joiner"] = test_mesh_joiner()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in results.items():
        if result:
            status = "✓ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        print(f"  {test_name}: {status}")

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests PASSED")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_all_blender_tests()
