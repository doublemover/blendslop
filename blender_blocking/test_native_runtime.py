"""Current stable runtime and ordinary native boolean correctness checks."""

import unittest

try:
    import bpy
except ImportError:
    bpy = None


@unittest.skipIf(bpy is None, "native Blender required")
class NativeRuntimeTests(unittest.TestCase):
    def test_current_release_identity(self):
        from utils.blender_version import (
            SUPPORTED_BLENDER_VERSION, get_version_info, require_supported_blender,
        )
        require_supported_blender()
        self.assertEqual(bpy.app.version, SUPPORTED_BLENDER_VERSION)
        self.assertEqual(get_version_info()["version"], bpy.app.version)

    def test_current_solver_enums(self):
        from utils.blender_version import (
            get_available_boolean_solvers, get_boolean_solver, resolve_boolean_solver,
        )
        self.assertEqual(get_boolean_solver(), "EXACT")
        available = get_available_boolean_solvers()
        for solver in ("FLOAT", "EXACT", "MANIFOLD"):
            self.assertIn(solver, available)
            self.assertEqual(resolve_boolean_solver(solver), solver)

    def test_exact_mesh_join_retains_geometry_and_extent(self):
        from placement.primitive_placement import MeshJoiner
        objects = []
        created_names = []
        result = None
        try:
            for x in (0.0, 0.5):
                bpy.ops.mesh.primitive_cube_add(location=(x, 0, 0))
                objects.append(bpy.context.object)
                created_names.append(bpy.context.object.name)
            result = MeshJoiner().join(objects, target_name="CurrentRuntimeJoin",
                                       mode="boolean", solver="EXACT")
            created_names.append(result.name)
            bpy.context.view_layer.update()
            self.assertGreater(len(result.data.polygons), 0)
            self.assertAlmostEqual(result.dimensions.x, 2.5, places=5)
        finally:
            for name in set(created_names):
                obj = bpy.data.objects.get(name)
                if obj is not None:
                    bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    unittest.main()
