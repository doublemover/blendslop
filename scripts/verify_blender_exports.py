"""Bounded real Blender export checks; all generated assets stay in --output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]


def world_vertices(objects, *, evaluated=False):
    import bpy
    points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh_obj = obj.evaluated_get(depsgraph) if evaluated else obj
        mesh = mesh_obj.to_mesh() if evaluated else obj.data
        try:
            points.extend(tuple(mesh_obj.matrix_world @ vertex.co) for vertex in mesh.vertices)
        finally:
            if evaluated:
                mesh_obj.to_mesh_clear()
    return points


def world_bounds(objects, *, evaluated=False):
    points = world_vertices(objects, evaluated=evaluated)
    return {"min": [min(p[i] for p in points) for i in range(3)],
            "max": [max(p[i] for p in points) for i in range(3)]}


def bounds_error(expected, actual):
    return max(abs(expected[k][i] - actual[k][i]) for k in ("min", "max") for i in range(3))


def vertex_distance(expected, actual):
    from scipy.spatial import cKDTree
    return max(float(cKDTree(expected).query(actual)[0].max()),
               float(cKDTree(actual).query(expected)[0].max()))


def triangle_count(objects, *, evaluated=False):
    import bpy
    count = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh_obj = obj.evaluated_get(depsgraph) if evaluated else obj
        mesh = mesh_obj.to_mesh() if evaluated else obj.data
        try:
            mesh.calc_loop_triangles()
            count += len(mesh.loop_triangles)
        finally:
            if evaluated:
                mesh_obj.to_mesh_clear()
    return count


def main():
    import bpy
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    from blender_blocking.integration.blender_ops.export_qa import run_export_roundtrip_qa
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-length", type=float, default=.01)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = args.scale_length
    expected_scale_length = scene.unit_settings.scale_length
    root = bpy.data.objects.new("CompatRoot", None)
    scene.collection.objects.link(root)
    root.location = (2.0, -3.0, 1.0)
    root.rotation_euler = (0.13, 0.21, 0.37)
    root.scale = (1.2, 0.8, 1.1)
    material = bpy.data.materials.new("CompatMaterial")
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.1, 0.5, 0.8, 1)
    objects = [root]
    for i, location in enumerate(((0, 0, 0), (1.5, 0.2, 0.7))):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.name = f"CompatMesh{i}"
        obj.parent = root
        obj.location = location
        obj.scale = (1.0, 0.6 + 0.1 * i, 1.4)
        obj.data.materials.append(material)
        bevel = obj.modifiers.new("CompatBevel", "BEVEL")
        bevel.width = 0.07
        bevel.segments = 2
        objects.append(obj)
    bpy.context.view_layer.update()
    expected = world_bounds(objects, evaluated=True)
    expected_vertices = world_vertices(objects, evaluated=True)
    expected_triangles = triangle_count(objects, evaluated=True)
    source_transforms = [(obj, tuple(obj.location), tuple(obj.scale)) for obj in objects]
    results = []
    for target in ("obj", "glb"):
        before = set(bpy.data.objects.keys())
        report = run_export_roundtrip_qa(objects, out / target, targets=(target,), cleanup_imports=False)[0]
        imported = [obj for obj in bpy.data.objects if obj.name not in before]
        actual = world_bounds(imported) if any(obj.type == "MESH" for obj in imported) else None
        # glTF stores metres and its importer restores the receiving scene's units.
        # The source and receiving scene are identical, so world coordinates must agree.
        error = bounds_error(expected, actual) if actual else None
        point_error = vertex_distance(expected_vertices, world_vertices(imported)) if actual else None
        triangles = triangle_count(imported)
        restored = all(tuple(obj.location) == location and tuple(obj.scale) == scale for obj, location, scale in source_transforms)
        hierarchy = any(obj.parent is not None for obj in imported)
        result = {**report.to_dict(), "world_bounds": actual, "max_bounds_error": error,
                  "max_vertex_distance": point_error, "triangles": triangles, "source_transforms_restored": restored,
                  "hierarchy_present": hierarchy, "modifier_applied": triangles == expected_triangles and point_error is not None and point_error < 1e-4}
        result["status"] = "pass" if report.status_ok and error is not None and error < 1e-4 and result["modifier_applied"] and restored and (report.material_count or 0) > 0 and (target != "glb" or hierarchy) else "fail"
        results.append(result)
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[1]
    stl = out / "roundtrip.stl"
    try:
        bpy.ops.wm.stl_export(filepath=str(stl), export_selected_objects=True, apply_modifiers=True, use_scene_unit=False)
        before = set(bpy.data.objects.keys())
        bpy.ops.wm.stl_import(filepath=str(stl), use_scene_unit=False)
        imported = [obj for obj in bpy.data.objects if obj.name not in before]
        error = bounds_error(expected, world_bounds(imported))
        results.append({"target": "stl", "status": "pass" if error < 1e-4 else "fail", "max_bounds_error": error,
                        "material_and_hierarchy": "not supported by format", "coordinate_units": "Blender units; scene-unit conversion disabled"})
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
    except Exception as exc:
        results.append({"target": "stl", "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    blend = out / "roundtrip.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    reopened = [bpy.data.objects.get(obj_name) for obj_name in ("CompatRoot", "CompatMesh0", "CompatMesh1")]
    error = bounds_error(expected, world_bounds(reopened, evaluated=True))
    blend_ok = all(obj is not None for obj in reopened) and all(obj.parent == reopened[0] and len(obj.modifiers) == 1 and len(obj.data.materials) == 1 for obj in reopened[1:]) and bpy.context.scene.unit_settings.scale_length == expected_scale_length
    results.append({"target": "blend", "status": "pass" if blend_ok and error < 1e-4 else "fail", "max_bounds_error": error,
                    "modifiers_materials_hierarchy": blend_ok, "scale_length": bpy.context.scene.unit_settings.scale_length})
    payload = {"blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode(), "python": sys.version,
               "fixture": "two transformed children, nonuniform parent scale/rotation, bevel modifiers, PBR material",
               "scale_length": expected_scale_length, "expected_world_bounds": expected, "expected_triangles": expected_triangles, "reports": results,
               "notes": "OBJ/STL flatten hierarchy; STL has no materials. No texture/animation/UV fidelity claim."}
    (out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if any(row["status"] != "pass" for row in results):
        raise RuntimeError("Export compatibility check failed; see result.json")


if __name__ == "__main__":
    main()
