#!/usr/bin/env python3
"""Seven retained source-control transactions; no fit, render or quality gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "blender_blocking"), str(ROOT), str(ROOT / "scripts")]
import test_runner
from run_surface_quality_check import _write
from run_quality_coverage_check import save_mesh

FAMILIES = ("sphere", "anisotropic_ellipsoid", "cylinder", "tapered_frustum",
            "capsule", "rounded_box", "thin_plate")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_receipt(owner, receipt):
    """Publish current bytes and refresh their owned digest after every update."""
    _write(owner.root/"results.json", receipt)
    owner.register_file("results.json", "final_output")


def edited_recipe(name, original):
    """Declare specific source controls before any native edit observation."""
    if name not in FAMILIES:
        raise ValueError("family has no frozen source-control transaction")
    wire = deepcopy(original)
    if len(wire.get("root_nodes", ())) != 1:
        raise ValueError("one retained source node is required")
    p = wire["root_nodes"][0]["parameters"]
    if name == "sphere":
        for key in ("width_world", "depth_world", "height_world"):
            p[key] *= 1.05
        control = "coupled fitted radial dimensions x 1.05"
    elif name in {"anisotropic_ellipsoid", "thin_plate"}:
        p["depth_world"] *= 1.05
        control = "local depth_world x 1.05"
    elif name == "cylinder":
        for key in ("radius_bottom", "radius_top", "width_world", "depth_world"):
            p[key] *= 1.05
        control = "coupled cylinder radii and radial dimensions x 1.05"
    elif name == "tapered_frustum":
        p["radius_top"] *= 1.10
        control = "radius_top x 1.10 with bottom radius and height fixed"
    elif name == "capsule":
        p["segment_height_world"] *= 1.05
        p["height_world"] = p["segment_height_world"] + 2*p["radius_world"]
        control = "straight segment_height_world x 1.05 with cap radius fixed"
    else:
        p["corner_radius_world"] *= 1.10
        control = "live BEVEL width x 1.10 with eight segments fixed"
    return wire, control


def physical_observation(name, data, source):
    """Measure evaluated world geometry in its actual native rotation frame."""
    import numpy as np
    rotation = np.asarray(source.matrix_world.to_quaternion().to_matrix(), float)
    center = np.asarray(source.matrix_world.translation, float)
    local = (data.vertices-center) @ rotation
    result = {"extents_world": np.ptp(local, axis=0).tolist()}
    if name == "tapered_frustum":
        for side, extremum in (("top", local[:,2].max()), ("bottom", local[:,2].min())):
            ring = local[np.isclose(local[:,2], extremum, atol=1e-6, rtol=0)]
            result[side+"_radius_world"] = float(np.linalg.norm(ring[:,:2], axis=1).max())
    if name == "rounded_box":
        face = local[np.isclose(local[:,0], local[:,0].max(), atol=1e-6, rtol=0)]
        result["corner_cut_world"] = float(local[:,1].max()-face[:,1].max())
    return result


def response_contract(name, before, after, parameters):
    """Evaluate physical edit response; the tolerance is not an identity gate."""
    import numpy as np
    a, b = np.asarray(before["extents_world"]), np.asarray(after["extents_world"])
    ratios = b/a
    if name == "sphere":
        expected = [1.05]*3
    elif name in {"anisotropic_ellipsoid", "thin_plate"}:
        expected = [1.,1.05,1.]
    elif name == "cylinder":
        expected = [1.05,1.05,1.]
    else:
        expected = [1.,1.,1.]
    checks = {"extents_response": bool(np.allclose(ratios, expected, atol=1e-5, rtol=0))}
    extra = {}
    if name == "capsule":
        expected_delta = .05*parameters["segment_height_world"]
        checks["extents_response"] = bool(np.allclose(ratios[:2], [1.,1.], atol=1e-5, rtol=0))
        checks["axial_segment_response"] = abs(float(b[2]-a[2])-expected_delta) <= 1e-5
        extra = {"height_delta_world": float(b[2]-a[2]), "expected_height_delta_world": expected_delta}
    elif name == "tapered_frustum":
        ratio = after["top_radius_world"]/before["top_radius_world"]
        checks["top_radius_response"] = abs(ratio-1.10) <= 1e-5
        checks["bottom_radius_fixed"] = abs(after["bottom_radius_world"]-before["bottom_radius_world"]) <= 1e-5
        extra = {"top_radius_ratio": ratio}
    elif name == "rounded_box":
        ratio = after["corner_cut_world"]/before["corner_cut_world"]
        checks["world_corner_radius_response"] = abs(ratio-1.10) <= 1e-5
        extra = {"corner_cut_ratio": ratio}
    return {"passed": all(checks.values()), "checks": checks, "extents_ratio": ratios.tolist(),
            "physical_observation_absolute_tolerance": 1e-5,
            "identity_comparison": "exact indexed binary64 arrays, without tolerance", **extra}


def _pose_controls(source):
    return {"rotation_mode": source.rotation_mode,
            **{name: tuple(getattr(source, name)) for name in
               ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")}}


def _apply_pose(source, controls):
    # Restore native controls directly; assigning a captured matrix would cause
    # another decomposition and can change binary32 pose components.
    source.rotation_mode = controls["rotation_mode"]
    for name, values in controls.items():
        if name != "rotation_mode":
            setattr(source, name, values)


def _cleanup(compiled):
    """Remove only transient objects and unused meshes created by this job."""
    import bpy
    for obj in reversed((*compiled.objects, compiled.root_object)):
        if obj is not None and obj.name in bpy.data.objects:
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)


def _compile_exact(wire, captured):
    import bpy
    from reconstruction.frozen_family import retained_family_program, retained_triangle_indices
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from primitives.shape_program_compiler import compile_shape_program
    compiled = compile_shape_program(retained_family_program(wire), lathe_segments=96, weighted_normals=False)
    children = output_mesh_targets([compiled.root_object])
    if len(children) != 1:
        raise ValueError("one owned mesh descendant is required")
    source = children[0]
    data = evaluated_arrays(compiled.root_object)
    if data.content_hash != captured.content_hash and wire["root_nodes"][0]["primitive_type"] == "ellipsoid":
        indices = retained_triangle_indices(captured, data)
        mesh = bpy.data.meshes.new(source.data.name+"FrozenTriangleOrder")
        mesh.from_pydata([tuple(vertex.co) for vertex in source.data.vertices], [], indices.tolist())
        mesh.update()
        old = source.data
        source.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
        bpy.context.view_layer.update()
        data = evaluated_arrays(compiled.root_object)
    if data.content_hash != captured.content_hash:
        raise ValueError("recompiled baseline changed the original indexed geometry")
    return compiled, source, data


def main():
    import bpy
    import numpy as np
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from reconstruction.frozen_family import retained_family_program
    from primitives.shape_program_compiler import compile_shape_program
    from utils.run_ownership import OwnedRun
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("output must stay below isolated repository temp/tasks")
    selected = json.loads(args.prepared_selection.read_text())
    inputs = {}
    for name in FAMILIES:
        row = selected["cases"][name]
        folder = Path(row["artifact_directory"])
        files = {file: _sha(folder/file) for file in ("program.json", "evaluated-exact.npz", "evaluated.obj")}
        if files["evaluated-exact.npz"] != row["npz_sha256"] or files["evaluated.obj"] != row["obj_sha256"]:
            raise ValueError("saved source archive identity changed: "+name)
        wire = json.loads((folder/"program.json").read_text())
        if wire != row["program"]:
            raise ValueError("saved source recipe differs from its selected receipt: "+name)
        inputs[name] = {"directory": str(folder.resolve()), "sha256": files,
                        "geometry_hash": row["geometry_hash"], "original_recipe": wire}
    frozen = {"protocol": "retained_family_source_edit_v1", "families": list(FAMILIES),
              "prepared_selection": {"path": str(args.prepared_selection.resolve()), "sha256": _sha(args.prepared_selection)},
              "inputs": inputs, "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in
                (Path(__file__), ROOT/"blender_blocking/primitives/shape_program_compiler.py",
                 ROOT/"blender_blocking/reconstruction/frozen_family.py", ROOT/"blender_blocking/test_shape_program_world_bevel.py")},
              "deadline_seconds": 120, "memory_limit_bytes": 8*1024**3, "threads": 2,
              "fits": 0, "renders": 0, "raw_surface_comparisons": 0, "qualification_children": 0,
              "baseline_tessellation": {"radial_segments": 96, "rounded_box_bevel_segments": 8},
              "scope": "specific recipe parameter regeneration or live bevel width; no blanket editability qualification",
              "environment": {"blender": bpy.app.version_string, "python": sys.version, "numpy": np.__version__}}
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(), producer="retained_family_source_edit", max_generated_bytes=33554432,
                     shared_inputs={"prepared_selection": str(args.prepared_selection.resolve())})
    with owner:
        _write(owner.root/"frozen-workload.json", frozen)
        owner.register_file("frozen-workload.json", "diagnostic")
        receipt = {"protocol": frozen["protocol"], "status": "running", "cases": {}, "aggregate_accepted": False}
        _publish_receipt(owner, receipt)
        for name in FAMILIES:
            if time.monotonic()-started > 120:
                raise TimeoutError("frozen source-control deadline exceeded")
            print("source-edit "+name, flush=True)
            folder = owner.root/name
            folder.mkdir()
            compiled = temporary = None
            try:
                entry = inputs[name]
                with np.load(Path(entry["directory"])/"evaluated-exact.npz", allow_pickle=False) as archive:
                    captured = GeometryArrays.capture(archive["vertices"], archive["faces"])
                if captured.content_hash != entry["geometry_hash"]:
                    raise ValueError("captured source geometry identity changed")
                original = entry["original_recipe"]
                effective = deepcopy(original)
                if name == "rounded_box":
                    effective["root_nodes"][0]["parameters"]["bevel_segments"] = 8
                edited, control = edited_recipe(name, effective)
                for filename, value in (("original-program.json", original), ("effective-program.json", effective), ("edited-program.json", edited)):
                    _write(folder/filename, value)
                compiled, source, before = _compile_exact(effective, captured)
                root_pointer, source_pointer = compiled.root_object.as_pointer(), source.as_pointer()
                mesh, pose = source.data, _pose_controls(source)
                before_observation = physical_observation(name, before, source)
                recipe_key = "source_edit_recipe_json"
                old_recipe = source.get(recipe_key)
                bevels = [(m, float(m.width), int(m.segments)) for m in source.modifiers if m.type == "BEVEL"]
                try:
                    source[recipe_key] = json.dumps(edited, sort_keys=True)
                    if name == "rounded_box":
                        if len(bevels) != 1 or bevels[0][2] != 8:
                            raise ValueError("retained live bevel must have eight segments")
                        bevels[0][0].width = edited["root_nodes"][0]["parameters"]["corner_radius_world"]
                        scope = "live BEVEL width on the same owned mesh descendant; eight segments fixed"
                    else:
                        temporary = compile_shape_program(retained_family_program(edited), lathe_segments=96, weighted_normals=False)
                        edited_sources = output_mesh_targets([temporary.root_object])
                        if len(edited_sources) != 1:
                            raise ValueError("edited recipe must produce one mesh descendant")
                        source.data = edited_sources[0].data
                        _apply_pose(source, _pose_controls(edited_sources[0]))
                        scope = "recipe parameter regeneration applied to the same owned root and mesh descendant"
                    bpy.context.view_layer.update()
                    changed = evaluated_arrays(compiled.root_object)
                    observation = physical_observation(name, changed, source)
                    response = response_contract(name, before_observation, observation, effective["root_nodes"][0]["parameters"])
                    save_mesh(compiled.root_object, folder)
                    response["geometry_changed"] = changed.content_hash != before.content_hash
                    response["same_root_pointer"] = compiled.root_object.as_pointer() == root_pointer
                    response["same_source_pointer"] = source.as_pointer() == source_pointer
                finally:
                    source.data = mesh
                    _apply_pose(source, pose)
                    for modifier, width, segments in bevels:
                        modifier.width, modifier.segments = width, segments
                    if old_recipe is None:
                        del source[recipe_key]
                    else:
                        source[recipe_key] = old_recipe
                    bpy.context.view_layer.update()
                restored = evaluated_arrays(compiled.root_object)
                exact_restoration = restored.content_hash == captured.content_hash and source.data == mesh and _pose_controls(source) == pose
                passed = all((response["passed"], response["geometry_changed"], response["same_root_pointer"], response["same_source_pointer"], exact_restoration))
                receipt["cases"][name] = {"status": "passed" if passed else "failed", "control": control, "scope": scope,
                    "baseline_geometry_hash": before.content_hash, "edited_geometry_hash": changed.content_hash,
                    "restored_geometry_hash": restored.content_hash, "exact_restoration": exact_restoration,
                    "original_mesh_pointer_restored": source.data == mesh, "response": response,
                    "before": before_observation, "after": observation,
                    "family_acceptance": "independent surface tolerance remains unavailable"}
            except Exception as exc:
                receipt["cases"][name] = {"status": "failed", "reason": type(exc).__name__+": "+str(exc)}
            finally:
                if temporary is not None:
                    _cleanup(temporary)
                if compiled is not None:
                    _cleanup(compiled)
            _publish_receipt(owner, receipt)
            for path in folder.iterdir():
                if path.is_file():
                    owner.register_file(path.relative_to(owner.root), "final_output")
        from test_shape_program_world_bevel import TestNativeShapeProgramWorldBevel
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestNativeShapeProgramWorldBevel))
        receipt["ordinary_native_world_bevel_tests"] = {"run": result.testsRun, "passed": result.wasSuccessful(), "skipped": len(result.skipped)}
        passed = result.wasSuccessful() and not result.skipped and all(row["status"] == "passed" for row in receipt["cases"].values())
        receipt.update(status="passed" if passed else "failed", elapsed_seconds=time.monotonic()-started,
                       run_root=str(owner.root), acceptance_scope=frozen["scope"])
        _publish_receipt(owner, receipt)
        print("SOURCE_EDIT_RESULT="+str(owner.root/"results.json"), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
