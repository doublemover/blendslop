#!/usr/bin/env python3
"""Saved observed arch/torus semantic edits with exact source restoration."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"blender_blocking"),str(ROOT),str(ROOT/"scripts")]
import test_runner
from run_surface_quality_check import _write
from run_family_source_edit_check import _compile_exact,_cleanup,_pose_controls,_apply_pose,_publish_receipt
from run_quality_coverage_check import save_mesh

FAMILIES = ("concave_arch","torus")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edited_recipe(name, wire):
    edited = deepcopy(wire)
    if len(edited.get("root_nodes",()))!=1:
        raise ValueError("semantic edit requires one additive source leaf")
    node = edited["root_nodes"][0]
    p=node["parameters"]
    if name=="concave_arch":
        outer=p["outer"]
        if node["primitive_type"]!="polygon_extrusion" or p.get("holes") or len(outer)!=8:
            raise ValueError("saved observed arch needs its eight-point open-cavity outline")
        if not all(outer[i][0]>0 for i in (4,5)) or not all(outer[i][0]<0 for i in (6,7)):
            raise ValueError("arch cavity wall indexing differs from frozen observed outline")
        for i in range(4,8):
            outer[i][0]*=1.10
        control="inner cavity wall X coordinates x1.10; outer outline, roof and extrusion depth fixed"
    elif name=="torus":
        if node["primitive_type"]!="torus":
            raise ValueError("tube-radius control requires a torus source")
        p["minor_radius"]*=1.05
        control="minor/tube radius x1.05; major radius fixed"
    else:
        raise ValueError("unknown structured source control")
    return edited,control


def observe(name,data,source):
    import numpy as np
    rotation=np.asarray(source.matrix_world.to_quaternion().to_matrix(),float)
    center=np.asarray(source.matrix_world.translation,float)
    local=(data.vertices-center)@rotation
    result={"extents_world":np.ptp(local,axis=0).tolist()}
    if name=="concave_arch":
        positive=local[:,0][local[:,0]>1e-6]
        negative=local[:,0][local[:,0]<-1e-6]
        if not len(positive) or not len(negative):
            raise ValueError("actual arch lacks distinct inner cavity walls")
        right,left=float(positive.min()),float(negative.max())
        inner=np.isclose(local[:,0],right,atol=1e-6,rtol=0)|np.isclose(local[:,0],left,atol=1e-6,rtol=0)
        result.update(opening_width_world=right-left,inner_roof_local_y=float(local[inner,1].max()))
    else:
        radial=np.linalg.norm(local[:,:2],axis=1)
        outer,inner=float(radial.max()),float(radial.min())
        result.update(outer_radius_world=outer,inner_radius_world=inner,
                      major_radius_world=(outer+inner)*.5,tube_radius_world=(outer-inner)*.5)
    return result


def response(name,before,after):
    import numpy as np
    a,b=np.asarray(before["extents_world"]),np.asarray(after["extents_world"])
    if name=="concave_arch":
        ratio=after["opening_width_world"]/before["opening_width_world"]
        checks={"opening_width_x1_10":abs(ratio-1.10)<=1e-5,
                "outer_dimensions_fixed":bool(np.allclose(a,b,atol=1e-5,rtol=0)),
                "cavity_roof_fixed":abs(after["inner_roof_local_y"]-before["inner_roof_local_y"])<=1e-5}
        values={"opening_width_ratio":ratio}
    else:
        ratio=after["tube_radius_world"]/before["tube_radius_world"]
        checks={"tube_radius_x1_05":abs(ratio-1.05)<=1e-5,
                "major_radius_fixed":abs(after["major_radius_world"]-before["major_radius_world"])<=1e-5,
                "axial_tube_height_x1_05":abs(float(b[2]/a[2])-1.05)<=1e-5,
                "outer_ring_expands":after["outer_radius_world"]>before["outer_radius_world"],
                "hole_narrows":after["inner_radius_world"]<before["inner_radius_world"]}
        values={"tube_radius_ratio":ratio,"major_radius_delta_world":after["major_radius_world"]-before["major_radius_world"]}
    return {"passed":all(checks.values()),"checks":checks,**values,
            "physical_response_absolute_tolerance":1e-5,"geometry_identity":"exact indexed binary64; no tolerance"}


def main():
    import bpy
    import numpy as np
    from reconstruction.native_geometry import GeometryArrays,evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from reconstruction.frozen_family import retained_family_program
    from primitives.shape_program_compiler import compile_shape_program
    from utils.run_ownership import OwnedRun
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/"temp/tasks"):
        raise ValueError("new semantic output must stay under isolated temp/tasks")
    saved=json.loads(args.saved_receipt.read_text())
    inputs={}
    for name in FAMILIES:
        row=saved["cases"][name];folder=Path(row["artifact_directory"])
        if _sha(folder/"evaluated-exact.npz")!=row["npz_sha256"] or _sha(folder/"evaluated.obj")!=row["obj_sha256"]:
            raise ValueError("saved source archive changed")
        wire=json.loads((folder/"program.json").read_text())
        if wire!=row["program"]:
            raise ValueError("saved recipe changed")
        edited,control=edited_recipe(name,wire)
        inputs[name]={"directory":str(folder),"original_recipe":wire,"edited_recipe":edited,"control":control,
                      "required_indexed_hash":row["geometry_hash"],"npz_sha256":row["npz_sha256"],
                      "obj_sha256":row["obj_sha256"],"program_sha256":_sha(folder/"program.json")}
    frozen={"protocol":"structured_family_source_edit_v1","inputs":inputs,
        "saved_receipt":{"path":str(args.saved_receipt.resolve()),"sha256":_sha(args.saved_receipt)},
        "source_sha256":{str(p.relative_to(ROOT)):_sha(p) for p in
          (Path(__file__),ROOT/"scripts/run_family_source_edit_check.py",ROOT/"blender_blocking/primitives/shape_program_compiler.py")},
        "radial_segments":96,"deadline_seconds":60,"memory_limit_bytes":8*1024**3,"threads":2,
        "fits":0,"renders":0,"raw_comparisons":0,"qualification_children":0,
        "scope":"recipe parameter regeneration applied to same owned source/root, not blanket editability",
        "environment":{"blender":bpy.app.version_string,"python":sys.version,"numpy":np.__version__}}
    started=time.monotonic()
    owner=OwnedRun(args.output.resolve(),producer="structured_family_source_edit",max_generated_bytes=16777216,
                   shared_inputs={"saved_receipt":str(args.saved_receipt.resolve())})
    with owner:
        _write(owner.root/"frozen-workload.json",frozen);owner.register_file("frozen-workload.json","diagnostic")
        receipt={"protocol":frozen["protocol"],"status":"running","cases":{},"aggregate_accepted":False}
        _publish_receipt(owner,receipt)
        for name,item in inputs.items():
            print("structured-source-edit "+name,flush=True)
            folder=owner.root/name;folder.mkdir();compiled=temporary=None
            try:
                if time.monotonic()-started>60:
                    raise TimeoutError("structured semantic deadline exceeded")
                with np.load(Path(item["directory"])/"evaluated-exact.npz",allow_pickle=False) as archive:
                    captured=GeometryArrays.capture(archive["vertices"],archive["faces"])
                if captured.content_hash!=item["required_indexed_hash"]:
                    raise ValueError("saved geometry identity changed")
                compiled,source,before=_compile_exact(item["original_recipe"],captured)
                mesh,pose=source.data,_pose_controls(source)
                root_pointer,source_pointer=compiled.root_object.as_pointer(),source.as_pointer()
                baseline=observe(name,before,source)
                _write(folder/"original-program.json",item["original_recipe"])
                _write(folder/"edited-program.json",item["edited_recipe"])
                tag="structured_source_edit_recipe_json";old_tag=source.get(tag)
                try:
                    temporary=compile_shape_program(retained_family_program(item["edited_recipe"]),lathe_segments=96,weighted_normals=False)
                    children=output_mesh_targets([temporary.root_object])
                    if len(children)!=1:
                        raise ValueError("edited recipe must retain one source mesh")
                    source.data=children[0].data;_apply_pose(source,_pose_controls(children[0]))
                    source[tag]=json.dumps(item["edited_recipe"],sort_keys=True)
                    bpy.context.view_layer.update()
                    changed=evaluated_arrays(compiled.root_object)
                    edited=observe(name,changed,source)
                    verdict=response(name,baseline,edited)
                    save_mesh(compiled.root_object,folder)
                finally:
                    source.data=mesh;_apply_pose(source,pose)
                    if old_tag is None:
                        if tag in source:del source[tag]
                    else:source[tag]=old_tag
                    bpy.context.view_layer.update()
                restored=evaluated_arrays(compiled.root_object)
                guards={"same_root_pointer":compiled.root_object.as_pointer()==root_pointer,
                    "same_source_pointer":source.as_pointer()==source_pointer,"original_mesh_pointer_restored":source.data==mesh,
                    "native_pose_restored":_pose_controls(source)==pose,"indexed_restoration":restored.content_hash==captured.content_hash,
                    "geometry_changed":changed.content_hash!=before.content_hash}
                passed=verdict["passed"] and all(guards.values())
                receipt["cases"][name]={"status":"passed" if passed else "failed","control":item["control"],
                    "scope":frozen["scope"],"baseline_geometry_hash":before.content_hash,"edited_geometry_hash":changed.content_hash,
                    "restored_geometry_hash":restored.content_hash,"guards":guards,"response":verdict,"before":baseline,"after":edited,
                    "family_surface_acceptance":"independently unqualified"}
            except Exception as exc:
                receipt["cases"][name]={"status":"failed","reason":type(exc).__name__+": "+str(exc)}
            finally:
                if temporary is not None:_cleanup(temporary)
                if compiled is not None:_cleanup(compiled)
            _publish_receipt(owner,receipt)
            for path in folder.iterdir():
                if path.is_file():owner.register_file(path.relative_to(owner.root),"final_output")
        passed=all(r["status"]=="passed" for r in receipt["cases"].values())
        receipt.update(status="passed" if passed else "failed",elapsed_seconds=time.monotonic()-started,run_root=str(owner.root))
        _publish_receipt(owner,receipt)
        print("STRUCTURED_EDIT_RESULT="+str(owner.root/"results.json"),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
