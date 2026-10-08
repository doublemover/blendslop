#!/usr/bin/env python3
"""One bounded mask-driven triangular-dot proposal; independent raw verdicts."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"blender_blocking"),str(ROOT),str(ROOT/"scripts")]
import test_runner
from run_surface_quality_check import _write
from run_quality_coverage_check import save_mesh, render_masks


def main():
    import bpy,numpy as np
    from PIL import Image
    from mathutils import Vector
    from reconstruction.types import Bounds2D, OrthographicCameraSpec, ViewConstraint, ReconstructionTarget
    from reconstruction.projection_contract import bounds_from_calibrated_masks
    from reconstruction.program_proposals import whole_primitive_programs
    from reconstruction.native_geometry import GeometryArrays,evaluated_arrays
    from reconstruction.output_qualification import qualify_retained_output
    from primitives.shape_program import ShapeProgram
    from primitives.shape_program_compiler import compile_shape_program
    from evaluation.surface_quality import compare_surface_arrays
    from evaluation.silhouette_eval import SilhouetteGateConfig,evaluate_silhouette_pair
    from synthetic.quality_contracts import triangle_preservation
    from utils.run_ownership import OwnedRun
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--qualification-python",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    parent=args.output.resolve()
    if not parent.is_relative_to(ROOT/"temp/tasks"):raise ValueError("output must stay below repository temp/tasks")
    owner=OwnedRun(parent,producer="triangle_support_reconstruction",max_generated_bytes=33554432,
                   shared_inputs={"reference_directory":str(args.reference.resolve()),"qualification_python":str(args.qualification_python)})
    with owner:
        output=owner.root
        _write(output/"frozen-workload.json",{"source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "working_diff_sha256":hashlib.sha256(subprocess.check_output(["git","diff","HEAD"],cwd=ROOT)).hexdigest(),
              "scope":"one authored-template triangular family fitted to front/side/top observed mask supports",
              "fit":{"evaluations_per_seed":160,"total_seconds":3.,"triangle_quarter_turn_seeds":4},
              "silhouette":{"area_iou_min":.7,"boundary_iou_min":.8,"signed_distance_loss_max":.05},
              "triangle":{"support_error_max_world":.005,"thickness_error_max_world":.005},
              "surface_acceptance":"blocked until independent triangle reference noise qualification; raw distances only",
              "helper_timeout_seconds":15.,"environment":{"blender":bpy.app.version_string,"python":sys.version,"numpy":np.__version__}})
        started=time.monotonic()
        previous=json.loads((args.reference.parent/"results.json").read_text())
        cameras=previous["cases"]["rounded_triangle_dot"]["reference_cameras"]
        masks={v:np.asarray(Image.open(args.reference/(v+"-mask.png")).convert("L"))<128 for v in ("front","side","top")}
        constraints=[];records={};boxes={}
        for view,mask in masks.items():
            coords=np.argwhere(mask);lo,hi=coords.min(axis=0),coords.max(axis=0)+1
            boxes[view]=Bounds2D(float(lo[1]),float(lo[0]),float(hi[1]),float(hi[0]))
            center=np.asarray(cameras[view]["matrix_world"])[:3,3];scale=cameras[view]["ortho_scale"]
            axes={"front":(0,2),"side":(1,2),"top":(0,1)}[view];u,v=center[list(axes)]
            viewport=(u-scale/2,u+scale/2,v-scale/2,v+scale/2)
            records[view]={"world_bounds":viewport,"world_units":"metres"}
            camera=OrthographicCameraSpec(view,{"front":"y","side":"x","top":"z"}[view],
                                          bounds=Bounds2D(viewport[0],viewport[2],viewport[1],viewport[3]))
            constraints.append(ViewConstraint(view,mask,camera,bbox=boxes[view]))
        target=ReconstructionTarget(tuple(constraints),bounds=bounds_from_calibrated_masks(masks,boxes,records),extras={"view_calibration":records})
        programs=whole_primitive_programs(target,ShapeProgram("1","triangle-support",()),max_evaluations=160,max_elapsed_s=3.)
        program=next(p for p in programs if p.root_nodes[0].primitive_type=="rounded_triangle")
        _write(output/"program.json",program.to_dict())
        compiled=compile_shape_program(program,weighted_normals=False)
        obj=compiled.root_object;data,hashes=save_mesh(obj,output)
        old=np.load(args.reference/"evaluated-exact.npz",allow_pickle=False)
        reference=GeometryArrays.capture(old["vertices"],old["faces"])
        views=["front","side","top","oblique_35_28","oblique_145_40"]
        paths,cameras=render_masks(obj,output,(Vector(reference.vertices.min(axis=0)),Vector(reference.vertices.max(axis=0))),views,camera_records=cameras)
        gates=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
        silhouettes={v:evaluate_silhouette_pair(np.asarray(Image.open(args.reference/(v+"-mask.png")).convert("L"))<128,paths[v],view=v,config=gates) for v in views}
        observation=compare_surface_arrays(reference,data)
        observation.pop("surface_passed");observation.pop("frozen_limits")
        observation["surface_status"]="blocked: independent reference tessellation/noise qualification absent"
        boundary=qualify_retained_output(data,{"native_qualification_python":str(args.qualification_python),"native_qualification_timeout_s":15.})
        artist=obj.scale.copy();obj.scale.x*=1.05;bpy.context.view_layer.update()
        changed=evaluated_arrays(obj);obj.scale=artist;bpy.context.view_layer.update()
        edit={"object_scale_changed":changed.content_hash!=data.content_hash,"exact_restoration":evaluated_arrays(obj).content_hash==data.content_hash,
              "scope":"live object edit; semantic corner/depth controls covered separately by native compiler tests"}
        receipt={**hashes,"elapsed_seconds":time.monotonic()-started,"program":program.to_dict(),"surface_observation":observation,
                 "silhouette":silhouettes,"boundary":boundary,"triangle_preservation":triangle_preservation(data.vertices),"editability":edit,
                 "aggregate_accepted":False,"blocker":"surface reference qualification remains independent of support/mask/topology success"}
        _write(output/"results.json",receipt)
        for name in ("frozen-workload.json","program.json","evaluated.obj","evaluated-exact.npz","results.json",*(v+"-mask.png" for v in views)):
            owner.register_file(name,"final_output" if name in {"program.json","evaluated.obj","evaluated-exact.npz","results.json"} else "diagnostic")
        print("TRIANGLE_RESULT="+str(output/"results.json"),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
