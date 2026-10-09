#!/usr/bin/env python3
"""One actual calibrated retained-vase reconstruction and historical-input repair."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"blender_blocking"),str(ROOT),str(ROOT/"scripts")]
import test_runner
from run_surface_quality_check import _write, _renders
from run_quality_coverage_check import save_mesh


def main():
    import bpy,numpy as np
    from PIL import Image
    from mathutils import Vector
    from dataclasses import replace
    from config import BlockingConfig
    from main_integration import BlockingWorkflow
    from reconstruction.types import Bounds2D, OrthographicCameraSpec, ViewConstraint, ReconstructionTarget, CandidateRequest
    from reconstruction.projection_contract import bounds_from_calibrated_masks
    from reconstruction.backends.profile_loft import ProfileLoftBackend
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.output_qualification import qualify_retained_output
    from evaluation.surface_quality import compare_surface_arrays
    from evaluation.canonical_artifacts import canonical_artifact_inventory
    from evaluation.silhouette_eval import SilhouetteGateConfig, evaluate_silhouette_pair
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--historical-input",type=Path,required=True)
    parser.add_argument("--qualification-python",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--coverage-transfer", type=Path,
                        help="Declared 16-bit grayscale LUT with adjacent transfer-settings.json")
    parser.add_argument("--skip-historical",action="store_true",help="Do not repeat unchanged historical repair")
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    config={"num_samples":129,"num_slices":65,"unit_scale":.01,"radial_segments":96,
            "surface_mode":"smooth","surface_subdivisions":2,"regularization_window":9,
            "regularization_max_deviation_u":.003,"cap_mode":"fan","smoothing_window":3}
    workload={"scope":"one changed calibrated smooth-vase reconstruction" + ("" if args.skip_historical else " plus historical-input connected repair"),
              "source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "working_source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [ROOT/"scripts/run_profile_surface_repair.py", ROOT/"scripts/run_surface_quality_check.py",
                      ROOT/"blender_blocking/evaluation/canonical_artifacts.py",
                      *[ROOT/"blender_blocking/reconstruction"/(name+".py") for name in
                        ["coverage_evidence", "profile_evidence", "projection_contract", "target_builder", "evidence_identity"]]]},
              "config":config,"historical_config":{"num_slices":129,"radial_segments":96,"subdivisions":2,
                    "regularization_window":9,"regularization_section_deviation_world_max":.06},
              "surface_limits":{"mean_world":.003,"normal_p95_degrees":2.5,"scope":"retained analytic vase only"},
              "silhouette_limits":{"area_iou_min":.7,"boundary_iou_min":.8,"signed_distance_loss_max":.05},
              "limits":{"seconds":300,"rss_bytes":8589934592},
              "reference":str(args.reference.resolve()),"historical_inputs":str(args.historical_input.resolve()),
              "environment":{"blender":bpy.app.version_string,"python":sys.version,"numpy":np.__version__}}
    if args.coverage_transfer:
        settings_path = args.coverage_transfer.with_name("transfer-settings.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        current = {key: getattr(bpy.context.scene.view_settings, key) for key in settings}
        if settings != current:
            raise ValueError("declared coverage transfer differs from current display settings")
        workload["coverage_transfer"] = {
            "lut_sha256": hashlib.sha256(args.coverage_transfer.read_bytes()).hexdigest(),
            "settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
            "settings": settings, "scope": "black/white linear coverage; declared display transfer"}
    workload["reference_sha256"] = {name: hashlib.sha256((args.reference/name).read_bytes()).hexdigest()
        for name in ["evaluated-exact.npz"] + [view+"-mask.png" for view in
                    ["front", "side", "top", "oblique_35_28", "oblique_145_40"]]}
    _write(output/"frozen-workload.json",workload)
    before=time.monotonic()
    previous=json.loads((args.reference.parent/"results.json").read_text())
    cameras=previous["cases"]["smooth_vase"]["reference_cameras"]
    masks={v:np.asarray(Image.open(args.reference/(v+"-mask.png")).convert("L"))<128 for v in ("front","side","top")}
    coverage = {}
    for view, record in cameras.items():
        if view+"-mask.png" in workload["reference_sha256"]:
            if workload["reference_sha256"][view+"-mask.png"] != record["png_sha256"]:
                raise ValueError("retained reference mask hash differs from frozen camera receipt")
    if args.coverage_transfer:
        from reconstruction.coverage_evidence import coverage_from_grayscale
        encoded = np.asarray(Image.open(args.coverage_transfer), float).ravel() / 65535
        linear = np.linspace(0., 1., len(encoded))
        coverage = {view: coverage_from_grayscale(
            np.asarray(Image.open(args.reference/(view+"-mask.png")).convert("L"), float)/255,
            linear, encoded) for view in masks}
    constraints=[];records={};boxes={}
    for view,mask in masks.items():
        coords=np.argwhere(mask);lo,hi=coords.min(axis=0),coords.max(axis=0)+1
        boxes[view]=Bounds2D(float(lo[1]),float(lo[0]),float(hi[1]),float(hi[0]))
        matrix=np.asarray(cameras[view]["matrix_world"]);center=matrix[:3,3];scale=cameras[view]["ortho_scale"]
        axes={"front":(0,2),"side":(1,2),"top":(0,1)}[view]
        u,v=center[list(axes)];bounds=(u-scale/2,u+scale/2,v-scale/2,v+scale/2)
        records[view]={"world_bounds":bounds,"world_units":"metres"}
        camera=OrthographicCameraSpec(view,{"front":"y","side":"x","top":"z"}[view],
                                     bounds=Bounds2D(bounds[0],bounds[2],bounds[1],bounds[3]))
        constraints.append(ViewConstraint(view,mask,camera,bbox=boxes[view],coverage_mask=coverage.get(view)))
    bounds=bounds_from_calibrated_masks(masks,boxes,records,coverage_masks=coverage)
    target=ReconstructionTarget(tuple(constraints),bounds=bounds,extras={"view_calibration":records})
    request=CandidateRequest("retained-smooth-vase","profile_loft",target,config,context=SimpleNamespace(blender_available=True))
    result=ProfileLoftBackend().reconstruct(request)
    receipt={"reconstruction_status":result.status,"errors":result.errors,
             "coverage_views": sorted(coverage), "inferred_bounds": bounds.to_dict()}
    if result.status!="success":_write(output/"results.json",receipt);return 1
    obj=result.payload;data,hashes=save_mesh(obj,output/"calibrated-vase")
    old=np.load(args.reference/"evaluated-exact.npz")
    reference=GeometryArrays.capture(old["vertices"],old["faces"])
    fixed_bounds=(Vector(reference.vertices.min(axis=0)),Vector(reference.vertices.max(axis=0)))
    captured_cameras={}
    paths=_renders(obj,output/"calibrated-vase",fixed_bounds,["front","side","top","oblique_35_28","oblique_145_40"],camera_records=cameras,captured_cameras=captured_cameras)
    inspection=canonical_artifact_inventory(output/"calibrated-vase",geometry_hash=data.content_hash,
        camera_records=captured_cameras,reference_camera_records=cameras,
        pass_states={name:"completed" for name in ("mask","neutral","normals")})
    _write(output/"calibrated-vase/canonical-inspection.json",inspection)
    surface=compare_surface_arrays(reference,data)
    qualification=qualify_retained_output(data,{"native_qualification_python":str(args.qualification_python),"native_qualification_timeout_s":15.})
    gates=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
    silhouette={view:evaluate_silhouette_pair(np.asarray(Image.open(args.reference/(view+"-mask.png")).convert("L"))<128,
             np.asarray(Image.open(path).convert("L"))<128,view=view,config=gates) for view,path in paths.items()}
    from integration.blender_ops.profile_loft_mesh import rebuild_loft_mesh
    from geometry.profile_models import EllipticalSlice
    recipe_json = obj["loft_recipe_json"]
    (output/"calibrated-vase/loft-recipe.json").write_text(recipe_json+"\n", encoding="utf-8")
    sections = [EllipticalSlice(**row) for row in json.loads(recipe_json)["slices"]]
    pointer, transform = obj.as_pointer(), obj.matrix_world.copy()
    rebuild_loft_mesh(obj, slices=[replace(section, rx=section.rx*1.05) for section in sections])
    edited = evaluated_arrays(obj)
    rebuild_loft_mesh(obj, slices=sections)
    restored = evaluated_arrays(obj)
    editability = {"semantic_x_radius_changed": edited.content_hash != data.content_hash,
        "same_object": obj.as_pointer() == pointer, "same_transform": obj.matrix_world == transform,
        "exact_restoration": restored.content_hash == data.content_hash,
        "scope": "actual fitted loft source-section radius edit and exact regeneration"}
    editability["passed"] = all(editability[key] for key in
        ["semantic_x_radius_changed", "same_object", "same_transform", "exact_restoration"])
    receipt["calibrated_vase"]={**hashes,"canonical_inspection":inspection,"surface":surface,"boundary":qualification,"silhouette":silhouette,"editability":editability,
              "accepted":surface["surface_passed"] and qualification["boundary_qualified"] and
                  editability["passed"] and all(row["passed"] for row in silhouette.values())}
    _write(output/"results.json",receipt)
    if args.skip_historical:
        receipt["elapsed_seconds"]=time.monotonic()-before
        _write(output/"results.json",receipt)
        return 0 if receipt["calibrated_vase"]["accepted"] else 1
    cfg=BlockingConfig();cfg.reconstruction.legacy_profile_geometry="connected";cfg.reconstruction.num_slices=129
    cfg.profile_sampling.num_samples=129;cfg.mesh_from_profile.radial_segments=96;cfg.mesh_from_profile.surface_subdivisions=2
    cfg.mesh_from_profile.regularization_window=9;cfg.mesh_from_profile.regularization_max_deviation_u=.06
    workflow=BlockingWorkflow(config=cfg)
    workflow.views={view:np.asarray(Image.open(args.historical_input/("vase_"+view+".png")).convert("RGB")) for view in ("front","side","top")}
    obj=workflow.create_3d_blockout();data,hashes=save_mesh(obj,output/"historical-vase")
    historical_cameras={}
    _renders(obj,output/"historical-vase",(Vector(data.vertices.min(axis=0)),Vector(data.vertices.max(axis=0))),["front","side","top","oblique_35_28","oblique_145_40"],captured_cameras=historical_cameras)
    historical_inspection=canonical_artifact_inventory(output/"historical-vase",geometry_hash=data.content_hash,
        camera_records=historical_cameras,pass_states={name:"completed" for name in ("mask","neutral","normals")})
    _write(output/"historical-vase/canonical-inspection.json",historical_inspection)
    receipt["historical_vase"]={**hashes,"canonical_inspection":historical_inspection,"boundary":qualify_retained_output(data,{"native_qualification_python":str(args.qualification_python),"native_qualification_timeout_s":15.}),
        "recipe":json.loads(obj["loft_recipe_json"]),"surface_acceptance":"blocked: source ellipse-strip scallops do not define an authored smooth reference",
        "input_sha256":{v:hashlib.sha256((args.historical_input/("vase_"+v+".png")).read_bytes()).hexdigest() for v in ("front","side","top")}}
    receipt["elapsed_seconds"]=time.monotonic()-before
    _write(output/"results.json",receipt)
    return 0 if receipt["calibrated_vase"]["accepted"] else 1


if __name__=="__main__":raise SystemExit(main())
