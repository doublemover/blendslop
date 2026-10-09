#!/usr/bin/env python3
"""Separate arch exterior checkpoint; controlled acquisition, bounded local fit.

This standalone experiment does not replace current five-view selected rows.
The plan declares every frame, parameter interval and held-out exposure before
launch. Missing display/canonical/boundary verdicts remain unrun explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'blender_blocking'), str(ROOT/'scripts'), str(ROOT)]
import test_runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded_json(value):
    """Retain numeric scalar types without weakening finite-value validation."""
    def native_scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('unsupported receipt value: '+type(item).__name__)
    return json.dumps(value,indent=2,allow_nan=False,default=native_scalar)+'\n'


def failure_receipt(primary, protocol, cases, frames):
    """An invalid case transport cannot erase the primary operation failure."""
    value={'protocol':protocol,'status':'failed','cases':cases,'rendered_frames':frames,
           'reason':type(primary).__name__+': '+str(primary),'aggregate_accepted':False}
    try:
        encoded_json(value)
    except (TypeError,ValueError) as secondary:
        value['cases']={}
        value['completed_case_keys']=list(cases)
        value['case_transport_error']=type(secondary).__name__+': '+str(secondary)
    return value


def verify_refined_baseline(item, wire):
    """Verify the real completed adaptive owner and refined receipt, never adopt it."""
    from reconstruction.native_geometry import GeometryArrays
    import numpy as np
    directory=Path(item['baseline_directory']).resolve()
    owner_root=directory.parent.parent
    if (directory.name!='refined' or directory.parent.name!='concave_arch'
            or Path(item['baseline_receipt']).resolve()!=owner_root/'results.json'):
        raise ValueError('baseline must be the original refined arch owner result')
    receipt=read_json(item['baseline_receipt']);bound=receipt['cases']['concave_arch']
    if (receipt['status']!='measured' or bound['status']!='measured'
            or bound['program']!=wire or bound['refined_geometry_hash']!=item['baseline_geometry_hash']
            or bound['refined_raw_surface']!=item['baseline_raw_surface']):
        raise ValueError('refined recipe/geometry/raw fields differ from their original receipt')
    manifest=read_json(owner_root/'run-ownership.json');lease=read_json(owner_root/'run-lease.json')
    if (manifest['state']!='succeeded' or lease['status']!='released'
            or manifest['run_root']!=str(owner_root)
            or manifest['run_id']!=lease['run_id'] or manifest['owner_token']!=lease['owner_token']):
        raise ValueError('refined baseline owner is not completed/released and identity-bound')
    artifacts={row['path']:row for row in manifest['artifacts']}
    verified={}
    for relative in ('results.json','concave_arch/refined/program.json',
                     'concave_arch/refined/evaluated-exact.npz','concave_arch/refined/evaluated.obj'):
        digest=sha(owner_root/relative)
        if relative not in artifacts or digest!=artifacts[relative]['sha256']:
            raise ValueError('original refined owned artifact bytes changed: '+relative)
        verified[relative]=digest
    if sha(directory/'program.json')!=item['program_sha256']:
        raise ValueError('retained refined recipe bytes changed')
    with np.load(directory/'evaluated-exact.npz',allow_pickle=False) as archive:
        captured=GeometryArrays.capture(archive['vertices'],archive['faces'])
    raw=item['baseline_raw_surface']
    if (captured.content_hash!=item['baseline_geometry_hash']
            or raw['candidate_geometry_hash']!=captured.content_hash
            or raw['reference_geometry_hash']!=item['source_geometry_hash']
            or raw['sample_count_per_direction']!=4096 or raw['seed']!=61007):
        raise ValueError('original refined archive/raw metric identities differ')
    return {'owner_root':str(owner_root),'state':'succeeded','lease':'released',
            'manifest_sha256':sha(owner_root/'run-ownership.json'),
            'lease_sha256':sha(owner_root/'run-lease.json'),'verified_artifacts':verified,
            'geometry_hash':captured.content_hash,'historical_adoption':False}


def depth_response(before, changed, matrix_world, multiplier=1.05):
    """Native local physical depth response; every XY vertex stays unchanged."""
    import numpy as np
    matrix=np.asarray(matrix_world,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not isinstance(multiplier,(int,float)) or isinstance(multiplier,bool)
            or not np.isfinite(multiplier) or multiplier<=1.):
        raise ValueError('physical depth response requires finite pose and multiplier>1')
    def local(arrays):
        return (np.column_stack((arrays.vertices,np.ones(len(arrays.vertices))))@np.linalg.inv(matrix).T)[:,:3]
    a,b=local(before),local(changed)
    same_shape=a.shape==b.shape
    old_depth=float(np.ptp(a[:,2]));new_depth=float(np.ptp(b[:,2]))
    checks={'fixed_all_outline_vertices_xy':bool(same_shape and np.allclose(a[:,:2],b[:,:2],atol=1e-5,rtol=0.)),
            'unchanged_triangle_inventory':bool(np.array_equal(before.faces,changed.faces)),
            'depth_response':bool(old_depth>0. and abs(new_depth-old_depth*multiplier)<=1e-5)}
    return {'control':'extrusion_depth_world','expected_multiplier':multiplier,
            'before_depth_world':old_depth,'after_depth_world':new_depth,
            'opening_and_notch_height_fixed':'all local outline vertices verified unchanged',
            'checks':checks,'passed':bool(all(checks.values()))}


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import raw_surface_observation
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.reference_noise import oriented_surface_identity
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from primitives.shape_program_compiler import compile_shape_program
    from reconstruction.adaptive_family import adaptive_crop_request
    from reconstruction.adaptive_arch_exterior import arch_exterior_controls, arch_exterior_update, refine_arch_exterior
    from reconstruction.frozen_family import retained_family_program
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _pose_controls, _apply_pose, _compile_exact
    from run_quality_coverage_check import save_mesh
    from run_surface_quality_check import _replay_orthographic_camera

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    plan=read_json(args.plan)
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('new adaptive checkpoint must remain in isolated temp/tasks')
    if (plan.get('protocol')!='bounded_arch_exterior_checkpoint_v1' or len(plan['cases'])!=1 or plan['cases'][0]['family']!='concave_arch'
            or plan['deadline_seconds']!=85 or plan['threads']!=2 or plan['memory_limit_bytes']!=8*1024**3
            or plan['segments']!=96 or plan['qualification_children']!=0
            or plan['fit_budget']!={'residual_calls':96,'seconds':1.}
            or plan['global_resolution']!=[512,512] or plan['crop_resolution']!=[1024,1024]
            or plan['crop_pixel_span']!=160 or plan['rendered_frames']!=6
            or plan['heldout_views']!=['oblique_145_40']
            or plan['cases'][0]['fit_view']!='oblique_35_28_expanded'
            or plan['cases'][0]['semantic_control']!='extrusion_depth_world'
            or plan['cases'][0]['semantic_multiplier']!=1.05):
        raise ValueError('only the frozen one-case six-frame exterior checkpoint is supported')
    for filename,digest in plan['input_sha256'].items():
        if sha(filename)!=digest:raise ValueError('frozen input changed: '+filename)
    frames_per_case=3+3*len(plan['heldout_views'])
    expected=frames_per_case
    if any(item.get('reuse_completed_case') for item in plan['cases']):
        raise ValueError('this new acquisition stage cannot reuse partial measurements')
    if plan['rendered_frames']!=expected:
        raise ValueError('declared acquisition count differs from the exact loop')
    if len(set(plan['heldout_views']))!=len(plan['heldout_views']) or not plan['heldout_views']:
        raise ValueError('declare distinct independent validation views')
    owner=OwnedRun(args.output.resolve(),producer='bounded_arch_exterior_checkpoint',
                   max_generated_bytes=134217728,shared_inputs={'plan':str(args.plan.resolve())})
    started=time.monotonic();frames=0;cases={}
    gates=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
    with owner:
        def publish(relative,value):
            path=owner.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(encoded_json(value),encoding='utf-8')
            owner.register_file(relative,'final_output')
        def save(obj,relative):
            folder=owner.root/relative
            arrays,files=save_mesh(obj,folder)
            for path in folder.iterdir():
                if path.is_file():owner.register_file(str(path.relative_to(owner.root)),'final_output')
            return arrays,files
        def deadline():
            if time.monotonic()-started>plan['deadline_seconds']:
                raise TimeoutError('fixed adaptive family deadline')
        bpy.ops.object.camera_add();camera=bpy.context.object
        camera.data.type='ORTHO';bpy.context.scene.camera=camera
        def measure(obj,declaration,label,resolution,expected):
            nonlocal frames
            deadline();folder=owner.root/label;folder.mkdir(parents=True)
            with controlled_measurement_session(target_objects=[obj],camera=camera,
                    resolution=tuple(resolution),samples=64,filter_size=1.5) as session:
                # Every acquisition replays the same declared camera, never a
                # second native snapshot decomposition for comparison passes.
                _replay_orthographic_camera(camera,declaration)
                row=render_controlled_measurement(session,folder/'alpha.exr')
            frames+=1
            if row['geometry_hashes']!=[expected] or not row['geometry_unchanged']:
                raise ValueError('acquisition lost stable exact geometry binding')
            np.save(folder/'coverage.npy',row['coverage'],allow_pickle=False)
            Image.fromarray(np.rint(row['coverage']*255).astype(np.uint8)).save(folder/'coverage.png')
            publish(label+'/measurement.json',{key:value for key,value in row.items() if key not in ('coverage','mask')})
            for name in ('alpha.exr','coverage.npy','coverage.png'):
                owner.register_file(str((folder/name).relative_to(owner.root)),'final_output')
            return row
        publish('frozen-workload.json',{**plan,'plan_path':str(args.plan.resolve()),'plan_sha256':sha(args.plan),
            'environment':{'blender':bpy.app.version_string,'python':sys.version},
            'full_required_five_view_admission':'unrun for changed candidate; current selection preserved',
            'artist_surface_limits':None})
        try:
            for item in plan['cases']:
                deadline();family=item['family'];print('adaptive family '+family,flush=True)
                directory=Path(item['baseline_directory']);wire=read_json(directory/'program.json')
                audit=verify_refined_baseline(item,wire)
                publish(family+'/baseline-owner-audit.json',audit)
                with np.load(directory/'evaluated-exact.npz',allow_pickle=False) as archive:
                    retained=GeometryArrays.capture(archive['vertices'],archive['faces'])
                if retained.content_hash!=item['baseline_geometry_hash']:
                    raise ValueError('retained baseline archive differs from its frozen identity')
                # The existing UV adapter restores face order only after exact
                # world vertices and complete oriented triangle inventory agree.
                # This is not a tolerant geometry/fit fallback; every final
                # baseline still has the unchanged retained indexed identity.
                baseline,baseline_source,baseline_arrays=_compile_exact(wire,retained)
                if baseline_arrays.content_hash!=retained.content_hash:
                    raise ValueError('retained indexed baseline replay differs')
                publish(family+'/baseline/program.json',wire)
                save(baseline.root_object,family+'/baseline')
                # Authored case dimensions are source-acquisition inputs only.
                # The local fitter receives coverage/camera and the saved recipe.
                reference=build_quality_reference(item['source_case']).object
                source_arrays=evaluated_arrays(reference)
                with np.load(item['source_archive'],allow_pickle=False) as archive:
                    retained_reference=GeometryArrays.capture(archive['vertices'],archive['faces'])
                if (retained_reference.content_hash!=item['source_geometry_hash']
                        or oriented_surface_identity(source_arrays)!=oriented_surface_identity(retained_reference)):
                    save(reference,family+'/source-guard-failed')
                    raise ValueError('authored source exact oriented identity differs')
                fitting=item['fit_view'];decl=item['cameras'][fitting]
                source_fit=measure(reference,decl,family+'/source-fit',[512,512],source_arrays.content_hash)
                base_fit=measure(baseline.root_object,decl,family+'/baseline-fit',[512,512],baseline_arrays.content_hash)
                request=adaptive_crop_request(source_fit['coverage'],base_fit['coverage'],
                    source_fit['contract']['camera'],fit_view=fitting,heldout_views=plan['heldout_views'],
                    prior_view_exposure=item['prior_view_exposure'],pixel_span=160,source_geometry_available=True)
                publish(family+'/crop-request.json',request)
                if request['status']!='requested':raise ValueError('no fitting residual requires new detail')
                crop=measure(reference,request['camera'],family+'/source-detail',[1024,1024],source_arrays.content_hash)
                observations={fitting:{'coverage':source_fit['coverage'],'camera':source_fit['contract']['camera']},
                              fitting+'_detail':{'coverage':crop['coverage'],'camera':crop['contract']['camera']}}
                refined_program=refine_arch_exterior(wire,observations,
                    parameter_bounds=item['parameter_bounds'],heldout_views=plan['heldout_views'],
                    prior_view_exposure=item['prior_view_exposure'],max_evaluations=96,max_elapsed_s=1.,
                    baseline_geometry_hash=item['baseline_geometry_hash'],
                    baseline_program_sha256=item['program_sha256'])
                # Recipe is frozen before any new raw reference proximity comparison.
                publish(family+'/refined/program.json',refined_program.to_dict())
                refined=compile_shape_program(refined_program,lathe_segments=96,weighted_normals=False)
                refined_arrays,_=save(refined.root_object,family+'/refined')
                baseline_metrics={};refined_metrics={}
                for view in plan['heldout_views']:
                    decl=item['cameras'][view]
                    source=measure(reference,decl,family+'/source-heldout/'+view,[512,512],source_arrays.content_hash)
                    old=measure(baseline.root_object,decl,family+'/baseline-heldout/'+view,[512,512],baseline_arrays.content_hash)
                    new=measure(refined.root_object,decl,family+'/refined-heldout/'+view,[512,512],refined_arrays.content_hash)
                    baseline_metrics[view]=compare_controlled_measurements(source,old,view=view,config=gates)
                    refined_metrics[view]=compare_controlled_measurements(source,new,view=view,config=gates)
                old_raw=item['baseline_raw_surface']
                if (old_raw['sample_count_per_direction']!=4096 or old_raw['seed']!=61007
                        or old_raw['reference_geometry_hash']!=retained_reference.content_hash
                        or old_raw['candidate_geometry_hash']!=baseline_arrays.content_hash):
                    raise ValueError('baseline raw receipt cannot be reused under unchanged protocol')
                raw=raw_surface_observation(retained_reference,refined_arrays,count=4096,seed=61007)
                controls=arch_exterior_controls(refined_program.to_dict())
                control=item['semantic_control'];edited_wire=arch_exterior_update(refined_program.to_dict(),
                    {control:controls[control]*item['semantic_multiplier']}).to_dict()
                publish(family+'/semantic-edited-program.json',edited_wire)
                edited=compile_shape_program(retained_family_program(edited_wire),lathe_segments=96,weighted_normals=False)
                sources=output_mesh_targets([refined.root_object]);temporary=output_mesh_targets([edited.root_object])
                if len(sources)!=1 or len(temporary)!=1:raise ValueError('semantic edit requires one native source')
                mesh,pose=sources[0].data,_pose_controls(sources[0]);pointer=sources[0].as_pointer()
                before=refined_arrays
                try:
                    sources[0].data=temporary[0].data;_apply_pose(sources[0],_pose_controls(temporary[0]))
                    bpy.context.view_layer.update();changed=evaluated_arrays(refined.root_object)
                    response=depth_response(before,changed,np.asarray(sources[0].matrix_world),item['semantic_multiplier'])
                finally:
                    sources[0].data=mesh;_apply_pose(sources[0],pose);bpy.context.view_layer.update()
                restored=evaluated_arrays(refined.root_object)
                transaction={'control':control+' x'+str(item['semantic_multiplier']),'response':response,
                    'geometry_changed':changed.content_hash!=before.content_hash,
                    'same_source_pointer':sources[0].as_pointer()==pointer,'original_mesh_restored':sources[0].data==mesh,
                    'exact_indexed_restoration':restored.content_hash==before.content_hash}
                transaction['passed']=bool(response['passed'] and all(transaction[k] for k in
                    ('geometry_changed','same_source_pointer','original_mesh_restored','exact_indexed_restoration')))
                old_l1=float(np.mean([v['coverage_l1'] for v in baseline_metrics.values()]))
                new_l1=float(np.mean([v['coverage_l1'] for v in refined_metrics.values()]))
                checkpoint=bool(all(v['passed'] for v in refined_metrics.values()) and new_l1<old_l1
                    and min(v['boundary_iou'] for v in refined_metrics.values())>=min(v['boundary_iou'] for v in baseline_metrics.values())
                    and raw['symmetric_mean_distance_world']<=old_raw['symmetric_mean_distance_world'] and transaction['passed']
                    and refined_program.metadata['adaptive_arch_exterior']['identifiability']=='locally_identified')
                stable=evaluated_arrays(baseline.root_object).content_hash==baseline_arrays.content_hash
                stable=stable and evaluated_arrays(reference).content_hash==source_arrays.content_hash
                if not stable:raise ValueError('experiment mutated frozen baseline/source geometry')
                cases[family]={'status':'measured','baseline_geometry_hash':baseline_arrays.content_hash,
                    'refined_geometry_hash':refined_arrays.content_hash,'program':refined_program.to_dict(),
                    'baseline_heldout':baseline_metrics,'refined_heldout':refined_metrics,
                    'baseline_raw_surface_reused':old_raw,'refined_raw_surface':raw,
                    'baseline_mean_heldout_coverage_l1':old_l1,'refined_mean_heldout_coverage_l1':new_l1,
                    'relative_heldout_coverage_l1_reduction':(old_l1-new_l1)/old_l1 if old_l1 else 0.,
                    'semantic_edit_restoration':transaction,'bounded_checkpoint_improved':checkpoint,
                    'source_original_geometry_hash':retained_reference.content_hash,'source_replayed_geometry_hash':source_arrays.content_hash,
                    'source_exact_oriented_equivalence':True,'fixed_inputs_geometry_stable':stable,
                    'artist_surface_limits':None,'full_required_five_view_admission':'unrun for refined geometry',
                    'native_boundary_qualification':'unrun','canonical_neutral_normal_passes':'unrun',
                    'selected_current_row_unchanged':True,'aggregate_accepted':False}
                publish('results.json',{'protocol':plan['protocol'],'status':'running','cases':cases,'rendered_frames':frames})
            publish('results.json',{'protocol':plan['protocol'],'status':'measured','cases':cases,
                'rendered_frames':frames,'source_frames':3,
                'candidate_frames':3,
                'reused_completed_measurements':0,
                'raw_comparisons':len(plan['cases']),'baseline_raw_observations_reused':len(plan['cases']),
                'qualification_children':0,'elapsed_seconds':time.monotonic()-started,'aggregate_accepted':False})
        except Exception as exc:
            publish('results.json',failure_receipt(exc,plan['protocol'],cases,frames))
            raise
        finally:
            for path in owner.root.rglob('*'):
                if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                    owner.register_file(str(path.relative_to(owner.root)),'final_output')
    print('ADAPTIVE_ARCH_EXTERIOR_RESULT='+str(owner.root/'results.json'),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
