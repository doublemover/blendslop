#!/usr/bin/env python3
"""Frozen saved-family detail checkpoint; controlled acquisition, bounded local fit.

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
    from reconstruction.adaptive_family import adaptive_crop_request, family_control_values, family_program_update, refine_family_detail
    from reconstruction.frozen_family import retained_family_program
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _pose_controls, _apply_pose
    from run_quality_coverage_check import save_mesh
    from run_surface_quality_check import _replay_orthographic_camera

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    plan=read_json(args.plan)
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('new adaptive checkpoint must remain in isolated temp/tasks')
    if (plan.get('protocol')!='bounded_adaptive_family_checkpoint_v1' or not 1<=len(plan['cases'])<=2
            or plan['deadline_seconds']!=90 or plan['threads']!=2 or plan['memory_limit_bytes']!=8*1024**3
            or plan['segments']!=96 or plan['qualification_children']!=0
            or plan['fit_budget']!={'residual_calls':96,'seconds':1.}
            or plan['global_resolution']!=[512,512] or plan['crop_resolution']!=[1024,1024]
            or plan['crop_pixel_span']!=160):
        raise ValueError('only the frozen small two-case detail checkpoint is supported')
    for filename,digest in plan['input_sha256'].items():
        if sha(filename)!=digest:raise ValueError('frozen input changed: '+filename)
    frames_per_case=3+3*len(plan['heldout_views'])
    expected=sum(0 if item.get('reuse_completed_case') else frames_per_case for item in plan['cases'])
    if plan['rendered_frames']!=expected:
        raise ValueError('declared acquisition count differs from the exact loop')
    if len(set(plan['heldout_views']))!=len(plan['heldout_views']) or not plan['heldout_views']:
        raise ValueError('declare distinct independent validation views')
    owner=OwnedRun(args.output.resolve(),producer='bounded_adaptive_family_checkpoint',
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
        def retained_case(item):
            from evaluation.controlled_measurement import coverage_and_mask,measurement_signature
            run=Path(item['reuse_completed_case'])
            manifest=read_json(run/'run-ownership.json');lease=read_json(run/'run-lease.json')
            if manifest['state']!='failed' or lease['status']!='released':
                raise ValueError('only the explicitly frozen joined failed owner is reusable')
            artifacts={entry['path']:entry for entry in manifest['artifacts']}
            for relative,entry in artifacts.items():
                if sha(run/relative)!=entry['sha256']:
                    raise ValueError('retained partial artifact identity changed: '+relative)
            def load(label,expected):
                folder=run/item['family']/label
                meta=read_json(folder/'measurement.json')
                coverage,mask=coverage_and_mask(np.load(folder/'coverage.npy',allow_pickle=False))
                if (measurement_signature(meta['contract'])!=meta['contract_sha256']
                        or meta['geometry_hashes']!=[expected] or not meta['geometry_unchanged']):
                    raise ValueError('retained partial measurement lost acquisition/geometry binding')
                return {**meta,'coverage':coverage,'mask':mask}
            program=read_json(run/item['family']/'refined/program.json')
            with np.load(run/item['family']/'refined/evaluated-exact.npz',allow_pickle=False) as archive:
                arrays=GeometryArrays.capture(archive['vertices'],archive['faces'])
            if arrays.content_hash!=item['reused_refined_geometry_hash']:
                raise ValueError('retained partial refined indexed identity changed')
            publish(item['family']+'/reused-partial.json',{'owner':str(run),'manifest_sha256':sha(run/'run-ownership.json'),
                'owner_state':'failed','lease':'released','verified_artifacts':len(artifacts),
                'measurements_reused':6,'geometry_hash':arrays.content_hash,'source_sha256':'original first-run frozen-workload remains retained'})
            return program,arrays,load
        publish('frozen-workload.json',{**plan,'plan_path':str(args.plan.resolve()),'plan_sha256':sha(args.plan),
            'environment':{'blender':bpy.app.version_string,'python':sys.version},
            'full_required_five_view_admission':'unrun for changed candidate; current selection preserved',
            'artist_surface_limits':None})
        try:
            for item in plan['cases']:
                deadline();family=item['family'];print('adaptive family '+family,flush=True)
                directory=Path(item['baseline_directory']);wire=read_json(directory/'program.json')
                receipt=read_json(item['baseline_receipt'])
                bound=receipt['cases'][family]
                if (bound['program']!=wire or bound['geometry_hash']!=item['baseline_geometry_hash']
                        or bound['surface_observation']!=item['baseline_raw_surface']):
                    raise ValueError('declared baseline is not bound to its retained receipt')
                manifest=read_json(directory.parent/'run-ownership.json')
                lease=read_json(directory.parent/'run-lease.json')
                if manifest['state']!='succeeded' or lease['status']!='released':
                    raise ValueError('retained baseline owner is not completed/released')
                artifacts={row['path']:row for row in manifest['artifacts']}
                for relative in ('results.json',family+'/program.json',family+'/evaluated-exact.npz',family+'/evaluated.obj'):
                    if sha(directory.parent/relative)!=artifacts[relative]['sha256']:
                        raise ValueError('retained baseline owned byte identity changed')
                if sha(directory/'program.json')!=item['program_sha256']:
                    raise ValueError('retained recipe changed')
                baseline_program=retained_family_program(wire)
                baseline=compile_shape_program(baseline_program,lathe_segments=96,weighted_normals=False)
                baseline_arrays=evaluated_arrays(baseline.root_object)
                if baseline_arrays.content_hash!=item['baseline_geometry_hash']:
                    save(baseline.root_object,family+'/baseline-replay-guard-failed')
                    raise ValueError('retained indexed baseline replay differs; no repair fallback')
                with np.load(directory/'evaluated-exact.npz',allow_pickle=False) as archive:
                    retained=GeometryArrays.capture(archive['vertices'],archive['faces'])
                if retained.content_hash!=baseline_arrays.content_hash:
                    raise ValueError('retained baseline archive differs')
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
                reused=None
                if item.get('reuse_completed_case'):
                    retained_wire,retained_refined,reused=retained_case(item)
                    refined_program=retained_family_program(retained_wire)
                    publish(family+'/refined/program.json',retained_wire)
                    # Validate all three fitting acquisitions, including censored
                    # crop metadata, even though no fitting/ROI selection runs.
                    reused('source-fit',source_arrays.content_hash)
                    reused('baseline-fit',baseline_arrays.content_hash)
                    reused('source-detail',source_arrays.content_hash)
                else:
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
                    refined_program=refine_family_detail(wire,family,observations,
                        parameter_bounds=item['parameter_bounds'],heldout_views=plan['heldout_views'],
                        prior_view_exposure=item['prior_view_exposure'],max_evaluations=96,max_elapsed_s=1.,segments=96)
                    # Both candidate recipes are immutable before raw reference proximity.
                    publish(family+'/refined/program.json',refined_program.to_dict())
                refined=compile_shape_program(refined_program,lathe_segments=96,weighted_normals=False)
                refined_arrays,_=save(refined.root_object,family+'/refined')
                if reused and refined_arrays.content_hash!=retained_refined.content_hash:
                    raise ValueError('retained refined recipe replay differs before reused evaluation')
                baseline_metrics={};refined_metrics={}
                for view in plan['heldout_views']:
                    decl=item['cameras'][view]
                    if reused:
                        source=reused('source-heldout/'+view,source_arrays.content_hash)
                        old=reused('baseline-heldout/'+view,baseline_arrays.content_hash)
                        new=reused('refined-heldout/'+view,refined_arrays.content_hash)
                    else:
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
                controls=family_control_values(refined_program.to_dict(),family)
                control=item['semantic_control'];edited_wire=family_program_update(refined_program.to_dict(),family,
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
                    from run_structured_family_source_edits import observe as structured_observe, response as structured_response
                    from run_family_source_edit_check import physical_observation, response_contract
                    if family in ('torus','concave_arch'):
                        response=structured_response(family,structured_observe(family,before,sources[0]),structured_observe(family,changed,sources[0]))
                    elif family in ('capsule','tapered_frustum'):
                        response=response_contract(family,physical_observation(family,before,sources[0]),
                            physical_observation(family,changed,sources[0]),refined_program.root_nodes[0].parameters)
                    else:
                        response={'passed':False,'status':'unsupported physical response in this checkpoint'}
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
                    and refined_program.metadata['adaptive_detail']['identifiability']=='locally_identified')
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
                'rendered_frames':frames,'source_frames':sum(0 if item.get('reuse_completed_case') else 2+len(plan['heldout_views']) for item in plan['cases']),
                'candidate_frames':sum(0 if item.get('reuse_completed_case') else 1+2*len(plan['heldout_views']) for item in plan['cases']),
                'reused_completed_measurements':sum(6 for item in plan['cases'] if item.get('reuse_completed_case')),
                'raw_comparisons':len(plan['cases']),'baseline_raw_observations_reused':len(plan['cases']),
                'qualification_children':0,'elapsed_seconds':time.monotonic()-started,'aggregate_accepted':False})
        except Exception as exc:
            publish('results.json',failure_receipt(exc,plan['protocol'],cases,frames))
            raise
        finally:
            for path in owner.root.rglob('*'):
                if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                    owner.register_file(str(path.relative_to(owner.root)),'final_output')
    print('ADAPTIVE_FAMILY_RESULT='+str(owner.root/'results.json'),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
