#!/usr/bin/env python3
"""Validate two frozen observed-alpha tail proposals; no native fitting/source render."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'blender_blocking'), str(ROOT/'scripts'), str(ROOT)]
# Blender disables user site; expose the already-installed packages before validation.
import test_runner
PROTOCOL = 'saved_arch_triangle_tail_checkpoint_v1'
THREE_VIEW_PROTOCOL = 'saved_triangle_three_view_tail_checkpoint_v1'
FAMILIES = ('concave_arch', 'rounded_triangle_dot')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify_file(entry, *, max_bytes=67108864):
    path = Path(entry['path']).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > max_bytes or sha(path) != entry['sha256']:
        raise ValueError('frozen file identity/bound changed: '+str(path))
    return path


def verify_owned(entry, *, expected_state='succeeded'):
    """Reuse an explicitly bound input artifact; never acquire/adopt its lease."""
    path = verify_file(entry)
    root = Path(entry['owner_root']).resolve(strict=True)
    manifest, lease = read(root/'run-ownership.json'), read(root/'run-lease.json')
    if (manifest['state'] != expected_state or lease['status'] != 'released'
            or manifest['producer'] != entry['producer']
            or manifest['run_id'] != lease['run_id'] or manifest['owner_token'] != lease['owner_token']):
        raise ValueError('retained artifact producer/closed lease differs')
    relative = path.relative_to(root).as_posix()
    artifacts = {row['path']: row for row in manifest['artifacts']}
    if relative not in artifacts or artifacts[relative]['sha256'] != entry['sha256']:
        raise ValueError('retained artifact differs from its original ownership record')
    return path


def validate_scope(plan):
    expected = {'work_seconds':85, 'join_seconds':5, 'threads':2,
        'memory_limit_bytes':8*1024**3, 'native_frames':7, 'source_acquisitions':0,
        'candidate_alpha_frames':5, 'candidate_neutral_frames':2, 'native_fits':0,
        'retained_alpha_frames':5, 'logical_frames':12,
        'raw_comparisons':2, 'raw_count_per_direction':4096, 'raw_seed':61007,
        'semantic_transactions':2, 'qualification_children':2, 'qualification_timeout_seconds':15,
        'resolution':[512,512], 'heldout_views':['oblique_145_40']}
    families = FAMILIES
    if plan.get('protocol') == THREE_VIEW_PROTOCOL:
        expected.update(native_frames=6,candidate_alpha_frames=5,candidate_neutral_frames=1,retained_alpha_frames=0,
            logical_frames=6,raw_comparisons=1,semantic_transactions=1,qualification_children=1)
        families = ('rounded_triangle_dot',)
    if (plan.get('protocol') not in (PROTOCOL,THREE_VIEW_PROTOCOL) or plan.get('scope') != expected
            or any(type(value) in (int,bool) and type(plan.get('scope',{}).get(key)) is not type(value)
                   for key,value in expected.items())
            or tuple(c['family'] for c in plan.get('cases', [])) != families):
        raise ValueError('only the frozen two-family12-frame or triangle-only6-frame saved-proposal scopes are supported')


def require_recipe_update(old, proposed, family):
    """No proposed body can change controls outside its observed follow-up."""
    if family == 'concave_arch':
        from reconstruction.arch_tail_refinement import arch_roof_update
        key, update = 'arch_roof_tail', arch_roof_update
    elif family == 'rounded_triangle_dot':
        from reconstruction.triangle_tail_refinement import triangle_pose_update
        key, update = 'triangle_pose_tail', triangle_pose_update
    else:
        raise ValueError('unsupported tail family')
    detail = proposed['metadata'][key]
    expected_controls = {'notch_height_from_bottom_world'} if family == 'concave_arch' else {'center_y_world','center_z_world','front_fraction'}
    if set(detail['free_controls']) != expected_controls or detail['local_rank'] != len(expected_controls):
        raise ValueError('proposal local rank/free-control declaration differs')
    if (detail['observed_tail_admitted'] is not True or detail['identifiability'] != 'locally_identified'
            or detail['interval_bound_active'] is not False or detail['heldout_fit_or_roi_used'] is not False
            or detail['heldout_views'] != ['oblique_145_40']
            or detail['baseline_program_file_sha256'] is None):
        raise ValueError('proposal lacks its conservative observed admission/rank/history')
    expected = update(old, detail['selected_controls']).to_dict()
    for field in ('schema_version','program_id','root_nodes','constraints','residual_patches'):
        if expected[field] != proposed[field]:
            raise ValueError('proposal changed an unobserved dimension/frame/template: '+field)
    for key, value in detail['selected_observed_tail'].items():
        if value > detail['baseline_observed_tail'][key]+1e-12:
            raise ValueError('proposal worsened an observed tail score')
    return detail


def load_observation(entry, expected, *, expected_state='succeeded'):
    import numpy as np
    from evaluation.controlled_measurement import measurement_signature
    coverage = np.load(verify_owned(entry['coverage'],expected_state=expected_state), allow_pickle=False)
    metadata = read(verify_owned(entry['metadata'],expected_state=expected_state))
    if (metadata['geometry_hashes'] != [expected] or metadata['geometry_unchanged'] is not True
            or measurement_signature(metadata['contract']) != metadata['contract_sha256']):
        raise ValueError('retained source/baseline alpha lost acquisition/geometry binding')
    if 'coverage_sha256' in metadata and metadata['coverage_sha256'] != entry['coverage']['sha256']:
        raise ValueError('retained NPY file differs from its producer measurement SHA')
    return {**metadata, 'coverage':coverage}


def verify_inputs(plan):
    if not 1 <= len(plan['input_sha256']) <= 128:
        raise ValueError('bounded frozen input inventory required')
    for path, digest in plan['input_sha256'].items():
        verify_file({'path':path,'sha256':digest})
    retained = plan.get('retained_input_sha256', {})
    if plan.get('protocol') == THREE_VIEW_PROTOCOL:
        if retained:raise ValueError('triangle three-view checkpoint renders all six frames; retained inputs forbidden')
    elif not 1 <= len(retained) <= 32:
        raise ValueError('bounded explicit retained input inventory required')
    retained_bytes = 0
    for path, digest in retained.items():
        checked = verify_file({'path':path,'sha256':digest})
        retained_bytes += checked.stat().st_size
    if retained_bytes > 67108864:
        raise ValueError('retained five-frame input inventory exceeds64MiB')
    toolchain = plan.get('toolchain_sha256', {})
    if not 1 <= len(toolchain) <= 4:
        raise ValueError('bounded existing executable/package inventory required')
    total = 0
    for path, digest in toolchain.items():
        checked = verify_file({'path':path,'sha256':digest},max_bytes=536870912)
        total += checked.stat().st_size
    if total > 536870912:
        raise ValueError('existing toolchain inventory exceeds512MiB')


def validate_plan(plan):
    from evaluation.canonical_artifacts import CANONICAL_VIEWS
    validate_scope(plan)
    verify_inputs(plan)
    for case in plan['cases']:
        old = read(verify_owned(case['baseline_program']))
        proposed = read(verify_owned(case['proposed_program']))
        detail = require_recipe_update(old, proposed, case['family'])
        if (detail['baseline_geometry_hash'] != case['baseline_geometry_hash']
                or detail['baseline_program_file_sha256'] != case['baseline_program']['sha256']):
            raise ValueError('proposal baseline body/recipe differs')
        raw = case['baseline_raw_surface']
        if (raw['candidate_geometry_hash'] != case['baseline_geometry_hash']
                or raw['reference_geometry_hash'] != case['source_geometry_hash']
                or raw['sample_count_per_direction'] != 4096 or raw['seed'] != 61007
                or raw['protocol'] != 'shared_world_area_sample_to_triangle_v1'):
            raise ValueError('old raw observation belongs to a different body/protocol')
        packet = read(verify_file(case['baseline_evidence']))['independent_case']
        if (packet['geometry_hash'] != case['baseline_geometry_hash']
                or packet['source_geometry_hash'] != case['source_geometry_hash']
                or packet['refined_raw_surface'] != raw
                or packet['silhouette']['views'] != case['baseline_views']):
            raise ValueError('baseline raw/silhouette fields changed from retained exact packet')
        if set(case['source_views']) != set(CANONICAL_VIEWS) or set(case['original_cameras']) != set(CANONICAL_VIEWS):
            raise ValueError('all five original declarations and source observations required')
        for view in CANONICAL_VIEWS: load_observation(case['source_views'][view],case['source_geometry_hash'])
        source = packet['canonical_inspection']['actual_matched_views']['oblique_35_28']
        preview = source['passes']['neutral']['source']
        if (source['actual_frame_match'] is not True or source['actual_clipping_match'] is not True
                or preview['geometry_hash'] != case['source_geometry_hash']
                or preview['sha256'] != case['source_neutral']['sha256']
                or preview['render_settings'] != case['neutral_settings']):
            raise ValueError('retained paired source neutral/frame/style differs')
        verify_owned(case['source_neutral'])
        reuse=case.get('retained_alpha',{})
        if case['family']=='concave_arch':
            if set(reuse)!=set(CANONICAL_VIEWS):raise ValueError('exact five retained arch alpha slots required')
            prior=read(verify_owned(case['retained_receipt'],expected_state='failed'))
            row=prior['cases']['concave_arch']
            if (prior['status']!='failed' or prior['completed_native_frames']!=5
                    or prior['attempted_native_frames']!=5 or prior['qualification_children_invoked']!=0
                    or prior['raw_comparisons']!=0 or prior['semantic_transactions']!=0
                    or row['program']!=proposed or row['geometry_hash']!=case['retained_geometry_hash']):
                raise ValueError('failed partial owner differs from exact five-frame attempt')
            for view,entry in reuse.items():
                for name in ('coverage','metadata','exr','mask'):
                    if entry[name]['producer']!='saved_arch_triangle_tail_checkpoint':
                        raise ValueError('retained failed alpha producer differs')
                    verify_owned(entry[name],expected_state='failed')
                measured=load_observation(entry,case['retained_geometry_hash'],expected_state='failed')
                source=load_observation(case['source_views'][view],case['source_geometry_hash'])
                from evaluation.controlled_measurement import measurement_signature
                if (measured['exr_sha256']!=entry['exr']['sha256'] or
                        measurement_signature(source['contract'])!=measurement_signature(measured['contract'])):
                    raise ValueError('retained alpha source/frame/acquisition differs')
            verify_owned(case['retained_npz'],expected_state='failed')
        elif reuse:raise ValueError('triangle frames remain unrun; reuse forbidden')
    return True


def encoded(value):
    import numpy as np
    def scalar(item):
        if isinstance(item,np.generic): return item.item()
        raise TypeError('unsupported receipt value: '+type(item).__name__)
    return json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n'


def execute(plan, output, plan_binding):
    import bpy
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import camera_record
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from evaluation.canonical_artifacts import raw_surface_observation, CANONICAL_VIEWS
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from primitives.shape_program_compiler import compile_shape_program
    from reconstruction.frozen_family import retained_family_program
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.output_targets import output_mesh_targets
    from reconstruction.output_qualification import qualify_retained_output
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _compile_exact, _pose_controls, _apply_pose
    from run_selected_canonical_inspection import load_exact, require_same_frame, _settings
    from run_surface_quality_check import _replay_orthographic_camera
    from run_quality_coverage_check import save_mesh
    from reconstruction.arch_tail_refinement import arch_roof_controls, arch_roof_update
    from reconstruction.triangle_tail_refinement import triangle_pose_controls, triangle_pose_update
    if list(bpy.app.version)!=[5,2,2] or bpy.app.build_hash.decode()!='d13f752e3b9c':
        raise ValueError('frozen Blender5.2.2 build required')
    validate_plan(plan)
    started=time.monotonic(); deadline=started+85.
    receipt={'protocol':plan['protocol'],'status':'running','cases':{},'completed_native_frames':0,
             'attempted_native_frames':0,'qualification_children_invoked':0,'raw_comparisons':0,
             'semantic_transactions':0,'native_fits':0,'source_acquisitions':0,'retained_alpha_frames_verified':0,'aggregate_accepted':False}
    owner=OwnedRun(output,producer='saved_arch_triangle_tail_checkpoint',max_generated_bytes=268435456,
                   shared_inputs={**plan['input_sha256'],**plan['toolchain_sha256'],**plan['retained_input_sha256'],plan_binding['path']:plan_binding['sha256']})
    with owner:
        def publish(relative,value):
            path=owner.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            data=encoded(value).encode();owner.reserve_bytes(len(data));path.write_bytes(data)
            owner.register_file(path.relative_to(owner.root),'final_output')
        def check_time():
            if time.monotonic()>deadline:raise TimeoutError('fixed85-second saved tail work deadline')
        publish('frozen-workload.json',{**plan,'plan_binding':plan_binding})
        publish('results.json',receipt)
        try:
            for case in plan['cases']:
                check_time();family=case['family'];wire=read(case['proposed_program']['path'])
                old=load_exact(case['baseline_npz']['path'],case['baseline_geometry_hash'])
                baseline,old_source,captured=_compile_exact(read(case['baseline_program']['path']),old)
                old_pointer=old_source.as_pointer()
                program=retained_family_program(wire)
                candidate=compile_shape_program(program,lathe_segments=96,weighted_normals=False)
                targets=output_mesh_targets([candidate.root_object])
                if len(targets)!=1:raise ValueError('tail proposal requires one real native mesh')
                obj=targets[0];arrays=evaluated_arrays(candidate.root_object)
                folder=owner.root/family;folder.mkdir();publish(family+'/program.json',wire)
                saved,hashes=save_mesh(candidate.root_object,folder)
                if saved.content_hash!=arrays.content_hash:raise ValueError('saved proposal changed indexed geometry')
                if case.get('retained_alpha') and arrays.content_hash!=case['retained_geometry_hash']:
                    raise ValueError('retained five alpha frames belong to a different proposal body')
                # Every recipe is saved/hash-bound before reading reference arrays.
                source_arrays=load_exact(case['source_npz']['path'],case['source_geometry_hash'])
                row={'status':'running','geometry_hash':arrays.content_hash,'source_geometry_hash':case['source_geometry_hash'],
                     'baseline_geometry_hash':captured.content_hash,'program':wire,'artifact_directory':str(folder),
                     'baseline_raw_surface_reused':case['baseline_raw_surface'],'artist_surface_limits':None,
                     'aggregate_accepted':False,'current_selection_unchanged':True,
                     'canonical_neutral_normal_passes':'one representative neutral; other changed-body previews unrun',**hashes}
                receipt['cases'][family]=row;publish('results.json',receipt)
                bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.type='ORTHO'
                metrics={};actual={}
                for view in CANONICAL_VIEWS:
                    check_time()
                    source=load_observation(case['source_views'][view],case['source_geometry_hash'])
                    declaration={**plan['replay_clips'],**case['original_cameras'][view]}
                    reuse=case.get('retained_alpha',{}).get(view)
                    if reuse:
                        measured=load_observation(reuse,arrays.content_hash,expected_state='failed')
                        measured['mask']=measured['coverage']>=.5
                        receipt['retained_alpha_frames_verified']+=1
                    else:
                        owner.reserve_bytes(8388608)
                        with controlled_measurement_session(target_objects=[obj],camera=camera,resolution=(512,512),samples=64,filter_size=1.5) as session:
                            _replay_orthographic_camera(camera,declaration)
                            receipt['attempted_native_frames']+=1
                            measured=render_controlled_measurement(session,folder/(view+'-mask.exr'))
                            receipt['completed_native_frames']+=1
                    if measured['geometry_hashes']!=[arrays.content_hash] or not measured['geometry_unchanged']:
                        raise ValueError('new alpha changed exact proposal geometry')
                    metrics[view]=compare_controlled_measurements(source,measured,view=view,
                        config=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05))
                    if reuse:
                        row.setdefault('retained_alpha_reuse',{})[view]={'original_artifacts':reuse,
                            'actual_geometry_hash':arrays.content_hash,'source_contract_sha256':source['contract_sha256'],
                            'measurement_contract_sha256':measured['contract_sha256'],'owner_adopted':False,'rerendered':False}
                    else:
                        np.save(folder/(view+'-coverage.npy'),measured['coverage'],allow_pickle=False)
                        Image.fromarray(np.where(measured['mask'],0,255).astype(np.uint8)).save(folder/(view+'-mask.png'))
                        meta={k:v for k,v in measured.items() if k not in ('coverage','mask')}
                        meta['coverage_sha256']=sha(folder/(view+'-coverage.npy'))
                        publish(family+'/'+view+'-measurement.json',meta)
                    actual[view]=measured['contract']['camera']
                check_time();view='oblique_35_28'
                with silhouette_session(target_objects=[obj],camera=camera,resolution=(512,512),color_mode='BW',
                    transparent_bg=False,engine='BLENDER_WORKBENCH',force_material=False,ensure_light_obj=False) as session:
                    _replay_orthographic_camera(camera,{**plan['replay_clips'],**case['original_cameras'][view]})
                    before_update=camera_record(camera,resolution=(512,512),pixel_aspect=(session.scene.render.pixel_aspect_x,session.scene.render.pixel_aspect_y))
                    bpy.context.view_layer.update()
                    shading=session.scene.display.shading;shading.light='STUDIO';shading.color_type='SINGLE'
                    shading.single_color=(.65,.65,.65);shading.show_shadows=True;shading.show_cavity=False;shading.background_type='WORLD'
                    settings=_settings(session.scene,'neutral')
                    if settings!=case['neutral_settings']:raise ValueError('candidate neutral settings differ from retained source')
                    frame=camera_record(camera,resolution=(512,512),pixel_aspect=(session.scene.render.pixel_aspect_x,session.scene.render.pixel_aspect_y))
                    source_frame=read(case['source_views'][view]['metadata']['path'])['contract']['camera']
                    camera_check={'expected_alpha_frame':actual[view],'bound_source_alpha_frame':source_frame,
                        'after_original_replay_before_update':before_update,'after_view_layer_update':frame}
                    publish(family+'/neutral-camera-check.json',camera_check)
                    require_same_frame(actual[view],frame)
                    require_same_frame(source_frame,frame)
                    owner.reserve_bytes(8388608);path=folder/(view+'-neutral.png')
                    receipt['attempted_native_frames']+=1;render_silhouette_frame(session,path);receipt['completed_native_frames']+=1
                    after_render=camera_record(camera,resolution=(512,512),pixel_aspect=(session.scene.render.pixel_aspect_x,session.scene.render.pixel_aspect_y))
                    camera_check['after_render']=after_render;publish(family+'/neutral-camera-check.json',camera_check)
                    require_same_frame(actual[view],after_render);require_same_frame(source_frame,after_render)
                row['neutral_preview']={'candidate':bind(path),'source':case['source_neutral'],'actual_camera':frame,
                                        'settings':settings,'geometry_hash':arrays.content_hash,'source_geometry_hash':case['source_geometry_hash']}
                check_time();raw=raw_surface_observation(source_arrays,arrays,count=4096,seed=61007)
                receipt['raw_comparisons']+=1;row['refined_raw_surface']=raw;row['silhouette']={'status':'passed' if all(v['passed'] for v in metrics.values()) else 'failed','views':metrics}
                if family=='concave_arch':
                    controls=arch_roof_controls(wire);control='notch_height_from_bottom_world';multiplier=1.005
                    edited_wire=arch_roof_update(wire,{control:controls[control]*multiplier}).to_dict()
                else:
                    controls=triangle_pose_controls(wire);control='front_fraction';multiplier=1.01
                    edited_wire=triangle_pose_update(wire,{control:controls[control]*multiplier}).to_dict()
                publish(family+'/semantic-edited-program.json',edited_wire)
                edited=compile_shape_program(retained_family_program(edited_wire),lathe_segments=96,weighted_normals=False)
                edited_targets=output_mesh_targets([edited.root_object])
                if len(edited_targets)!=1:raise ValueError('physical edit requires one real mesh')
                mesh=obj.data;pose=_pose_controls(obj);pointer=obj.as_pointer()
                before=np.array([tuple(v.co) for v in mesh.vertices]);after=np.array([tuple(v.co) for v in edited_targets[0].data.vertices])
                checks={'fixed_vertex_inventory':before.shape==after.shape}
                if family=='concave_arch':
                    roof=np.unique(before[:,1])[1];moving=before[:,1]==roof
                    delta=controls[control]*(multiplier-1.)
                    checks.update(fixed_all_other_vertices=bool(np.array_equal(before[~moving],after[~moving])),
                        fixed_x_and_depth=bool(np.array_equal(before[:,[0,2]],after[:,[0,2]])),
                        roof_height_response=bool(np.allclose(after[moving,1]-before[moving,1],delta,atol=1e-5,rtol=0.)))
                else:
                    fraction=controls[control];front=before[:,2]>=0;ratios=np.where(front,multiplier,(1.-fraction*multiplier)/(1.-fraction))
                    checks.update(fixed_all_outline_vertices_xy=bool(np.array_equal(before[:,:2],after[:,:2])),
                        asymmetric_depth_response=bool(np.allclose(after[:,2],before[:,2]*ratios,atol=1e-5,rtol=0.)))
                try:
                    obj.data=edited_targets[0].data;_apply_pose(obj,_pose_controls(edited_targets[0]));bpy.context.view_layer.update()
                    changed=evaluated_arrays(candidate.root_object)
                finally:
                    obj.data=mesh;_apply_pose(obj,pose);bpy.context.view_layer.update()
                restored=evaluated_arrays(candidate.root_object)
                edit={'control':control,'multiplier':multiplier,'response_checks':checks,
                      'same_source_pointer':obj.as_pointer()==pointer,'original_mesh_restored':obj.data==mesh,
                      'geometry_changed':changed.content_hash!=arrays.content_hash,
                      'exact_indexed_restoration':restored.content_hash==arrays.content_hash}
                edit['passed']=bool(all(checks.values()) and all(edit[k] for k in ('same_source_pointer','original_mesh_restored','geometry_changed','exact_indexed_restoration')))
                receipt['semantic_transactions']+=1;row['semantic_edit_restoration']=edit
                check_time()
                if deadline-time.monotonic()<20.:raise TimeoutError('insufficient fixed allowance before15-second qualifier')
                receipt['qualification_children_invoked']+=1
                boundary=qualify_retained_output(arrays,{'native_qualification_python':plan['qualification_python'],
                    'native_qualification_timeout_s':15.,'native_run_ownership_root':output/'qualification-children'})
                if boundary['geometry_content_hash']!=arrays.content_hash:raise ValueError('boundary belongs to a different body')
                row['boundary']=boundary
                old_raw=case['baseline_raw_surface']
                tail_keys=('symmetric_mean_distance_world','distance_p95_world','sampled_max_distance_world')
                row['raw_tail_nonworse']={key:raw[key]<=old_raw[key] for key in tail_keys}
                row['heldout_alpha_nonworse']=metrics['oblique_145_40']['coverage_l1']<=case['baseline_views']['oblique_145_40']['coverage_l1']
                row['bounded_checkpoint_improved']=bool(row['silhouette']['status']=='passed' and all(row['raw_tail_nonworse'].values())
                    and row['heldout_alpha_nonworse'] and edit['passed'] and boundary.get('single_solid_qualified'))
                row['fixed_baseline_geometry_stable']=evaluated_arrays(baseline.root_object).content_hash==captured.content_hash and old_source.as_pointer()==old_pointer
                if not row['fixed_baseline_geometry_stable'] or evaluated_arrays(candidate.root_object).content_hash!=arrays.content_hash:
                    raise ValueError('validation mutated frozen baseline/proposal')
                row['status']='measured';publish('results.json',receipt)
            verify_inputs(plan)
            if (receipt['completed_native_frames']!=plan['scope']['native_frames']
                    or receipt['retained_alpha_frames_verified']!=plan['scope']['retained_alpha_frames']
                    or receipt['raw_comparisons']!=plan['scope']['raw_comparisons']
                    or receipt['semantic_transactions']!=plan['scope']['semantic_transactions']
                    or receipt['qualification_children_invoked']!=plan['scope']['qualification_children']):
                raise ValueError('unexpected final exact frozen scope counts')
            receipt.update(status='measured',elapsed_seconds=time.monotonic()-started)
            publish('results.json',receipt)
        except BaseException as primary:
            receipt.update(status='failed',reason=type(primary).__name__+': '+str(primary))
            try:publish('results.json',receipt)
            except BaseException as secondary:
                if hasattr(primary,'add_note'):primary.add_note('receipt publication also failed: '+repr(secondary))
            raise
        finally:
            active_primary=sys.exception()
            try:
                for path in owner.root.rglob('*'):
                    if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                        owner.register_file(path.relative_to(owner.root),'final_output')
            except BaseException as auxiliary:
                if active_primary is None:raise
                if hasattr(active_primary,'add_note'):
                    active_primary.add_note('artifact registration also failed: '+repr(auxiliary))
    print('TAIL_CHECKPOINT_RESULT='+str(owner.root/'results.json'),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):raise ValueError('owned output must stay in isolated temp/tasks')
    if sha(args.plan)!=args.plan_sha256:raise ValueError('frozen plan bytes changed')
    plan=read(args.plan);validate_plan(plan)
    if args.validate_only:
        print('TAIL_CHECKPOINT_VALIDATED='+args.plan_sha256,flush=True)
        return 0
    return execute(plan,args.output.resolve(),bind(args.plan))


if __name__=='__main__':raise SystemExit(main())
