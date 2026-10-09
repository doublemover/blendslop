#!/usr/bin/env python3
"""One rounded-box512global +1024residual-crop experiment, reserved obliques."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking'), str(ROOT/'scripts')]


def main():
    import bpy
    import numpy as np
    import test_runner
    from mathutils import Matrix
    from evaluation.canonical_artifacts import raw_surface_observation
    from evaluation.reference_noise import oriented_surface_identity
    from evaluation.adaptive_measurement import residual_crop_request
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from reconstruction.frozen_family import observed_target, fitted_family_program
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.grouped_solids import solid_guard
    from primitives.shape_program_compiler import compile_shape_program
    from synthetic.quality_references import build_quality_reference
    from run_surface_quality_check import _replay_orthographic_camera
    from utils.run_ownership import OwnedRun
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-source',type=Path,help='Completed owned source-global measurements; verify all retained hashes')
    parser.add_argument('--reuse-baseline',type=Path,help='Completed corrected comparison; reuse frozen baseline and compare against prior selected quality')
    parser.add_argument('--crop-resolution',type=int,choices=(1024,2048),default=1024)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('output must stay in isolated project temp/tasks')
    source_root=args.reference_root.resolve(strict=True)
    if args.reuse_baseline and not args.reuse_source:
        raise ValueError('baseline continuation requires explicit completed source measurements')
    original_workload=json.loads((source_root/'frozen-workload.json').read_text())
    case=next(row for row in original_workload['cases'] if row['name']=='rounded_box')
    original_row=json.loads((source_root/'results.json').read_text())['cases']['rounded_box']
    input_cameras=original_row['reference_cameras']
    training=('front','side','top')
    heldout=('oblique_35_28','oblique_145_40')
    deadline=120.;start=time.monotonic()
    frozen={'protocol':'controlled_residual_crop_experiment_v1','family':'rounded_box',
        'source_case':case,'source_geometry_hash':original_row['geometry_hash'],
        'source_workload_sha256':hashlib.sha256((source_root/'frozen-workload.json').read_bytes()).hexdigest(),
        'original_camera_declarations':input_cameras,'fit_views':training,'heldout_views':heldout,
        'baseline_resolution':[512,512],'crop_resolution':[args.crop_resolution,args.crop_resolution],'crop_pixel_span':160,
        'samples':64,'fit_allowance_per_strategy':{'residual_calls':256,'seconds':3.},
        'renders':3 if args.reuse_baseline else (8 if args.reuse_source else 13),'candidate_tessellation':{'bevel_segments':8},'raw_surface_samples_per_direction':4096,'seed':61007,
        'admission':'both strict heldout gates pass; mean heldout coverage L1 strictly improves; minimum boundary IoU and raw mean world surface distance do not worsen',
        'meaningful_win':'at least10percent relative reduction in heldout coverage L1; diagnostic experiment criterion, not family acceptance',
        'independence':'heldout pixels excluded from fitting/crop selection; final comparison only',
        'scope':'controlled source geometry permits real crop rendering; external image zooms cannot create new truth',
        'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),ROOT/'blender_blocking/evaluation/adaptive_measurement.py',ROOT/'blender_blocking/evaluation/controlled_measurement.py',ROOT/'blender_blocking/integration/blender_ops/measurement_render.py',ROOT/'blender_blocking/reconstruction/frozen_family.py',ROOT/'blender_blocking/primitives/shape_program_compiler.py',ROOT/'blender_blocking/synthetic/quality_references.py',ROOT/'blender_blocking/synthetic/blender_builders.py')},
        'prior_selected_run':str(args.reuse_baseline.resolve()) if args.reuse_baseline else None,
        'continuation_rule':'retain prior selected recipe unless strict heldout, mean coverage, minimum boundary and raw mean surface all improve or remain no worse',
        'qualification_children':0,'deadline_seconds':deadline,'threads':2,'rss_limit_bytes':8*1024**3}
    owner=OwnedRun(args.output.resolve(),producer='controlled_residual_crop_experiment',max_generated_bytes=134217728,
                   shared_inputs={'reference_root':str(source_root)})
    with owner:
        def publish(relative,value,category='final_output'):
            path=owner.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
            owner.register_file(relative,category)
        publish('frozen-workload.json',frozen,'diagnostic')
        reference=build_quality_reference(case).object
        reference_arrays=evaluated_arrays(reference)
        with np.load(source_root/'rounded_box/evaluated-exact.npz',allow_pickle=False) as old:
            retained=GeometryArrays.capture(old['vertices'],old['faces'])
        if retained.content_hash!=original_row['geometry_hash'] or oriented_surface_identity(retained)!=oriented_surface_identity(reference_arrays):
            raise ValueError('authored source geometry changed before experiment')
        bpy.ops.object.camera_add();camera=bpy.context.object
        camera.data.type='ORTHO';bpy.context.scene.camera=camera
        gates=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
        timings={}
        def measure(obj,cameras,names,label,resolution):
            since=time.monotonic();records={}
            folder=owner.root/label;folder.mkdir()
            with controlled_measurement_session(target_objects=[obj],camera=camera,resolution=resolution,samples=64) as session:
                for view in names:
                    if time.monotonic()-start>deadline:raise TimeoutError('fixed experiment deadline')
                    _replay_orthographic_camera(camera,cameras[view])
                    record=render_controlled_measurement(session,folder/(view+'.exr'))
                    owner.register_file(str((folder/(view+'.exr')).relative_to(owner.root)),'diagnostic')
                    np.save(folder/(view+'-coverage.npy'),record['coverage'],allow_pickle=False)
                    owner.register_file(str((folder/(view+'-coverage.npy')).relative_to(owner.root)),'final_output')
                    meta={key:value for key,value in record.items() if key not in ('coverage','mask')}
                    publish(str((folder/(view+'-measurement.json')).relative_to(owner.root)),meta)
                    records[view]=record
            timings[label]=time.monotonic()-since
            return records
        if args.reuse_source:
            from evaluation.controlled_measurement import coverage_and_mask, measurement_signature
            reused=args.reuse_source.resolve(strict=True)
            manifest=json.loads((reused/'run-ownership.json').read_text())
            lease=json.loads((reused/'run-lease.json').read_text())
            if lease.get('status')!='released' or manifest.get('state')!='succeeded':
                raise ValueError('reused measurement owner is not completed/released')
            records={row['path']:row for row in manifest['artifacts']}
            source={}
            for view in training+heldout:
                for suffix in ('-coverage.npy','-measurement.json','.exr'):
                    relative='source-global/'+view+suffix
                    path=reused/relative; bound=records[relative]
                    if path.stat().st_size>33554432 or hashlib.sha256(path.read_bytes()).hexdigest()!=bound['sha256']:
                        raise ValueError('reused source acquisition byte identity changed')
                meta=json.loads((reused/'source-global'/ (view+'-measurement.json')).read_text())
                if (measurement_signature(meta['contract'])!=meta['contract_sha256']
                        or meta['geometry_hashes']!=[reference_arrays.content_hash]):
                    raise ValueError('reused source acquisition contract/geometry identity changed')
                coverage,mask=coverage_and_mask(np.load(reused/'source-global'/(view+'-coverage.npy'),allow_pickle=False))
                if coverage.shape!=(512,512):raise ValueError('reused source dimensions changed')
                source[view]={**meta,'coverage':coverage,'mask':mask}
            publish('reused-source.json',{'root':str(reused),'manifest_sha256':hashlib.sha256((reused/'run-ownership.json').read_bytes()).hexdigest(),
                'source_geometry_hash':reference_arrays.content_hash,'all_source_files_verified':True,
                'source_measurement_protocol':'unchanged from retained alpha acquisitions'})
            timings['source-global-reused']=0.
        else:
            source=measure(reference,input_cameras,training+heldout,'source-global',(512,512))
        def fit(records,label):
            since=time.monotonic()
            masks={name:row['mask'] for name,row in records.items()}
            coverage={name:row['coverage'] for name,row in records.items()}
            cameras={name:row['contract']['camera'] for name,row in records.items()}
            target=observed_target(masks,cameras,coverage_masks=coverage)
            program=fitted_family_program('rounded_box',target,masks,cameras,coverage_masks=coverage,
                max_evaluations=256,max_elapsed_s=3.)
            # Freeze the existing family producer's eight-segment native model
            # explicitly; default compiler tessellation is three for other uses.
            from dataclasses import replace
            node=program.root_nodes[0]
            program=replace(program,root_nodes=(replace(node,parameters={**node.parameters,'bevel_segments':8}),),
                metadata={**program.metadata,'candidate_tessellation':{'bevel_segments':8}})
            publish(label+'/program.json',program.to_dict())
            compiled=compile_shape_program(program,lathe_segments=96,weighted_normals=False)
            arrays=evaluated_arrays(compiled.root_object)
            np.savez_compressed(owner.root/label/'evaluated-exact.npz',vertices=arrays.vertices,faces=arrays.faces)
            owner.register_file(label+'/evaluated-exact.npz','final_output')
            timings[label+'-fit']=time.monotonic()-since
            return compiled,arrays,program
        prior = None
        if args.reuse_baseline:
            from run_admitted_box_followup import retained_file
            from reconstruction.frozen_family import retained_family_program
            retained_run = args.reuse_baseline.resolve(strict=True)
            prior = json.loads(retained_file(retained_run,'results.json').read_text())
            if not prior.get('crop_admitted') or prior['crop_resolution'] != [1024,1024]:
                raise ValueError('only completed admitted1024 comparison may supply the baseline')
            prior_workload = json.loads(retained_file(retained_run,'frozen-workload.json').read_text())
            prior_program_path = retained_file(retained_run,'refined/program.json')
            prior_archive_path = retained_file(retained_run,'refined/evaluated-exact.npz')
            if json.loads(prior_program_path.read_text()) != prior['refined']['program']:
                raise ValueError('prior selected recipe changed')
            with np.load(prior_archive_path,allow_pickle=False) as saved:
                prior_arrays = GeometryArrays.capture(saved['vertices'],saved['faces'])
            if prior_arrays.content_hash != prior['refined']['geometry_hash']:
                raise ValueError('prior selected exact geometry changed')
            if (prior_workload['source_geometry_hash'] != frozen['source_geometry_hash']
                    or prior_workload['original_camera_declarations'] != input_cameras):
                raise ValueError('continued source geometry/camera declarations changed')
            baseline_program = retained_family_program(json.loads(retained_file(retained_run,'baseline/program.json').read_text()))
            with np.load(retained_file(retained_run,'baseline/evaluated-exact.npz'),allow_pickle=False) as saved:
                baseline_arrays = GeometryArrays.capture(saved['vertices'],saved['faces'])
            if baseline_arrays.content_hash != prior['baseline']['geometry_hash']:
                raise ValueError('continued baseline exact geometry changed')
            baseline_views = {}
            from evaluation.controlled_measurement import coverage_and_mask,measurement_signature
            for view in training+heldout:
                meta = json.loads(retained_file(retained_run,'baseline-global/'+view+'-measurement.json').read_text())
                coverage,mask = coverage_and_mask(np.load(retained_file(retained_run,'baseline-global/'+view+'-coverage.npy'),allow_pickle=False))
                retained_file(retained_run,'baseline-global/'+view+'.exr')
                if (measurement_signature(meta['contract']) != meta['contract_sha256']
                        or meta['geometry_hashes'] != [baseline_arrays.content_hash]):
                    raise ValueError('continued baseline measurement identity changed')
                baseline_views[view] = {**meta,'coverage':coverage,'mask':mask}
            publish('baseline/program.json',baseline_program.to_dict())
            np.savez_compressed(owner.root/'baseline/evaluated-exact.npz',vertices=baseline_arrays.vertices,faces=baseline_arrays.faces)
            owner.register_file('baseline/evaluated-exact.npz','final_output')
            publish('reused-baseline.json',{'root':str(retained_run),'receipt_sha256':hashlib.sha256((retained_run/'results.json').read_bytes()).hexdigest(),'verified_measurements':5,'geometry_hash':baseline_arrays.content_hash})
            timings['baseline-fit-reused']=0.
            timings['baseline-global-reused']=0.
        else:
            baseline,baseline_arrays,baseline_program=fit({view:source[view] for view in training},'baseline')
            baseline_views=measure(baseline.root_object,input_cameras,training+heldout,'baseline-global',(512,512))
        request=residual_crop_request(source['top']['coverage'],baseline_views['top']['coverage'],
            source['top']['contract']['camera'],pixel_span=160)
        publish('crop-request.json',request)
        if request['status']!='requested':raise ValueError('no crop required by fitting residual')
        crop=measure(reference,{'top_crop':request['camera']},('top_crop',),'source-crop',(args.crop_resolution,args.crop_resolution))
        refined,refined_arrays,refined_program=fit({**{view:source[view] for view in training},**crop},'refined')
        refined_views=measure(refined.root_object,input_cameras,heldout,'refined-heldout',(512,512))
        baseline_metrics={view:compare_controlled_measurements(source[view],baseline_views[view],view=view,config=gates) for view in heldout}
        refined_metrics={view:compare_controlled_measurements(source[view],refined_views[view],view=view,config=gates) for view in heldout}
        # Candidate recipes above are frozen before any reference-array proximity metrics.
        surfaces={'baseline':prior['baseline']['surface'] if prior else raw_surface_observation(reference_arrays,baseline_arrays,count=4096,seed=61007),
                  'refined':raw_surface_observation(reference_arrays,refined_arrays,count=4096,seed=61007)}
        old_l1=float(np.mean([row['coverage_l1'] for row in baseline_metrics.values()]))
        new_l1=float(np.mean([row['coverage_l1'] for row in refined_metrics.values()]))
        old_boundary=min(row['boundary_iou'] for row in baseline_metrics.values())
        new_boundary=min(row['boundary_iou'] for row in refined_metrics.values())
        prior_l1=float(np.mean([row['coverage_l1'] for row in prior['refined']['heldout'].values()])) if prior else old_l1
        prior_boundary=min(row['boundary_iou'] for row in prior['refined']['heldout'].values()) if prior else old_boundary
        prior_surface=prior['refined']['surface'] if prior else surfaces['baseline']
        admitted=(all(row['passed'] for row in refined_metrics.values()) and new_l1<prior_l1
            and new_boundary>=prior_boundary and surfaces['refined']['symmetric_mean_distance_world']<=prior_surface['symmetric_mean_distance_world'])
        improvement=(old_l1-new_l1)/old_l1 if old_l1 else 0.
        selected_recipe='refined/program.json' if admitted else 'baseline/program.json'
        if prior and not admitted:
            # An extra acquisition can fail to improve. Retain the previous best,
            # rather than accidentally reverting to the original coarse baseline.
            import shutil
            publish('prior-selected/program.json',prior['refined']['program'])
            shutil.copyfile(prior_archive_path,owner.root/'prior-selected/evaluated-exact.npz')
            owner.register_file('prior-selected/evaluated-exact.npz','final_output')
            selected_recipe='prior-selected/program.json'
        receipt={'protocol':frozen['protocol'],'status':'measured','aggregate_accepted':False,
            'baseline':{'program':baseline_program.to_dict(),'geometry_hash':baseline_arrays.content_hash,'heldout':baseline_metrics,'surface':surfaces['baseline'],'topology_screen':solid_guard(baseline_arrays)},
            'refined':{'program':refined_program.to_dict(),'geometry_hash':refined_arrays.content_hash,'heldout':refined_metrics,'surface':surfaces['refined'],'topology_screen':solid_guard(refined_arrays)},
            'crop_request':request,'crop_resolution':[args.crop_resolution,args.crop_resolution],'source_frames':1 if args.reuse_source else 6,'reused_source_frames':5 if args.reuse_source else 0,'candidate_frames':2 if args.reuse_baseline else 7,
            'baseline_frames_reused':5 if args.reuse_baseline else 0,'baseline_raw_surface_reused':bool(prior),
            'prior_selected':prior['refined'] if prior else None,
            'relative_coverage_l1_reduction_vs_prior':(prior_l1-new_l1)/prior_l1 if prior_l1 else 0.,
            'continuation_admission':'strict heldout gates, alpha L1, minimum boundary and raw mean surface no worse than prior1024 selected candidate' if prior else None,
            'timings_seconds':timings,'elapsed_seconds':time.monotonic()-start,'crop_admitted':bool(admitted),
            'relative_heldout_coverage_l1_reduction':improvement,'meaningful_win':bool(admitted and improvement>=.1),
            'selected_recipe':selected_recipe,
            'family_surface_limits':None,'heldout_selection_scope':'one predeclared strategy comparison; no fit/ROI access to heldout pixels'}
        publish('results.json',receipt)
        print('ADAPTIVE_RESULT='+str(owner.root/'results.json'),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
