#!/usr/bin/env python3
"""Finish the admitted saved box with three axis views, one boundary, one edit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking'), str(ROOT/'scripts')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_file(root, relative):
    manifest = json.loads((root/'run-ownership.json').read_text())
    lease = json.loads((root/'run-lease.json').read_text())
    if manifest.get('state') != 'succeeded' or lease.get('status') != 'released':
        raise ValueError('input run must be completed and released')
    bound = next(row for row in manifest['artifacts'] if row['path'] == relative)
    path = root/relative
    if path.stat().st_size > 33554432 or sha(path) != bound['sha256']:
        raise ValueError('retained input bytes changed: '+relative)
    return path


def main():
    import bpy
    import numpy as np
    import test_runner
    from evaluation.controlled_measurement import coverage_and_mask, measurement_signature, compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.native_qualification import qualify_geometry, toolchain_identity
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _compile_exact, physical_observation, response_contract, _pose_controls
    from run_surface_quality_check import _replay_orthographic_camera
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--source-measurements', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('output must remain in isolated project temp/tasks')
    experiment, source_root = args.experiment.resolve(strict=True), args.source_measurements.resolve(strict=True)
    receipt_path = retained_file(experiment, 'results.json')
    previous = json.loads(receipt_path.read_text())
    if previous.get('selected_recipe') != 'refined/program.json' or not previous.get('crop_admitted'):
        raise ValueError('only the already admitted refined recipe may be continued')
    wire_path = retained_file(experiment, 'refined/program.json')
    archive_path = retained_file(experiment, 'refined/evaluated-exact.npz')
    wire = json.loads(wire_path.read_text())
    with np.load(archive_path, allow_pickle=False) as archive:
        captured = GeometryArrays.capture(archive['vertices'], archive['faces'])
    if captured.content_hash != previous['refined']['geometry_hash']:
        raise ValueError('selected geometry identity changed')
    cameras = json.loads(retained_file(experiment, 'frozen-workload.json').read_text())['original_camera_declarations']
    sources = {}
    for view in ('front', 'side', 'top'):
        meta_path = retained_file(source_root, 'source-global/'+view+'-measurement.json')
        coverage_path = retained_file(source_root, 'source-global/'+view+'-coverage.npy')
        retained_file(source_root, 'source-global/'+view+'.exr')
        meta = json.loads(meta_path.read_text())
        if measurement_signature(meta['contract']) != meta['contract_sha256']:
            raise ValueError('source measurement signature changed')
        coverage, mask = coverage_and_mask(np.load(coverage_path, allow_pickle=False))
        sources[view] = {**meta, 'coverage':coverage, 'mask':mask}
    identity = toolchain_identity(args.python)
    started = time.monotonic()
    owner = OwnedRun(args.output.resolve(), producer='admitted_box_followup', max_generated_bytes=67108864,
        shared_inputs={'experiment':str(experiment),'source_measurements':str(source_root),'python':str(args.python)})
    with owner:
        def publish(relative, value, category='final_output'):
            path = owner.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
            owner.register_file(relative, category)
        frozen = {'protocol':'admitted_box_followup_v1','experiment_sha256':sha(receipt_path),
            'recipe_sha256':sha(wire_path),'archive_sha256':sha(archive_path),'geometry_hash':captured.content_hash,
            'toolchain_identity':identity,'renders':3,'fits':0,'raw_surface_recomputations':0,
            'qualification_children':1,'helper_timeout_seconds':15.,'semantic_edit':'live world corner radius x1.10; exact restore',
            'deadline_seconds':60.,'threads':2,'rss_limit_bytes':8*1024**3,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'blender_blocking/integration/blender_ops/measurement_render.py',ROOT/'blender_blocking/primitives/shape_program_compiler.py',ROOT/'scripts/run_family_source_edit_check.py')}}
        publish('frozen-workload.json', frozen, 'diagnostic')
        compiled, source, baseline = _compile_exact(wire, captured)
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.data.type = 'ORTHO'
        bpy.context.scene.camera = camera
        gates = SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
        metrics = {}
        with controlled_measurement_session(target_objects=[compiled.root_object], camera=camera, resolution=(512,512), samples=64) as session:
            for view in ('front','side','top'):
                _replay_orthographic_camera(camera, cameras[view])
                record = render_controlled_measurement(session, owner.root/(view+'.exr'))
                owner.register_file(view+'.exr','diagnostic')
                np.save(owner.root/(view+'-coverage.npy'),record['coverage'],allow_pickle=False)
                owner.register_file(view+'-coverage.npy','final_output')
                publish(view+'-measurement.json',{key:value for key,value in record.items() if key not in ('coverage','mask')})
                metrics[view] = compare_controlled_measurements(sources[view], record, view=view, config=gates)
        root_pointer, source_pointer, mesh_pointer = compiled.root_object.as_pointer(), source.as_pointer(), source.data.as_pointer()
        pose = _pose_controls(source)
        bevels = [m for m in source.modifiers if m.type == 'BEVEL']
        if len(bevels) != 1 or bevels[0].segments != 8:
            raise ValueError('saved eight-segment live bevel is required')
        bevel = bevels[0]
        width = float(bevel.width)
        before = physical_observation('rounded_box', baseline, source)
        try:
            bevel.width = width*1.10
            bpy.context.view_layer.update()
            changed = evaluated_arrays(compiled.root_object)
            after = physical_observation('rounded_box', changed, source)
            response = response_contract('rounded_box', before, after, wire['root_nodes'][0]['parameters'])
        finally:
            bevel.width = width
            bpy.context.view_layer.update()
        restored = evaluated_arrays(compiled.root_object)
        edits = {'before':before,'after':after,'response':response,'changed_geometry_hash':changed.content_hash,
            'restored_geometry_hash':restored.content_hash,'exact_restore':restored.content_hash == captured.content_hash,
            'geometry_changed':changed.content_hash != captured.content_hash,
            'same_root':compiled.root_object.as_pointer() == root_pointer,'same_source':source.as_pointer() == source_pointer,
            'same_source_mesh':source.data.as_pointer() == mesh_pointer,'pose_restored':_pose_controls(source) == pose,
            'modifier_restored':float(bevel.width) == width and bevel.segments == 8}
        edits['passed'] = response['passed'] and all(edits[k] for k in ('exact_restore','geometry_changed','same_root','same_source','same_source_mesh','pose_restored','modifier_restored'))
        if time.monotonic()-started > 40.:
            raise TimeoutError('outer allowance insufficient for unchanged15 second helper')
        boundary = qualify_geometry(captured, python=args.python, timeout_s=15., ownership_root=args.output.resolve()/'qualification-children')
        if boundary.get('geometry_content_hash') != captured.content_hash or boundary.get('toolchain_identity') not in (None,identity):
            raise ValueError('native boundary receipt identity changed')
        all_views = {**metrics, **previous['refined']['heldout']}
        passed = all(row['passed'] for row in all_views.values()) and edits['passed'] and boundary.get('single_solid_qualified',False)
        result = {'protocol':frozen['protocol'],'geometry_hash':captured.content_hash,'all_five_silhouette_views':all_views,
            'boundary':boundary,'semantic_edit':edits,'status':'independent_checks_passed' if passed else 'required_check_failed',
            'family_surface_limits':None,'raw_surface':previous['refined']['surface'],'raw_surface_reused':True,
            'aggregate_accepted':False,'scope':'three newly measured axis views plus retained independent obliques; boundary and edit verdicts remain independent; family surface tolerance absent',
            'elapsed_seconds':time.monotonic()-started}
        publish('results.json',result)
        if not passed:owner.mark_failed('required followup check failed')
        print('BOX_FOLLOWUP_RESULT='+str(owner.root/'results.json'),flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
