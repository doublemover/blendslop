#!/usr/bin/env python3
"""No-render exact integration of declared shared-plane generic compilation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'blender_blocking'),str(ROOT)]
import test_runner  # Existing optional dependency locations before native imports.
EXPECTED='85dcaee9140c32afbbc5e6325bec2f3e0a7e2e27174c0f979a722fa5e4ed6012'
PROTOCOL='multipart_shared_plane_generic_integration_v1'
ARCHIVE=ROOT/'docs/quality-multipart-planar-join-continuation-20261009/native-evidence/candidate-evaluated-exact.npz'
PROGRAM=ARCHIVE.with_name('candidate-program.json')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_plan():
    archive,program=ARCHIVE,PROGRAM
    import numpy as np
    from reconstruction.native_geometry import GeometryArrays
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.multipart_planar_join import _relation
    _relation(retained_multipart_program(read(program)))
    with zipfile.ZipFile(archive) as retained:
        if sum(item.file_size for item in retained.infolist())>67108864:raise ValueError('archive exceeds bounded decompression')
    with np.load(archive,allow_pickle=False) as retained:
        exact=GeometryArrays.capture(retained['vertices'],retained['faces'])
    if exact.content_hash!=EXPECTED:raise ValueError('retained exact85dc identity changed')
    files=[archive,program,*[ROOT/name for name in (
        'scripts/run_multipart_shared_plane_integration_check.py',
        'blender_blocking/primitives/shape_program.py','blender_blocking/primitives/shape_program_compiler.py',
        'blender_blocking/reconstruction/multipart_planar_join.py','blender_blocking/reconstruction/multipart_family.py',
        'blender_blocking/reconstruction/frozen_family.py','blender_blocking/reconstruction/native_geometry.py',
        'blender_blocking/metrics/topology_receipt.py','blender_blocking/test_multipart_planar_join.py',
        'blender_blocking/test_shape_program_shared_plane.py','blender_blocking/test_runner.py',
        'scripts/run_bounded_owned_command.py','blender_blocking/utils/owned_process_supervisor.py',
        'blender_blocking/utils/run_ownership.py')]]
    return {'protocol':PROTOCOL,'expected_indexed_geometry_hash':EXPECTED,'program':str(program),'archive':str(archive),
        'input_sha256':{str(p):sha(p) for p in files},'work_seconds':30.,'join_seconds':5.,'threads':2,
        'rss_bytes':8589934592,'committed_bytes':8589934592,'frames':0,'fits':0,'raw_pairs':0,'qualifier_children':0,
        'compiler_options':{'bevel_modifier':False,'weighted_normals':False},
        'blender_version':[5,2,2],'blender_build_hash':'d13f752e3b9c',
        'scope':'exact generic/standalone replay and scoped live depth restoration only; existing measurements/selection/history unchanged'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    plan=read(args.plan)
    fixed={'protocol':PROTOCOL,'expected_indexed_geometry_hash':EXPECTED,'work_seconds':30.,'join_seconds':5.,
        'threads':2,'rss_bytes':8589934592,'committed_bytes':8589934592,'frames':0,'fits':0,'raw_pairs':0,
        'qualifier_children':0,'compiler_options':{'bevel_modifier':False,'weighted_normals':False},
        'program':str(PROGRAM),'archive':str(ARCHIVE),'blender_version':[5,2,2],'blender_build_hash':'d13f752e3b9c'}
    for key,value in fixed.items():
        if plan.get(key)!=value or type(plan.get(key)) is not type(value):raise ValueError('changed integration scope: '+key)
    if not 1<=len(plan['input_sha256'])<=32:raise ValueError('unbounded integration input inventory')
    if any(str(path) not in plan['input_sha256'] for path in (PROGRAM,ARCHIVE)):
        raise ValueError('required retained program/archive lacks frozen input identity')
    for path,digest in plan['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('frozen input changed: '+path)
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):raise ValueError('fresh output must stay in isolated temp/tasks')
    import bpy
    if (tuple(bpy.app.version)!=(5,2,2) or bpy.app.build_hash.decode('ascii')!='d13f752e3b9c'):
        raise RuntimeError('fixed Blender5.2.2/build identity required')
    import numpy as np
    from primitives.shape_program_compiler import compile_shape_program
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.multipart_planar_join import compile_shared_plane_multipart,shared_plane_receipt,shared_depth_coordinates
    from reconstruction.native_geometry import GeometryArrays,evaluated_arrays
    from utils.run_ownership import OwnedRun
    with zipfile.ZipFile(plan['archive']) as archive:
        if sum(item.file_size for item in archive.infolist())>67108864:raise ValueError('archive exceeds bounded decompression')
    with np.load(plan['archive'],allow_pickle=False) as saved:
        expected=GeometryArrays.capture(saved['vertices'],saved['faces'])
    if expected.content_hash!=EXPECTED:raise ValueError('saved exact85dc identity changed')
    program=retained_multipart_program(read(plan['program']))
    # Production preflight also independently refuses unsupported declarations.
    owner=OwnedRun(args.output.resolve(),producer='multipart_shared_plane_generic_integration',max_generated_bytes=8388608,
        shared_inputs={'plan':str(args.plan),'plan_sha256':sha(args.plan)})
    started=time.monotonic()
    receipt={'protocol':PROTOCOL,'status':'running','frames':0,'fits':0,'raw_pairs':0,'qualifier_children':0,
        'expected_indexed_geometry_hash':EXPECTED,'artist_limits':None,'aggregate_accepted':False,
        'runtime':{'blender':bpy.app.version_string,'python':sys.version,'python_executable':sys.executable}}
    with owner:
        def publish(name,value):
            path=owner.root/name;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            owner.register_file(name,'final_output')
        publish('frozen-workload.json',plan);publish('results.json',receipt)
        try:
            collection='SharedPlaneIntegration_'+owner.run_id
            compiled=compile_shape_program(program,collection_name=collection,bevel_modifier=False,weighted_normals=False)
            by_id={part.get('blendslop_shape_node_id'):part for part in compiled.objects}
            parts=[by_id[node.node_id] for node in program.root_nodes];obj=parts[0]
            observed=shared_plane_receipt(program,parts)
            for index,operand in enumerate(parts[1:]):
                modifier=obj.modifiers.new('ObservedExactUnion'+str(index+1),'BOOLEAN')
                modifier.operation,modifier.solver,modifier.object='UNION','EXACT',operand
                operand['blendslop_export_exclude']=True;operand.hide_render=True;operand.hide_set(True)
            bpy.context.view_layer.update()
            actual=evaluated_arrays(obj)
            receipt['generic_replay']={'actual_hash':actual.content_hash,'matches_expected_indexed':actual.content_hash==EXPECTED,
                'relation':observed,'collection_names':sorted({c.name for p in parts for c in p.users_collection}),
                'same_caller_collection':all(any(c.name==collection for c in p.users_collection) for p in parts),
                'live_exact_unions':len(obj.modifiers)==2 and all(m.type=='BOOLEAN' and m.solver=='EXACT' and m.operation=='UNION' for m in obj.modifiers)}
            if not all(receipt['generic_replay'][key] for key in ['matches_expected_indexed','same_caller_collection','live_exact_unions']):
                raise ValueError('generic compiled geometry/collection/live-union contract failed')
            standalone,sources,report=compile_shared_plane_multipart(program)
            standalone_data=evaluated_arrays(standalone)
            receipt['standalone_replay']={'actual_hash':standalone_data.content_hash,
                'matches_expected_indexed':standalone_data.content_hash==EXPECTED,'relation':report}
            if standalone_data.content_hash!=EXPECTED:raise ValueError('standalone exact85dc replay failed')
            arm=by_id[report['arm_node_id']]
            co=np.empty(len(arm.data.vertices)*3,np.float32);arm.data.vertices.foreach_get('co',co);before=co.reshape(-1,3).copy()
            def controls():
                return {'compiled_root_pointer':compiled.root_object.as_pointer(),'output_pointer':obj.as_pointer(),
                    'sources':[(p.as_pointer(),p.data.as_pointer(),[list(row) for row in p.matrix_world]) for p in parts],
                    'modifiers':[(m.name,m.type,m.operation,m.solver,m.object.as_pointer()) for m in obj.modifiers]}
            original=controls();depth=float(np.ptp(before[:,1].astype(float))*float(arm.scale.y))
            far=float(arm.location.y)+float(arm.scale.y)*.5
            try:
                changed=shared_depth_coordinates(before,depth_world=depth*1.05,frame_depth_world=float(arm.scale.y))
                arm.data.vertices.foreach_set('co',changed.ravel());arm.data.update();bpy.context.view_layer.update()
                edited=evaluated_arrays(obj)
                ratio=float(np.ptp(changed[:,1].astype(float))*float(arm.scale.y)/depth)
                far_after=float(arm.location.y)+float(arm.scale.y)*float(changed[:,1].max())
            finally:
                arm.data.vertices.foreach_set('co',before.ravel());arm.data.update();bpy.context.view_layer.update()
            restored=evaluated_arrays(obj)
            receipt['depth_edit']={'depth_ratio':ratio,'fixed_far_plane':far_after==far,
                'geometry_changed':edited.content_hash!=EXPECTED,'restored_hash':restored.content_hash,
                'exact_indexed_restoration':restored.content_hash==EXPECTED,'source_controls_restored':controls()==original,
                'original_controls':original,'restored_controls':controls()}
            edit=receipt['depth_edit'];edit['passed']=bool(abs(ratio-1.05)<=1e-5 and all(edit[key] for key in
                ['fixed_far_plane','geometry_changed','exact_indexed_restoration','source_controls_restored']))
            if not edit['passed']:raise ValueError('generic source depth edit/restore failed')
            np.savez_compressed(owner.root/'generic-exact.npz',vertices=actual.vertices,faces=actual.faces)
            owner.register_file('generic-exact.npz','final_output')
            receipt.update(status='integration_passed',elapsed_seconds=time.monotonic()-started)
            publish('results.json',receipt);return 0
        except BaseException as exc:
            receipt.update(status='failed',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-started)
            try:owner.mark_failed(receipt['error']);publish('results.json',receipt)
            except BaseException as secondary:exc.add_note('secondary receipt error: '+repr(secondary))
            raise


if __name__=='__main__':raise SystemExit(main())
