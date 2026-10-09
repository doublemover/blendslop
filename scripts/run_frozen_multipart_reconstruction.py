#!/usr/bin/env python3
"""One bounded observed multipart preparation; no renders or qualifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'blender_blocking'),str(ROOT/'scripts'),str(ROOT)]
import test_runner


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def part_controls(parts, output, baseline):
    """Specific per-part dimension response, same objects and exact restoration."""
    import bpy
    import numpy as np
    from reconstruction.native_geometry import evaluated_arrays
    records=[]
    for part,axis,dimension in zip(parts,(0,2,1),('base width','tall arm height','short arm depth')):
        old_scale=part.scale.copy(); pointer=part.as_pointer(); mesh_pointer=part.data.as_pointer()
        excluded=bool(part.get('blendslop_export_exclude',False))
        part['blendslop_export_exclude']=False
        before=evaluated_arrays(part)
        part.scale[axis]*=1.05; bpy.context.view_layer.update()
        changed=evaluated_arrays(part); union=evaluated_arrays(output)
        ratio=float(np.ptp(changed.vertices[:,axis])/np.ptp(before.vertices[:,axis]))
        part.scale=old_scale; bpy.context.view_layer.update()
        part['blendslop_export_exclude']=excluded
        restored=evaluated_arrays(output)
        exact=restored.content_hash==baseline.content_hash
        same=part.as_pointer()==pointer and part.data.as_pointer()==mesh_pointer
        passed=abs(ratio-1.05)<=1e-5 and union.content_hash!=baseline.content_hash and exact and same
        records.append({'control':dimension+' x 1.05','node':part.get('blendslop_shape_node_id',part.name),
                        'dimension_ratio':ratio,'union_geometry_changed':union.content_hash!=baseline.content_hash,
                        'exact_restoration':exact,'same_source_and_mesh_pointers':same,'passed':bool(passed),
                        'baseline_geometry_hash':baseline.content_hash,'edited_geometry_hash':union.content_hash,
                        'restored_geometry_hash':restored.content_hash})
    return {'status':'passed' if all(r['passed'] for r in records) else 'failed','transactions':records,
            'scope':'three live per-part dimensions through two Exact unions; no blanket artist-control qualification'}


def compile_live_multipart(program):
    """Three unchanged editable leaves and exactly two live Exact union modifiers."""
    import bpy
    from primitives.shape_program_compiler import compile_shape_program
    if len(program.root_nodes)!=3 or any(node.operation!='add' or node.primitive_type!='box' or node.children for node in program.root_nodes):
        raise ValueError('live multipart adapter accepts exactly three additive box leaves')
    compiled=compile_shape_program(program,bevel_modifier=False,weighted_normals=False)
    by_node={part.get('blendslop_shape_node_id'):part for part in compiled.objects}
    parts=[by_node[node.node_id] for node in program.root_nodes]
    obj=parts[0]
    for index,operand in enumerate(parts[1:]):
        modifier=obj.modifiers.new('ObservedExactUnion'+str(index+1),'BOOLEAN')
        modifier.operation='UNION'; modifier.solver='EXACT'; modifier.object=operand
        operand['blendslop_export_exclude']=True; operand.hide_render=True; operand.hide_set(True)
    bpy.context.view_layer.update()
    return obj,parts


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from primitives.shape_program_compiler import compile_shape_program
    from reconstruction.multipart_family import FIT_VIEWS,HELD_OUT_VIEW,fitted_multipart_program
    from reconstruction.coverage_evidence import coverage_from_grayscale
    from reconstruction.native_geometry import GeometryArrays,evaluated_arrays
    from reconstruction.grouped_solids import solid_guard
    from evaluation.canonical_artifacts import canonical_artifact_inventory,raw_surface_observation
    from synthetic.quality_references import quality_feature_verdict
    from utils.run_ownership import OwnedRun
    from run_surface_quality_check import _write
    from run_quality_coverage_check import save_mesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--coverage-transfer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    parent=args.output.resolve()
    if not parent.is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('owned output must stay within isolated temp/tasks')
    name='asymmetric_multipart_solid'; source=args.reference_root/name
    reference_receipt=args.reference_root/'results.json'
    cameras=json.loads(reference_receipt.read_text())['cases'][name]['reference_cameras']
    encoded=np.asarray(Image.open(args.coverage_transfer),float).reshape(-1)/65535
    if not 64<=len(encoded)<=8193: raise ValueError('explicit transfer sample count out of bounds')
    transfer=(np.linspace(0,1,len(encoded)),encoded)
    mask_records={}
    masks,coverage={},{}
    for view in (*FIT_VIEWS,HELD_OUT_VIEW):
        path=source/(view+'-mask.png'); digest=_sha(path)
        if digest!=cameras[view]['png_sha256']: raise ValueError('frozen source mask changed')
        mask_records[view]={'path':str(path.resolve()),'sha256':digest,'fit_used':view in FIT_VIEWS}
        if view in FIT_VIEWS:
            image=np.asarray(Image.open(path).convert('L'))
            if image.shape!=(512,512): raise ValueError('original mask resolution changed')
            masks[view]=image<128
            coverage[view]=coverage_from_grayscale(image.astype(float)/255,*transfer)
    settings=args.coverage_transfer.with_name('transfer-settings.json')
    frozen={'protocol':'frozen_multipart_four_view_v1','fit_views':list(FIT_VIEWS),'held_out_view':HELD_OUT_VIEW,
            'held_out_scope':'pixels not decoded or used during fitting; SHA retained independently',
            'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in (Path(__file__),ROOT/'blender_blocking/reconstruction/multipart_family.py',ROOT/'blender_blocking/primitives/shape_program_compiler.py',ROOT/'blender_blocking/evaluation/canonical_artifacts.py',ROOT/'scripts/run_quality_coverage_check.py')},
            'observed_masks':mask_records,'reference_receipt':{'path':str(reference_receipt),'sha256':_sha(reference_receipt)},
            'cameras':cameras,'coverage_transfer':{'path':str(args.coverage_transfer),'sha256':_sha(args.coverage_transfer),'settings_sha256':_sha(settings),'settings':json.loads(settings.read_text())},
            'budget':{'residual_calls':160,'fit_seconds':3.,'editable_boxes':3,'live_exact_union_modifiers':2,'outer_seconds':120,'rss_bytes':8*1024**3,'threads':2},
            'renders':0,'qualification_children':0,'helper_executor':False,'sdf_fallback':False,
            'surface':{'samples_per_direction':4096,'seed':61007,'qualified_limits':None},
            'resolution':[512,512],'measurement':'original display-calibrated AgX workload; no controlled-alpha relabeling',
            'environment':{'blender':bpy.app.version_string,'python':sys.version}}
    owner=OwnedRun(parent,producer='frozen_multipart_reconstruction',max_generated_bytes=67108864,
                   shared_inputs={'reference_root':str(args.reference_root),'scope':'four observed fit views; fifth held out'})
    started=time.monotonic()
    with owner:
        output=owner.root
        _write(output/'frozen-workload.json',frozen)
        try:
            program=fitted_multipart_program(masks,cameras,coverage_masks=coverage,max_evaluations=160,max_elapsed_s=3.)
            _write(output/'program.json',program.to_dict())  # Before any GT geometry read.
            obj,parts=compile_live_multipart(program)
            data,hashes=save_mesh(obj,output)
            control=part_controls(parts,obj,data)
            if evaluated_arrays(obj).content_hash!=data.content_hash:
                raise ValueError('per-part edits failed exact restoration before comparison')
            reference_npz=np.load(source/'evaluated-exact.npz',allow_pickle=False)
            reference=GeometryArrays.capture(reference_npz['vertices'],reference_npz['faces'])
            observation=raw_surface_observation(reference,data,count=4096,seed=61007)
            row={'status':'measured' if program.metadata['fit_status']=='locally_identified' else 'underconstrained',
                 **hashes,'artifact_directory':str(output),'program':program.to_dict(),'surface_observation':observation,
                 'features':quality_feature_verdict(name,data),'editability':control,
                 'topology':{'status':'unqualified','screen':solid_guard(data),'boundary_qualified':False},
                 'silhouette':{'status':'unrun','reason':'geometry-only phase; held-out admission remains untouched'},
                 'inspection_artifacts':canonical_artifact_inventory(output,geometry_hash=data.content_hash,camera_records={},pass_states={'mask':'unrun','neutral':'unrun','normals':'unrun'}),
                 'union_modifiers':[{'name':m.name,'operation':m.operation,'solver':m.solver} for m in obj.modifiers if m.type=='BOOLEAN'],
                 'aggregate_accepted':False}
            receipt={'protocol':frozen['protocol'],'status':'actual_row_retained','cases':{name:row},'elapsed_seconds':time.monotonic()-started,'aggregate_accepted':False,'run_root':str(output)}
            _write(output/'results.json',receipt)
            for path in output.iterdir():
                if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                    owner.register_file(path.relative_to(output),'final_output' if path.suffix in ('.json','.npz','.obj') else 'diagnostic')
        except Exception as exc:
            _write(output/'results.json',{'protocol':frozen['protocol'],'status':'failed',
                   'cases':{name:{'status':'failed','reason':type(exc).__name__+': '+str(exc),'aggregate_accepted':False}},
                   'run_root':str(output),'elapsed_seconds':time.monotonic()-started,'aggregate_accepted':False})
            for path in output.iterdir():
                if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                    owner.register_file(path.relative_to(output),'diagnostic')
            raise
        print('MULTIPART_RESULT='+str(output/'results.json'),flush=True)
        return 0 if control['status']=='passed' and row['topology']['screen']['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
