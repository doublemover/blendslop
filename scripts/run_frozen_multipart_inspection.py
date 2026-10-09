#!/usr/bin/env python3
"""Saved multipart inspection only: exact geometry, five views, fifteen frames."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'blender_blocking'),str(ROOT/'scripts'),str(ROOT)]
import test_runner


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from mathutils import Vector
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.native_geometry import GeometryArrays
    from evaluation.reference_noise import oriented_surface_identity
    from evaluation.canonical_artifacts import CANONICAL_VIEWS,canonical_artifact_inventory
    from evaluation.silhouette_eval import SilhouetteGateConfig,evaluate_silhouette_pair
    from utils.run_ownership import OwnedRun
    from run_frozen_multipart_reconstruction import _sha,compile_live_multipart
    from run_surface_quality_check import _write
    from run_quality_coverage_check import render_masks,save_mesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('output outside isolated temp/tasks')
    name='asymmetric_multipart_solid'
    prepared=json.loads(args.prepared_run.read_text()); prior=prepared['cases'][name]
    folder=Path(prior['artifact_directory']); frozen=json.loads((folder/'frozen-workload.json').read_text())
    for filename,key in [('evaluated-exact.npz','npz_sha256'),('evaluated.obj','obj_sha256')]:
        if _sha(folder/filename)!=prior[key]: raise ValueError('saved multipart artifact changed')
    saved=json.loads((folder/'program.json').read_text())
    if saved!=prior['program']: raise ValueError('saved multipart program changed')
    source=np.load(folder/'evaluated-exact.npz',allow_pickle=False)
    captured=GeometryArrays.capture(source['vertices'],source['faces'])
    if captured.content_hash!=prior['geometry_hash']: raise ValueError('saved multipart indexed geometry changed')
    masks={}
    for view in CANONICAL_VIEWS:
        record=frozen['observed_masks'][view]; path=Path(record['path'])
        if _sha(path)!=record['sha256']: raise ValueError('source admission mask changed')
        image=np.asarray(Image.open(path).convert('L'))
        if image.shape!=(512,512): raise ValueError('source resolution changed')
        masks[view]=image<128
    owner=OwnedRun(args.output.resolve(),producer='frozen_multipart_inspection',max_generated_bytes=67108864,
                   shared_inputs={'prepared_run':str(args.prepared_run),'scope':'geometry-preserving saved-only admission'})
    started=time.monotonic()
    with owner:
        output=owner.root
        provenance={'protocol':'frozen_multipart_saved_inspection_v1','prepared_receipt':str(args.prepared_run),
                    'prepared_receipt_sha256':_sha(args.prepared_run),'original_geometry_hash':captured.content_hash,
                    'original_program_sha256':_sha(folder/'program.json'),'source_sha256':{str(p.relative_to(ROOT)):_sha(p) for p in (Path(__file__),ROOT/'scripts/run_frozen_multipart_reconstruction.py',ROOT/'blender_blocking/reconstruction/multipart_family.py',ROOT/'scripts/run_quality_coverage_check.py',ROOT/'scripts/run_surface_quality_check.py',ROOT/'blender_blocking/evaluation/canonical_artifacts.py',ROOT/'blender_blocking/primitives/shape_program_compiler.py')},
                    'cameras':frozen['cameras'],'observed_masks':frozen['observed_masks'],'resolution':[512,512],
                    'measurement':'preserved frozen display-calibrated AgX protocol; no controlled alpha relabeling',
                    'render_frames':15,'passes':['mask','neutral','normals'],'fit_evaluations':0,'raw_recomputations':0,
                    'qualification_children':0,'budget_seconds':60,'budget_rss_bytes':8*1024**3,'blender_threads':2}
        _write(output/'frozen-workload.json',provenance)
        try:
            program=retained_multipart_program(saved); obj,parts=compile_live_multipart(program)
            data,hashes=save_mesh(obj,output)
            if data.content_hash!=captured.content_hash:
                _write(output/'replay-identity-diagnostic.json',{'required_hash':captured.content_hash,'replayed_hash':data.content_hash,
                    'vertices_exact':bool(np.array_equal(captured.vertices,data.vertices)),'faces_exact':bool(np.array_equal(captured.faces,data.faces)),
                    'source_oriented_surface':oriented_surface_identity(captured),'replayed_oriented_surface':oriented_surface_identity(data),
                    'admission':'blocked before frames; diagnostics do not override indexed identity'})
                for index,part in enumerate(parts):
                    mesh=part.data; mesh.calc_loop_triangles()
                    np.savez_compressed(output/('replay-local-'+str(index)+'.npz'),vertices=np.asarray([tuple(v.co) for v in mesh.vertices],float),faces=np.asarray([tuple(f.vertices) for f in mesh.loop_triangles],int))
                _write(output/'replay-poses.json',{'poses':[[list(row) for row in part.matrix_world] for part in parts]})
                raise ValueError('saved multipart Boolean replay changed exact indexed geometry')
            candidate,cameras=render_masks(obj,output,(Vector(data.vertices.min(axis=0)),Vector(data.vertices.max(axis=0))),CANONICAL_VIEWS,
                        camera_records=frozen['cameras'],inspection_passes=('mask','neutral','normals'))
            gates=SilhouetteGateConfig(min_area_iou=.7,min_boundary_iou=.8,max_signed_distance_loss=.05)
            views={view:evaluate_silhouette_pair(masks[view],candidate[view],view=view,config=gates) for view in CANONICAL_VIEWS}
            inventory=canonical_artifact_inventory(output,geometry_hash=data.content_hash,camera_records=cameras,
                    reference_camera_records=frozen['cameras'],pass_states={'mask':'completed','neutral':'completed','normals':'completed'})
            _write(output/'camera-snapshots.json',cameras); _write(output/'program.json',saved)
            interval=saved['metadata']['identifiability']['original_short_far_interval']
            endpoint=saved['metadata']['identifiability']['fitted_short_far']; pixel=float(frozen['cameras']['side']['ortho_scale'])/512
            active=min(endpoint-interval[0],interval[1]-endpoint)<=pixel*.25
            row={**prior,**hashes,'artifact_directory':str(output),'prepared_receipt':str(args.prepared_run),
                 'silhouette':{'status':'passed' if all(x['passed'] for x in views.values()) else 'failed','views':views},
                 'inspection_artifacts':inventory,'fit_uncertainty':{'status':'underconstrained' if active else 'local_rank_only',
                    'original_short_far_interval':interval,'fitted_endpoint':endpoint,'interval_bound_active':active,
                    'scope':'saved local rank does not qualify an endpoint at its interval bound; no fitting performed'},
                 'aggregate_accepted':False}
            row['status']='failed' if row['silhouette']['status']=='failed' else 'incomplete'
            receipt={'protocol':provenance['protocol'],'status':'actual_inspection_retained','cases':{name:row},'run_root':str(output),'elapsed_seconds':time.monotonic()-started,'aggregate_accepted':False}
            _write(output/'results.json',receipt)
        except Exception as exc:
            _write(output/'results.json',{'protocol':provenance['protocol'],'status':'failed','cases':{name:{'status':'failed','reason':type(exc).__name__+': '+str(exc),'aggregate_accepted':False}},'run_root':str(output),'aggregate_accepted':False})
            raise
        finally:
            for path in output.iterdir():
                if path.is_file() and path.name not in ('run-ownership.json','run-lease.json'):
                    owner.register_file(path.relative_to(output),'final_output' if path.suffix in ('.json','.npz','.obj') else 'diagnostic')
        print('MULTIPART_INSPECTION='+str(output/'results.json'),flush=True)
        return int(row['silhouette']['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
