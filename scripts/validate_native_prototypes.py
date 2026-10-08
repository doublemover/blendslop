"""Bounded native query and changed-algorithm SDF checks, outside the default route."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
from blender_blocking.reconstruction.native_geometry import GeometryArrays,GeometryCache,NativeOwnedGeometry
from blender_blocking.reconstruction.native_queries import raycast_batch,scalar_raycast,proximity_batch,orthographic_pixel_rays
from blender_blocking.reconstruction.native_csg import sdf_grid_mesh
from blender_blocking.evaluation.comparable_geometry import read_obj,occupancy
from blender_blocking.reconstruction.mesh_io import write_obj
from blender_blocking.evaluation.protocols.core import shared_transform,apply_transform
from blender_blocking.evaluation.protocols.bundles import evaluate_mesh_profile
from blender_blocking.evaluation.geometry import volumetric_iou
from blender_blocking.evaluation.solid_validity import solid_validity_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    args.output.mkdir(parents=True,exist_ok=False);results=[]
    for group,case in (('paired','torus'),('heldout','chair')):
        rows=json.loads((args.phase/group/'final.json').read_text())['rows']
        row=next(r for r in rows if r['case']==case and r['requested_mode']=='visual_hull_voxel')
        payload=json.loads(Path(row['result_path']).read_text());path=Path(payload['mesh_path'])
        data=GeometryArrays.capture(*read_obj(path));cache=GeometryCache()
        extent=float(np.ptp(data.vertices,axis=0).max());lo=data.vertices.min(axis=0);hi=data.vertices.max(axis=0)
        origins,directions=orthographic_pixel_rays([lo[0],hi[0],lo[2],hi[2]],32,32,
                         horizontal_axis=0,vertical_axis=2,depth_axis=1,depth=lo[1]-extent,direction_sign=1)
        started=time.perf_counter();scalar=scalar_raycast(data,origins,directions,far=extent*3,cache=cache);scalar_s=time.perf_counter()-started
        owner=NativeOwnedGeometry(data,'PrototypeTarget')
        try:
            started=time.perf_counter();native=raycast_batch(owner,origins,directions,far=extent*3);native_s=time.perf_counter()-started
            same=np.array_equal(native['native_hit'],scalar['native_hit'])
            valid=native['native_hit']&scalar['native_hit']
            error=float(np.max(np.abs(native['native_distance'][valid]-scalar['native_distance'][valid]))) if valid.any() else 0
            proximity=proximity_batch(owner,data.vertices[::max(1,len(data.vertices)//1024)])
        finally:owner.release()
        results.append({'case':case,'kind':'native_queries','same_hit_masks':same,'max_distance_error':error,
                        'scalar_s':scalar_s,'native_s':native_s,'timing_repetitions':1,'parallel_speed_claim':False,
                        'proximity_surface_max_distance':float(np.max(proximity['native_distance']))})
        (args.output/'result.json').write_text(json.dumps({'rows':results},indent=2),encoding='utf-8')
        if not same or error>extent*1e-5:raise RuntimeError('native query fixture mismatch')
        reference_manifest=json.loads((args.phase/group/'references'/case/'manifest.json').read_text())
        from PIL import Image
        masks={v:np.asarray(Image.open(args.phase/group/'references'/case/(v+'.png')).convert('RGBA'))[:,:,3]>127 for v in ('front','side','top')}
        from blender_blocking.reconstruction.targets import make_reconstruction_target
        from blender_blocking.reconstruction.types import CandidateRequest,CandidateResult,CandidateMetrics
        from blender_blocking.reconstruction.measured_selection import render_evidence
        from types import SimpleNamespace
        target=make_reconstruction_target(masks_by_view=masks)
        from dataclasses import replace
        target=replace(target,extras={**target.extras,'view_calibration':reference_manifest['calibration']})
        for resolution in (128,256):
            started=time.perf_counter();changed,receipt=sdf_grid_mesh(data,data,operation='UNION',resolution=resolution)
            cell=args.output/case/str(resolution);cell.mkdir(parents=True)
            mesh=write_obj(cell/'sdf.obj',{'vertices':changed.vertices,'faces':changed.faces})
            solid=solid_validity_report(changed.vertices,changed.faces)
            matrix=shared_transform(data.vertices,1)
            volume=volumetric_iou(occupancy(apply_transform(data.vertices,matrix),data.faces),
                                 occupancy(apply_transform(changed.vertices,matrix),changed.faces))
            owned=NativeOwnedGeometry(changed,'SDFEvidence')
            try:
                request=CandidateRequest('sdf','experimental_sdf',target,{},artifact_root=cell,
                         context=SimpleNamespace(cost_recorder=None))
                candidate=CandidateResult('sdf','experimental_sdf','success',mesh_path=mesh,
                         metric_result=CandidateMetrics(),geometry=owned)
                rendered=render_evidence(candidate,request)
                view_metrics=rendered.metric_result.per_view
            finally:owned.release()
            geometry=evaluate_mesh_profile(path,mesh,profile='common_surface_v1',seed=row['seed'])
            results.append({'case':case,'kind':'changed_algorithm','receipt':receipt,'vertices':len(changed.vertices),
                  'faces':len(changed.faces),'solid':solid,'common_frame_cell_parity_iou_24':volume,
                  'views':view_metrics,'geometry':geometry['metrics'],'wall_s':time.perf_counter()-started,
                  'editability':'dense native triangle mesh; no primitive editability claim',
                  'self_intersection':'pending bounded native external screen; structural watertightness alone is insufficient'})
            (args.output/'result.json').write_text(json.dumps({'rows':results},indent=2),encoding='utf-8')
    print('Native prototypes completed; changed-algorithm rows remain separate',flush=True)


if __name__=='__main__':main()
