"""Saved-artifact inspection only; stage/count/elapsed and five-second heartbeats."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse, hashlib, json, os, sys, time
import numpy as np
ROOT=Path(os.environ.get('BLENDSLOP_REPO_ROOT', Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.evaluation.comparable_geometry import read_obj
from blender_blocking.evaluation.triangle_contacts import verify_reported_pairs, within_part_boundary_guard
from blender_blocking.reconstruction.native_geometry import GeometryArrays
from blender_blocking.reconstruction.grouped_solids import solid_guard


def reference_intersects(left,right):
    """Independent exact edge/triangle plane clipping, not separating axes."""
    def vec(a,b):return tuple(a[i]-b[i] for i in range(3))
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    a,b=[[tuple(Fraction.from_float(float(x)) for x in v) for v in t] for t in (left,right)]
    normals=[cross(vec(t[1],t[0]),vec(t[2],t[0])) for t in (a,b)]
    if not all(any(n) for n in normals):return None
    def inside(p,t,n):return all(dot(cross(vec(t[(i+1)%3],t[i]),vec(p,t[i])),n)>=0 for i in range(3))
    for t,other,n in ((a,b,normals[1]),(b,a,normals[0])):
        for i,p in enumerate(t):
            q=t[(i+1)%3];dp,dq=dot(n,vec(p,other[0])),dot(n,vec(q,other[0]))
            if dp==0 and inside(p,other,n):return True
            if dp*dq<0:
                cut=tuple((p[k]*dq-q[k]*dp)/(dq-dp) for k in range(3))
                if inside(cut,other,n):return True
    if any(cross(*normals)):return False
    if dot(normals[0],vec(b[0],a[0]))!=0:return False
    axes=[k for k in range(3) if k!=max(range(3),key=lambda k:abs(normals[0][k]))]
    def orient(p,q,r):return (q[axes[0]]-p[axes[0]])*(r[axes[1]]-p[axes[1]])-(q[axes[1]]-p[axes[1]])*(r[axes[0]]-p[axes[0]])
    def on(p,q,r):return orient(p,q,r)==0 and all(min(p[k],q[k])<=r[k]<=max(p[k],q[k]) for k in axes)
    for i,p in enumerate(a):
        q=a[(i+1)%3]
        for j,r in enumerate(b):
            s=b[(j+1)%3]
            if orient(p,q,r)*orient(p,q,s)<0 and orient(r,s,p)*orient(r,s,q)<0:return True
            if on(p,q,r) or on(p,q,s) or on(r,s,p) or on(r,s,q):return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import open3d as o3d
    rows=json.loads(args.exports_receipt.read_text())['mesh_exports']
    started=time.monotonic();last_heartbeat=started
    record={'schema':'saved_solid_pair_classification_v1','source_revision':'post-151408f uncommitted candidate',
        'input_receipt_sha256':hashlib.sha256(args.exports_receipt.read_bytes()).hexdigest(),
        'classifier_sha256':hashlib.sha256((ROOT/'blender_blocking/evaluation/triangle_contacts.py').read_bytes()).hexdigest(),
        'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'open3d':o3d.__version__,'rows':[],'reconstruction':False,'quality_or_performance_campaign':False}
    guard_cache={};counts=Counter();pair_counts=Counter();labels=Counter();independent=0
    def heartbeat(done,total):
        nonlocal last_heartbeat
        now=time.monotonic()
        if now-last_heartbeat>=5:
            print(f'HEARTBEAT stage=exact_geometry completed={done}/{total} cases={len(record["rows"])}/{len(rows)} elapsed={now-started:.1f}s',flush=True)
            last_heartbeat=now
    def persist():
        record.update(primary_case_counts=dict(counts),pair_classification_counts=dict(pair_counts),overlapping_case_label_counts=dict(labels),
            independent_reference_checks=independent,elapsed_s=time.monotonic()-started,completed_cases=len(record['rows']),total_cases=len(rows))
        temporary=args.output/'result.json.tmp';temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');os.replace(temporary,args.output/'result.json')
    print(f'STAGE inventory cases={len(rows)} existing_Open3D={o3d.__version__}',flush=True)
    for index,row in enumerate(rows):
        q=row['boundary_qualification'];item={k:row[k] for k in ('case','variant','mode','selected_backend','representation','mesh')}
        item['historical_qualification']=q
        if q.get('manifold_validated') is True:item['primary_classification']='qualified_original_receipt_not_remeasured'
        else:
            path=Path(row['mesh']);item['obj_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            vertices,faces=read_obj(path);data=GeometryArrays.capture(vertices,faces)
            assert data.content_hash==q['geometry_content_hash'], ('saved geometry identity mismatch',path)
            item.update(geometry_content_hash=data.content_hash,triangle_count=len(faces))
            if q.get('status')=='unavailable':
                if data.content_hash not in guard_cache:guard_cache[data.content_hash]=solid_guard(data)
                guard=guard_cache[data.content_hash];item['solid_guard']=guard
                reasons=[]
                if not guard['valid_solid']:reasons.append('topology_or_volume_guard_failed')
                if len(faces)>60000:reasons.append('triangle_limit_exceeded')
                item['unqualified_reasons']=reasons or ['historical_unavailable_reason_unresolved']
                item['primary_classification']='unqualified_due_guard_or_bound';item['dependency_unavailable']=False
            else:
                mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
                pairs=np.asarray(mesh.get_self_intersecting_triangles());assert len(pairs)==q['self_intersections']
                print(f'STAGE verify case={index+1}/{len(rows)} {row["case"]}/{row["variant"]}/{row["mode"]} raw_pairs={len(pairs)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                exact=verify_reported_pairs(vertices,faces,pairs,max_pairs=None,progress=heartbeat)
                item['exact_pair_verification']=exact;pair_counts.update(exact['counts'])
                np.savez_compressed(args.output/(f'pairs-{index:03d}.npz'),pairs=pairs)
                choices={}
                for sample in exact['samples']:choices.setdefault((sample['classification'],sample['between_components']),sample)
                refs=[]
                for sample in choices.values():
                    a,b=sample['faces'];actual=reference_intersects(vertices[faces[a]],vertices[faces[b]])
                    expected=sample['relation']!='disjoint';assert actual is None or actual==expected, ('independent disagreement',row,sample)
                    refs.append({**sample,'reference_intersects':actual,'agreement':actual is not None and actual==expected,
                        'coordinates':vertices[faces[[a,b]]].tolist()});independent+=1
                item['independent_reference_pairs']=refs
                case_labels=[k for k,v in exact['counts'].items() if v]
                for label in case_labels:labels[label]+=1
                if row['variant']=='cpu_dvx' or exact['non_disjoint_pairs']==0:
                    part=within_part_boundary_guard(vertices,faces,timeout_s=15.,progress=heartbeat);item['within_part_guard']=part
                    if part.get('reason')=='within_part_boundary_defect':
                        if 'true_self_intersection' not in case_labels:case_labels.append('true_self_intersection');labels['true_self_intersection']+=1
                        a,b=part['first_blocking_pair']['faces'];actual=reference_intersects(vertices[faces[a]],vertices[faces[b]])
                        assert actual is True;part['independent_reference_intersects']=actual;independent+=1
                    if exact['non_disjoint_pairs']==0:
                        item['new_geometric_qualification']=bool(part['passed'] and mesh.is_edge_manifold(False) and mesh.is_vertex_manifold() and mesh.is_orientable())
                priority=('true_self_intersection','duplicate_or_coincident_surface','multipart_overlap','boundary_contact','numerical_false_positive','indeterminate_degenerate')
                item['primary_classification']=next((c for c in priority if c in case_labels),'unclassified');item['case_labels']=case_labels
        counts[item['primary_classification']]+=1;record['rows'].append(item);persist()
        print(f'CASE {index+1}/{len(rows)} {row["case"]}/{row["variant"]}/{row["mode"]} classification={item["primary_classification"]} elapsed={time.monotonic()-started:.1f}s',flush=True)
    record['complete']=True;persist()
    print('CLASSIFICATION_COMPLETE',json.dumps({k:record[k] for k in ('completed_cases','primary_case_counts','pair_classification_counts','independent_reference_checks','elapsed_s')}),flush=True)


if __name__=='__main__':main()
