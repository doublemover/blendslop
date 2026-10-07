"""Reproducible sample bundles and metric-aligned mesh diagnostics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import PROFILES, EVALUATOR_VERSION, evaluate_points, shared_transform
from blender_blocking.evaluation.comparable_geometry import read_obj, sample_surface


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_sample_bundle(directory, result, *, case_id, seed, original_units, source_hashes,
                       camera_hashes=None, mask_hashes=None, evaluator_revision=None):
    if not original_units or not source_hashes:
        raise ValueError("units and source hashes are required")
    out=Path(directory);out.mkdir(parents=True,exist_ok=False)
    import sys
    import scipy
    receipt={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    receipt.update(schema="metric_case_bundle_v1",case_id=case_id,seed=int(seed),
                   original_units=original_units,source_hashes=dict(source_hashes),
                   camera_hashes=dict(camera_hashes or {}),mask_hashes=dict(mask_hashes or {}),
                   evaluator_version=EVALUATOR_VERSION,evaluator_revision=evaluator_revision or EVALUATOR_VERSION,
                   dependencies={"numpy":np.__version__, "scipy":scipy.__version__, "python":sys.version}, failures=[] if result['status']=='available' else [result.get('reason')])
    if result['status']=='available':
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        arrays['reference_sample_ids']=np.arange(len(result['reference_points']),dtype=np.int64)
        arrays['prediction_sample_ids']=np.arange(len(result['prediction_points']),dtype=np.int64)
        np.savez_compressed(out/'samples.npz',**arrays)
        receipt['samples_sha256']=sha256(out/'samples.npz')
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def reload_sample_bundle(directory):
    out=Path(directory);m=json.loads((out/'manifest.json').read_text())
    if m['status']!='available': return m
    if sha256(out/'samples.npz') != m['samples_sha256']:
        raise ValueError("sample bundle hash mismatch")
    with np.load(out/'samples.npz',allow_pickle=False) as source:
        arrays={k:source[k].copy() for k in source.files}
    return {**m,**arrays}


def evaluate_mesh_profile(reference_path,prediction_path,*,profile,seed=1234,reference_fps=None,
                          provided_transform=None,reference_metadata=None):
    c=PROFILES[profile]
    if profile=='legacy_bbox_v1':
        from blender_blocking.evaluation.comparable_geometry import compare_meshes
        return compare_meshes(reference_path,prediction_path,seed=seed)
    rv,rf=read_obj(reference_path);pv,pf=read_obj(prediction_path)
    if c.native and profile.startswith('superflex') and reference_fps is None:
        return {"status":"unavailable","profile":profile,"reason":"released 4096 reference FPS points missing"}
    matrix=provided_transform
    if matrix is None and c.frame=='target_bbox':matrix=shared_transform(rv,c.extent)
    rp=reference_fps if reference_fps is not None else sample_surface(rv,rf,count=c.reference_count,seed=seed)[0]
    pp=sample_surface(pv,pf,count=c.prediction_count,seed=seed)[0]
    result=evaluate_points(rp,pp,profile=profile,matrix=matrix,reference_metadata=reference_metadata)
    result['sampling']={"reference":"released_fps" if reference_fps is not None else "area_weighted_triangles",
                        "prediction":"area_weighted_triangles","seed":seed,
                        "reference_count":c.reference_count,"prediction_count":c.prediction_count}
    result['source_hashes']={"reference":sha256(reference_path),"prediction":sha256(prediction_path)}
    result['representation']=c.representation
    result['claim']='metric-aligned diagnostics on our cases; dataset reproduction requires native prepared data'
    return result
