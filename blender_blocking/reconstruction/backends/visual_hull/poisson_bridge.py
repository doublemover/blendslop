"""Explicit optional CPU helper bridge: fixed files, own capped child, no installs."""
from __future__ import annotations
from pathlib import Path
import json,os,subprocess,sys,time,uuid
import numpy as np


def run_external_poisson(mesh_result,method,config):
    from volume import MeshExtractionResult
    python=Path(config['external_open3d_python']).resolve()
    if not python.is_file(): raise ValueError('configured Open3D helper Python is missing')
    root=Path(config['postprocess_artifact_root']).resolve()/('cpu-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True,exist_ok=False)
    np.savez(root/'input.npz',vertices=mesh_result.vertices,faces=mesh_result.faces)
    (root/'config.json').write_text(json.dumps(dict(config),default=str))
    script=Path(__file__).resolve().parents[4]/'scripts'/'open3d_cpu_worker.py'
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    started=time.perf_counter()
    with (root/'worker.log').open('w') as log:
        child=subprocess.Popen([str(python),str(script),'--root',str(root),'--method',method],stdout=log,stderr=subprocess.STDOUT,
            env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try: code=child.wait(timeout=min(90.,float(config.get('poisson_timeout_s',90.))))
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise RuntimeError('owned Open3D child exceeded explicit timeout; log retained')
    if code!=0: raise RuntimeError(f'Open3D helper failed ({code}); log: {root / "worker.log"}')
    arrays=np.load(root/'output.npz',allow_pickle=False)
    metadata=json.loads((root/'result.json').read_text())
    return MeshExtractionResult(status='ok',method='screened_poisson_external_open3d',requested_method=mesh_result.method,
        vertices=arrays['vertices'],faces=arrays['faces'],normals=arrays['normals'],topology=metadata['topology'],
        metrics={**metadata['metrics'],'external_helper':metadata,'external_wall_s':time.perf_counter()-started,'bridge_artifacts':str(root)},
        message='explicit configured Open3D CPU helper completed')
