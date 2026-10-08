"""Poisson worker for an explicitly configured, already installed helper Python."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',default='screened_poisson');args=p.parse_args()
    import open3d as o3d
    from volume import MeshExtractionResult
    from reconstruction.backends.visual_hull.poisson import _run_open3d_poisson
    data=np.load(args.root/'input.npz',allow_pickle=False)
    source=MeshExtractionResult(status='ok',method='matched_source_mesh',vertices=data['vertices'],faces=data['faces'])
    config=json.loads((args.root/'config.json').read_text())
    started=time.perf_counter();result=_run_open3d_poisson(source,args.method,config)
    np.savez(args.root/'output.npz',vertices=result.vertices,faces=result.faces,normals=result.normals)
    (args.root/'result.json').write_text(json.dumps({'open3d':o3d.__version__,'python':sys.version,'binary':sys.executable,
        'source_sha256':hashlib.sha256((args.root/'input.npz').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((args.root/'config.json').read_bytes()).hexdigest(),
        'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy':np.__version__,'kernel_wall_s':time.perf_counter()-started,
        'metrics':result.metrics,'topology':result.topology},indent=2))
if __name__=='__main__':main()
