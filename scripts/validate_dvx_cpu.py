"""Opt-in bounded synthetic pinned-runtime contracts, never reconstruction scores."""
from pathlib import Path
import argparse
import importlib.util
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ROOT / 'blender_blocking/reconstruction/differentiable'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approved', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.approved:
        parser.error('official dependency installation and execution must be approved')
    started = time.perf_counter()
    receipts = []

    def stage(label, operation):
        print(f'stage={label} completed={len(receipts)} elapsed={time.perf_counter()-started:.2f}s', flush=True)
        value = operation()
        receipts.append({'stage': label, 'result': value})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'synthetic_contracts_only': True,
            'quality_and_performance_gains': 'unmeasured', 'platform': platform.platform(),
            'python': sys.version, 'elapsed_s': time.perf_counter()-started,
            'receipts': receipts}, indent=2))

    adapter = load('blendslop_dvx_adapter_validation', NUMERIC / 'dvx_adapter.py')
    def qualify():
        result = adapter.tiny_gradient_check(approved=True)
        assert result['passed']
        return result
    stage('pinned_forward_backward', qualify)
    import numpy as np
    import torch
    import dvx.torch as dvx
    coordinates = load('blendslop_coordinates_validation', NUMERIC / 'mesh_conditioning.py')
    helper = load('blendslop_helper_validation', NUMERIC / 'helper_session.py')
    vertices = np.array([[-.4,-.4,-.4],[.5,-.4,-.4],[0.,.5,-.4],[0.,0.,.5]])
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]], np.int64)

    def pullbacks():
        gradient = np.arange(12).reshape(4,3)/13.
        result = {}
        for mode in ('cage','differential','vertices'):
            parameters, decode, report = coordinates.torch_coordinates(vertices, faces, mode=mode)
            output = decode()
            np.testing.assert_allclose(output.detach().numpy(), vertices, atol=2e-7)
            (output*torch.tensor(gradient,dtype=torch.float32)).sum().backward()
            expected = (coordinates.trilinear_cage(vertices).pullback(gradient) if mode=='cage'
                else coordinates.DifferentialCoordinates(vertices,faces).pullback(gradient)
                if mode=='differential' else gradient)
            np.testing.assert_allclose(parameters.grad.numpy(),expected,atol=2e-7)
            result[mode] = {**report, 'pullback_max_error': float(np.max(np.abs(parameters.grad.numpy()-expected)))}
        return result

    stage('torch_coordinate_pullbacks', pullbacks)
    shifted = torch.tensor(vertices+[.015,0.,0.],dtype=torch.float32)
    face_tensor = torch.tensor(faces,dtype=torch.int64)

    def target(n, rays=False):
        grid = dvx.voxelize(n,shifted,face_tensor).detach().numpy()
        row = {'target_zyx':grid,'target_valid_zyx':np.ones_like(grid)}
        if rays:
            row['ray_targets'] = {'synthetic_front':{'foreground':np.clip(grid,0.,1.).max(axis=1),
                'valid':np.ones((n,n)), 'depth_axis_zyx':1}}
        return row

    common = {'execution_approved':True, 'vertices':vertices,'faces':faces,
        'center':np.zeros(3),'scale':1.,'steps':1,'timeout_s':20.,'learning_rate':.001}

    def fit(mode, rays=False):
        payload = {**common,'parameterization':mode,
            'objective':'observed_rays' if rays else 'filtered_occupancy',**target(16,rays)}
        if rays:
            payload.update(steps=2,stages=[target(16,True),target(32,True)])
        result = adapter.fit_job(payload)
        assert np.isfinite(result['vertices']).all() and result['optimizer_updates']==payload['steps']
        assert result['final_update_evaluated'] and result['finest_evaluated_resolution']==(32 if rays else 16)
        assert result['stop_reason']=='requested_steps' and not result['partial']
        return {key:result[key] for key in ('history','coordinate_model','stages','optimizer_updates',
            'objective_evaluations','final_update_evaluated','finest_evaluated_resolution','stop_reason')}

    for mode in ('cage','differential','vertices'):
        stage('filtered_'+mode, lambda mode=mode:fit(mode))
    stage('staged_observed_rays', lambda:fit('cage',True))

    def filter_match():
        import types,importlib
        from types import SimpleNamespace
        package='blendslop_ray_validation'
        for name,path in ((package,ROOT/'blender_blocking/reconstruction'),(package+'.differentiable',NUMERIC)):
            namespace=types.ModuleType(name);namespace.__path__=[str(path)];sys.modules[name]=namespace
        ray=importlib.import_module(package+'.differentiable.ray_evidence')
        mask=np.zeros((64,64),bool);mask[20:45,17:43]=True
        bounds=SimpleNamespace(x0=-1.,x1=1.,y0=-1.,y1=1.)
        constraint=SimpleNamespace(view='front',mask=mask,valid_mask=None,camera=SimpleNamespace(bounds=bounds),diagnostics={})
        target=SimpleNamespace(constraints=(constraint,))
        lo=np.array([-1.+17/32.,-.7,1.-45/32.]);hi=np.array([-1.+43/32.,.7,1.-20/32.])
        corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
        triangles=np.array([(f[0],f[i],f[i+1]) for f in quads for i in (1,2)])
        mesh_faces=torch.tensor(triangles,dtype=torch.int64)
        rows=[]
        for n in (16,32):
            expected=ray.prepare_ray_targets(target,np.zeros(3),1.,n)['front']['foreground']
            for dtype,tolerance in ((torch.float32,2e-5),(torch.float64,1e-12)):
                mesh=torch.tensor(lo+corners*(hi-lo),dtype=dtype)
                grid=dvx.voxelize(n,mesh,mesh_faces,method='cf').detach().numpy()
                error=float(np.max(np.abs(grid.max(axis=1)-expected)))
                assert error<tolerance,(str(dtype),n,error,tolerance)
                thin_lo=lo.copy();thin_hi=hi.copy();thin_lo[1]=-.00625;thin_hi[1]=.00625
                thin=torch.tensor(thin_lo+corners*(thin_hi-thin_lo),dtype=dtype)
                thin_grid=dvx.voxelize(n,thin,mesh_faces,method='cf').detach().numpy()
                gap=float(np.max(expected-thin_grid.max(axis=1)))
                assert gap>.5
                rows.append({'resolution':n,'dtype':str(dtype),'maximum_abs_box_filter_error':error,
                    'declared_numeric_tolerance':tolerance,'thin_depth_max_ray_gap':gap,
                    'thin_depth_ray_surrogate_is_exact_projection':False})
        return {'geometry':'axis-aligned original-pixel-cell box extruded through full depth cells',
            'filter':'official DVX cf one-voxel box','rows':rows,
            'limitation':'max of filtered volume along depth is not exact projected coverage for subvoxel-thin material'}
    stage('actual_pixel_box_filter_match',filter_match)

    def warm_jobs():
        payload = {**common,'parameterization':'cage','objective':'filtered_occupancy',**target(16)}
        with helper.NumericHelperSession(sys.executable,ROOT/'scripts/dvx_worker.py',
                source_paths=[NUMERIC/'dvx_adapter.py',NUMERIC/'conditioned_solver.py',NUMERIC/'mesh_conditioning.py']) as session:
            probe = session.call(timeout_s=10.)
            assert probe['available']
            jobs = [session.call(payload,timeout_s=20.) for _ in range(2)]
            assert all(job['final_update_evaluated'] for job in jobs)
            assert len({probe['pid'],*[job['helper_session']['pid'] for job in jobs]})==1
            assert session.starts==1
            return {'same_owned_process':True,'starts':session.starts,'jobs':2,
                'updates':[job['optimizer_updates'] for job in jobs], 'dependencies':probe}

    stage('actual_owned_helper_jobs',warm_jobs)
    print(f'complete stages={len(receipts)} elapsed={time.perf_counter()-started:.2f}s receipt={args.output}',flush=True)


if __name__=='__main__':
    main()
