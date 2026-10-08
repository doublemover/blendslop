"""Opt-in tiny pinned CPU projected-union/filter/pullback fixtures."""
from pathlib import Path
import argparse,importlib.util,json,sys,time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approved',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not args.approved:parser.error('approved pinned dependency execution required')
    started=time.perf_counter();rows=[]
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
        sys.modules[name]=value;spec.loader.exec_module(value);return value
    adapter=module('projected_dvx_state',ROOT/'blender_blocking/reconstruction/differentiable/dvx_adapter.py')
    state=adapter.dependency_state();assert state['available']
    import numpy as np
    import torch
    torch.set_num_threads(2)
    operator=module('projected_mesh_operator',ROOT/'blender_blocking/reconstruction/differentiable/projected_mesh_rays.py')
    def stage(label,operation):
        print(f'stage={label} completed={len(rows)} elapsed={time.perf_counter()-started:.2f}s',flush=True)
        rows.append({'stage':label,'result':operation()})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'pinned_runtime':state,'synthetic_contracts_only':True,
            'quality_and_speed_gains':'unmeasured','rows':rows,'elapsed_s':time.perf_counter()-started},indent=2))

    def thin_box():
        lo=np.array([-.5,-.00625,-.375]);hi=np.array([.375,.00625,.5])
        corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
        faces=np.array([(f[0],f[i],f[i+1]) for f in quads for i in (1,2)])
        v=torch.tensor(lo+corners*(hi-lo),dtype=torch.float64,requires_grad=True)
        grids,reports=operator.projected_mesh_grids(v,faces,16,['front'])
        expected=np.zeros((16,16));expected[5:12,4:11]=1.
        np.testing.assert_allclose(grids['front'].detach().numpy(),expected,atol=1e-12)
        (grids['front']*torch.linspace(.3,1.7,256,dtype=torch.float64).reshape(16,16)).sum().backward()
        assert torch.isfinite(v.grad).all() and float(v.grad.abs().max())>0.
        assert float(v.grad[:,1].abs().max())==0.
        return {'maximum_projection_error':float(np.max(np.abs(grids['front'].detach().numpy()-expected))),
            'invisible_depth_gradient':float(v.grad[:,1].abs().max()),'report':reports}
    stage('subvoxel_thin_box_projection',thin_box)

    def gradient(intersections):
        xy=np.array([[-.7,-.4],[.5,-.4],[0.,.6],[-.5,.1],[.7,.1],[.2,-.7]]) if intersections else np.array([[-.37,-.23],[.49,-.19],[.05,.43]])
        faces=np.array([[0,1,2],[3,4,5]]) if intersections else np.array([[0,1,2]])
        base=np.column_stack((xy[:,0],np.zeros(len(xy)),xy[:,1]))
        v=torch.tensor(base,dtype=torch.float64,requires_grad=True)
        weights=torch.linspace(.3,1.7,256,dtype=torch.float64).reshape(16,16)
        evaluate=lambda value:(operator.projected_mesh_grids(value,faces,16,['front'])[0]['front']*weights).mean()
        loss=evaluate(v);loss.backward();analytic=v.grad.detach().numpy();epsilon=1e-6;errors=[]
        for index,axis in ((0,0),(1,2),(2,0)):
            plus=base.copy();minus=base.copy();plus[index,axis]+=epsilon;minus[index,axis]-=epsilon
            numeric=(float(evaluate(torch.tensor(plus,dtype=torch.float64)))-float(evaluate(torch.tensor(minus,dtype=torch.float64))))/(2*epsilon)
            errors.append(abs(numeric-analytic[index,axis]))
        assert max(errors)<2e-8,(errors,analytic)
        return {'maximum_local_derivative_error':max(errors),'active_intersections':intersections}
    stage('original_vertex_pullback',lambda:gradient(False))
    stage('edge_intersection_pullback',lambda:gradient(True))

    def staged_job():
        lo=np.array([-.47,-.00625,-.36]);hi=np.array([.37,.00625,.48])
        corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
        faces=np.array([(f[0],f[i],f[i+1]) for f in quads for i in (1,2)])
        vertices=lo+corners*(hi-lo);shifted=torch.tensor(vertices+[.015,0.,0.],dtype=torch.float32)
        stages=[]
        for n in (16,32):
            reference=operator.projected_mesh_grids(shifted,faces,n,['front'])[0]['front'].detach().numpy()
            stages.append({'target_zyx':np.zeros((n,)*3),'target_valid_zyx':np.zeros((n,)*3),
                'ray_targets':{'front':{'foreground':np.clip(reference,0.,1.),'valid':np.ones((n,n)),
                    'weights':np.ones((n,n)),'depth_axis_zyx':1,'filter':'synthetic original projection box'}}})
        result=adapter.fit_job({'execution_approved':True,'vertices':vertices,'faces':faces,
            'center':np.zeros(3),'scale':1.,'objective':'observed_projected_rays','parameterization':'cage',
            'steps':2,'timeout_s':10.,'learning_rate':.001,'stages':stages,**stages[-1]})
        assert result['optimizer_updates']==2 and result['final_update_evaluated']
        assert result['finest_evaluated_resolution']==32 and np.isfinite(result['vertices']).all()
        return {key:result[key] for key in ('history','stages','projected_operator_semantics','optimizer_updates',
            'objective_evaluations','final_update_evaluated','finest_evaluated_resolution')}
    stage('conditioned_projected_ray_staging',staged_job)
    print(f'complete stages={len(rows)} elapsed={time.perf_counter()-started:.2f}s',flush=True)


if __name__=='__main__':main()
