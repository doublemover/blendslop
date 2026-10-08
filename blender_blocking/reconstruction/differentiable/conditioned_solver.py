"""Optional pinned-CPU DVX refinement of a retained seed with local protection."""
from __future__ import annotations
import time
import sys
import numpy as np


def fit_conditioned_job(payload,state):
    """Called only after the adapter's existing execution/dependency checks."""
    import torch
    import dvx.torch as dvx
    from .mesh_conditioning import torch_coordinates,mesh_edges
    world=np.asarray(payload['vertices'],float);faces=np.asarray(payload['faces'],np.int64)
    center=np.asarray(payload['center'],float);scale=float(payload['scale'])
    normalized=(world-center)/scale
    if not np.isfinite(normalized).all() or np.max(np.abs(normalized))>=1.:
        raise ValueError('fixed conditioned DVX transform does not contain its seed')
    seed=torch.tensor(normalized,dtype=torch.float32);f=torch.tensor(faces,dtype=torch.int64)
    edges=torch.tensor(mesh_edges(faces),dtype=torch.int64)
    old_lengths=torch.linalg.vector_norm(seed[edges[:,0]]-seed[edges[:,1]],dim=1)
    old_triangles=seed[f]
    old_cross=torch.linalg.cross(old_triangles[:,1]-old_triangles[:,0],old_triangles[:,2]-old_triangles[:,0])
    old_areas=torch.linalg.vector_norm(old_cross,dim=1)
    if bool((old_lengths<=0).any()) or bool((old_areas<=0).any()):
        raise ValueError('conditioned DVX seed has collapsed edges or triangles')
    anchor=torch.tensor(payload.get('anchor_weights',np.ones(len(seed))),dtype=torch.float32)
    if anchor.shape != (len(seed),) or not torch.isfinite(anchor).all() or bool((anchor<0).any()):
        raise ValueError('conditioned seed anchor weights are invalid')
    mode=payload.get('parameterization','cage')
    parameters,decode,coordinate_report=torch_coordinates(normalized,faces,mode=mode,
        strength=float(payload.get('differential_strength',4.)))
    optimizer=torch.optim.Adam([parameters],lr=float(payload.get('learning_rate',.005)))
    stages=payload.get('stages') or [payload]
    requested=min(64,max(1,int(payload.get('steps',12))))
    allocations=[requested//len(stages)+(i<requested%len(stages)) for i in range(len(stages))]
    if requested<len(stages):
        # Spend scarce work at the final physical feature resolution rather than
        # inventing unevaluated finer stages after exhausting a coarse schedule.
        stages=stages[-requested:];allocations=[1]*len(stages)
    started=time.perf_counter();history=[];stage_history=[]
    evaluations=updates=0;best_index=None;best=seed.clone();stop_reason='requested_steps'
    final_evaluated=False;reached=None;stage_best=float('inf');best_parameters=parameters.detach().clone()
    objective=payload.get('objective','observed_rays')
    projected_reports={}
    part_ranges=payload.get('part_ranges') or [{'face_start':0,'face_count':len(faces)}]
    multipart=payload.get('voxel_semantics')=='per_part_union_coverage'
    weights={'displacement':.02,'edge_distortion':.05,'area_barrier':.1,'normal_change':.005,
             **payload.get('regularization_weights',{})}

    def timed_out():
        return payload.get('timeout_s') is not None and time.perf_counter()-started>=payload['timeout_s']

    def snapshot():
        return {'vertices':best.detach().numpy()*scale+center,'faces':faces,'history':list(history),
            'dependencies':state,'fixed_transform':{'center':center.tolist(),'scale':scale},
            'layout':'DVX z,y,x','target_transferred_once':True,'device':'cpu','requested_steps':requested,
            'optimizer_updates':updates,'objective_evaluations':evaluations,'best_evaluation':best_index,
            'stop_reason':stop_reason,'optimization_wall_s':time.perf_counter()-started,
            'final_update_evaluated':final_evaluated,'partial':stop_reason!='requested_steps',
            'coordinate_model':coordinate_report,'evidence_objective':objective,
            'ray_operator_semantics':'max depth of box-filtered winding coverage; subvoxel thickness is not exact projected coverage' if objective=='observed_rays' else None,
            'projected_operator_semantics':'opaque projected triangle union; fixed active-topology boundary pullback and DVX 2D box filter' if objective=='observed_projected_rays' else None,
            'voxel_semantics':'per_part_union_coverage_surrogate' if multipart else 'retained_seed_winding_surface',
            'regularization_weights':weights,'stages':list(stage_history),'finest_evaluated_resolution':reached,
            'stage_losses_comparable':False,'seed_parameters_reproduce_deformation':False}

    def checkpoint():
        import os,pickle,uuid
        from pathlib import Path
        value=snapshot()
        for name in payload.get('progress_paths',()):
            path=Path(name);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            with temporary.open('wb') as stream:
                pickle.dump({'status':'partial','value':value,'recorded_evaluations':evaluations},stream,protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary,path)

    for stage_index,(stage,steps) in enumerate(zip(stages,allocations)):
        if timed_out():
            stop_reason='elapsed_time_budget';break
        target=torch.tensor(np.asarray(stage['target_zyx'],np.float32))
        valid=torch.tensor(np.asarray(stage.get('target_valid_zyx',np.ones(target.shape)),np.float32))
        n=int(target.shape[0])
        if (target.shape != (n,n,n) or n not in (16,32,64) or valid.shape != target.shape or
            not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(valid).all()) or
            bool((valid<0).any()) or bool((valid>1).any())):
            raise ValueError('conditioned DVX stage has invalid fixed-cube target dimensions')
        ray_targets={name:{'foreground':torch.tensor(row['foreground'],dtype=torch.float32),
                           'valid':torch.tensor(row['valid'],dtype=torch.float32),
                           'weights':torch.tensor(row.get('weights',row['valid']),dtype=torch.float32),
                           'depth_axis_zyx':int(row['depth_axis_zyx'])}
                     for name,row in stage.get('ray_targets',{}).items()}
        for row in ray_targets.values():
            if (row['foreground'].shape != (n,n) or row['valid'].shape != (n,n) or
                not bool(torch.isfinite(row['foreground']).all()) or not bool(torch.isfinite(row['valid']).all()) or
                bool((row['foreground']<0).any()) or bool((row['foreground']>1).any()) or
                bool((row['valid']<0).any()) or bool((row['valid']>1).any()) or
                row['weights'].shape!=(n,n) or not bool(torch.isfinite(row['weights']).all()) or
                bool((row['weights']<0).any()) or bool((row['weights']>row['valid']+1e-6).any())):
                raise ValueError('conditioned ray target dimensions or weights are invalid')
        stage_best=float('inf');final_evaluated=False;stage_start=time.perf_counter()
        terms={}

        def evaluate():
            nonlocal evaluations,best,best_index,stage_best,terms,reached,best_parameters,projected_reports
            evaluations+=1
            reached=n
            vertices=decode()
            if objective=='observed_projected_rays':
                from .projected_mesh_rays import projected_mesh_grids
                projections,projected_reports=projected_mesh_grids(vertices,f,n,list(ray_targets),maximum_faces=2048)
                numerator=vertices.sum()*0.;denominator=0.
                for name,row in ray_targets.items():
                    predicted=torch.clamp(projections[name],0.,1.)
                    numerator=numerator+((predicted-row['foreground'])**2*row['weights']).sum()
                    denominator=denominator+row['weights'].sum()
                if not bool(denominator>0):raise ValueError('projected ray objective has no positive reliability-weighted footprint')
                evidence=numerator/denominator
            elif multipart:
                grids=[dvx.voxelize(n,vertices,f[int(p['face_start']):int(p['face_start'])+int(p['face_count'])],method='cf') for p in part_ranges]
                grid=1.-torch.prod(1.-torch.clamp(torch.stack(grids),0.,1.),dim=0)
            elif objective!='observed_projected_rays':
                grid=dvx.voxelize(n,vertices,f,method='cf')
            if objective=='observed_rays':
                coverage=torch.clamp(grid,0.,1.)
                numerator=coverage.sum()*0.;denominator=0.
                for row in ray_targets.values():
                    predicted=torch.amax(coverage,dim=row['depth_axis_zyx'])
                    numerator=numerator+((predicted-row['foreground'])**2*row['weights']).sum()
                    denominator=denominator+row['weights'].sum()
                if not bool(denominator>0):
                    raise ValueError('conditioned ray objective has no positive reliability-weighted footprint')
                evidence=numerator/denominator
            elif objective=='filtered_occupancy':
                if not bool(valid.sum()>0):
                    raise ValueError('conditioned occupancy objective has no observed cells')
                evidence=(((grid-target)**2)*valid).sum()/valid.sum()
            elif objective!='observed_projected_rays':
                raise ValueError('unsupported conditioned DVX evidence objective')
            displacement=((vertices-seed)**2*anchor[:,None]).sum()/torch.clamp(anchor.sum()*3.,min=1e-12)
            lengths=torch.linalg.vector_norm(vertices[edges[:,0]]-vertices[edges[:,1]],dim=1)
            triangles=vertices[f]
            cross=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            areas=torch.linalg.vector_norm(cross,dim=1)
            distortion=((lengths/old_lengths-1.)**2).mean()
            area_barrier=torch.clamp(float(payload.get('minimum_area_ratio',.1))-areas/old_areas,min=0.).square().mean()
            cosine=(old_cross*cross).sum(dim=1)/torch.clamp(old_areas*areas,min=1e-12)
            normal_change=(1.-torch.clamp(cosine,-1.,1.)).square().mean()
            components={'displacement':displacement,'edge_distortion':distortion,
                        'area_barrier':area_barrier,'normal_change':normal_change}
            loss=evidence+sum(weights[name]*value for name,value in components.items())
            if bool(torch.isfinite(loss)):
                value=float(loss.detach());history.append(value)
                terms={'evidence':float(evidence.detach()),**{name:float(value.detach()) for name,value in components.items()}}
                if value<stage_best:
                    best,stage_best,best_index=vertices.detach().clone(),value,evaluations
                    best_parameters=parameters.detach().clone()
                checkpoint()
            print(f'DVX stage={stage_index+1}/{len(stages)} grid={n} evaluations={evaluations} updates={updates} elapsed={time.perf_counter()-started:.2f}s',file=sys.stderr,flush=True)
            return loss

        for step in range(int(steps)):
            if timed_out():
                stop_reason='elapsed_time_budget';break
            optimizer.zero_grad(set_to_none=True)
            loss=evaluate()
            if not bool(torch.isfinite(loss)):
                stop_reason='nonfinite_loss';break
            loss.backward()
            if parameters.grad is None or not bool(torch.isfinite(parameters.grad).all()):
                stop_reason='nonfinite_or_missing_gradient';break
            before=parameters.detach().clone();optimizer.step();updates+=1
            with torch.no_grad():
                proposed=parameters.detach().clone()
                for retry in range(9):
                    if bool((decode().abs()<.98).all()):
                        break
                    parameters.copy_(before+(proposed-before)*(.5**(retry+1)))
                if not bool((decode().abs()<.98).all()):
                    parameters.copy_(before)
        if stop_reason=='requested_steps' and not timed_out():
            with torch.no_grad():
                final_evaluated=bool(torch.isfinite(evaluate()))
            if not final_evaluated:
                stop_reason='nonfinite_final_loss'
        elif stop_reason=='requested_steps':
            stop_reason='elapsed_time_budget_before_final_evaluation'
        stage_history.append({'resolution':n,'updates_requested':int(steps),'best_loss':stage_best if np.isfinite(stage_best) else None,
                              'voxelizer_filter':{'method':'cf','normalized_box_half_width':1./n},
                              'ray_filters':{name:row.get('filter','unspecified') for name,row in stage.get('ray_targets',{}).items()},
                              'last_projected_boundary_reports':dict(projected_reports),
                              'last_terms':terms,'elapsed_s':time.perf_counter()-stage_start})
        if stop_reason!='requested_steps':
            break
        with torch.no_grad():
            parameters.copy_(best_parameters)
        # Keep coordinates/Adam state across fixed-topology grid levels. Best
        # candidates are reset and compared within each level, never across losses.
    if not history:
        raise TimeoutError('conditioned DVX allowance exhausted without a finite scored state')
    return snapshot()
