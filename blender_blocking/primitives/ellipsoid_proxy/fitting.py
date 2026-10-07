"""Bounded opaque union fitting; confidence/opacity cannot replace mesh geometry."""
from copy import deepcopy
import time
import numpy as np


def opaque_parts(primitives):
    result=deepcopy(list(primitives))
    for part in result:
        if hasattr(part,'opacity'):part.opacity=1.
        if hasattr(part,'density'):part.density=1.
        part.confidence=1.
    return result


class OpaqueProxyObjective:
    def __init__(self,target,*,resolution=48):
        import cv2
        from primitives.soft_silhouette import OrthographicCamera,ProjectedSilhouetteCache
        from reconstruction.projection_contract import pixel_cell_viewport
        from reconstruction.visibility import valid_evidence
        from reconstruction.pixel_evidence import observed_pixel_evidence,resize_pixel_evidence
        self.records=[];cameras=[]
        for constraint in target.constraints:
            mask=np.asarray(constraint.mask,bool);valid=valid_evidence(constraint)
            if not valid.any():continue
            height,width=mask.shape
            ratio=min(1.,int(resolution)/max(width,height))
            out_width=max(1,round(width*ratio));out_height=max(1,round(height*ratio))
            evidence=resize_pixel_evidence(observed_pixel_evidence(constraint),(out_height,out_width))
            fraction=evidence['reliability_weight'];reference=evidence['foreground']
            foreground=reference*fraction
            axes,bounds=pixel_cell_viewport(target,constraint)
            camera=OrthographicCamera(constraint.view,axes,(out_width,out_height),bounds)
            cameras.append(camera)
            # Balance observed foreground/background; unknown pixels have zero
            # weight and partial cells carry their true observed fraction.
            pos=float(foreground.sum());neg=float((fraction-foreground).sum())
            weight=fraction*(reference/max(pos,1.)+(1.-reference)/max(neg,1.))
            weight/=max(float(weight.sum()),np.finfo(float).tiny)
            self.records.append((constraint.view,reference,weight))
        if not self.records:raise ValueError('fitted proxy requires observed pixels')
        self.cache=ProjectedSilhouetteCache(cameras,softness=24.)
        self.last_vector=None

    def residual_vector(self,primitives,*,evaluated=None):
        from scipy.special import logsumexp
        masks=self.cache.render(opaque_parts(primitives))
        per_view=[];vectors=[]
        for name,reference,weight in self.records:
            residual=(masks[name]-reference)*np.sqrt(weight)
            per_view.append(float(np.sum(residual**2)))
            vectors.append(residual.ravel()/np.sqrt(len(self.records)))
        # Smooth worst-view proposal objective, zero for all-zero residuals.
        worst=max(0.,float((logsumexp(np.asarray(per_view)*12.)-np.log(len(per_view)))/12.))
        return np.concatenate([*vectors,np.array([np.sqrt(worst)])])

    def __call__(self,primitives):
        from placement.resfit_objective import ResFitObjectiveResult
        vector=self.residual_vector(primitives)
        return ResFitObjectiveResult(float(vector@vector),{'opaque_observed_union':float(vector@vector)})

    def parameter_gradients(self,primitives):
        from scipy.special import softmax
        from placement.resfit_parameters import pullback_render_gradients
        parts=opaque_parts(primitives);masks=self.cache.render(parts)
        losses=[float(np.sum((masks[name]-reference)**2*weight)) for name,reference,weight in self.records]
        worst=softmax(np.asarray(losses)*12.)
        pixel_gradients={name:2.*(masks[name]-reference)*weight*(1./len(losses)+worst[index])
            for index,(name,reference,weight) in enumerate(self.records)}
        rows=pullback_render_gradients(parts,*self.cache.backward(pixel_gradients))
        return {ref:value for ref,value in rows.items() if ref[1] not in {'opacity','density','confidence'}}


def hard_evidence(target,primitives,*,resolution=20):
    from reconstruction.mesh_io import combine_primitive_meshes,mesh_arrays_from_object
    from reconstruction.projected_metrics import projected_mesh_masks
    from reconstruction.visibility import valid_evidence,evaluate_visible_pair
    from blender_blocking.evaluation.silhouette_eval import SilhouetteGateConfig
    from reconstruction.feature_evidence import known_empty_feature_guard
    vertices,faces=mesh_arrays_from_object(combine_primitive_meshes(primitives,resolution=resolution))
    masks=projected_mesh_masks(target,vertices,faces)
    metrics={constraint.view:evaluate_visible_pair(np.asarray(constraint.mask,bool),masks[constraint.view],
        constraint,view=constraint.view,config=SilhouetteGateConfig(min_area_iou=.7),required=True)
        for constraint in target.constraints}
    areas=[row['area_iou'] for row in metrics.values() if row['area_iou'] is not None]
    boundaries=[row['boundary_iou'] for row in metrics.values() if row['boundary_iou'] is not None]
    empty={};holes={}
    for constraint in target.constraints:
        valid=valid_evidence(constraint);reference=np.asarray(constraint.mask,bool)
        known_empty=valid&~reference
        empty[constraint.view]=float(np.count_nonzero(masks[constraint.view]&known_empty)/max(1,np.count_nonzero(known_empty)))
        holes[constraint.view]=known_empty_feature_guard(constraint,masks[constraint.view])
    return {'masks':masks,'metrics':metrics,'minimum':min(areas) if areas else 0.,
        'mean':float(np.mean(areas)) if areas else 0.,'boundary':float(np.mean(boundaries)) if boundaries else 0.,
        'empty':empty,'holes':holes,'parts':len(primitives)}


def admissible(proposal,current,*,allow_equal=False):
    if any(not row['passed'] for row in proposal['holes'].values()):return False
    if proposal['minimum']<current['minimum']-1e-12:return False
    if proposal['boundary']<current['boundary']-1e-12:return False
    if any(value>current['empty'][view]+1e-12 for view,value in proposal['empty'].items()):return False
    score=lambda row:row['minimum']+row['mean']+.25*row['boundary']
    gain=score(proposal)-score(current)
    return gain>1e-12 or (allow_equal and gain>=-1e-12 and proposal['parts']<current['parts'])


def unique_foreground(target,primitives):
    from reconstruction.visibility import valid_evidence
    from reconstruction.mesh_io import combine_primitive_meshes,mesh_arrays_from_object
    from reconstruction.projected_metrics import projected_mesh_masks
    individual=[projected_mesh_masks(target,*mesh_arrays_from_object(combine_primitive_meshes([part],resolution=20)))
        for part in primitives]
    result=np.zeros(len(primitives),int)
    for constraint in target.constraints:
        support=np.asarray(constraint.mask,bool)&valid_evidence(constraint)
        rows=np.stack([row[constraint.view] for row in individual])
        unique=rows&(rows.sum(axis=0)==1)&support
        result+=unique.sum(axis=(1,2))
    return result


def missing_region_seed(target,current,points,*,family):
    """Select one actual missing component, then keep only compatible world rays."""
    from scipy.ndimage import label
    from reconstruction.visibility import valid_evidence,point_support
    from reconstruction.projection_contract import project_vertices
    from placement.resfit_initialization import (PrimitiveInitializationConfig,
        initialize_ellipsoids_from_points,initialize_gaussians_from_points)
    points=np.asarray(points,float)
    if not len(points):return None
    allowed,observed=point_support(target,points)
    candidates=[]
    for constraint in target.constraints:
        missing=np.asarray(constraint.mask,bool)&valid_evidence(constraint)&~current['masks'][constraint.view]
        components,count=label(missing)
        for index in range(1,count+1):
            region=components==index
            if region.sum()>=4:candidates.append((int(region.sum()),constraint,region))
    if not candidates:return None
    _,constraint,region=max(candidates,key=lambda row:row[0])
    pixels=np.floor(project_vertices(target,constraint,points)+.5).astype(int)
    inside=(pixels[:,0]>=0)&(pixels[:,0]<region.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<region.shape[0])
    selected=np.zeros(len(points),bool);indices=np.flatnonzero(inside)
    selected[indices]=region[pixels[indices,1],pixels[indices,0]]
    subset=points[selected&allowed&(observed>0)]
    if len(subset)<4:return None
    config=PrimitiveInitializationConfig(primitive_count=1,target_point_count=len(subset))
    initialize=initialize_gaussians_from_points if family.startswith('gaussian') else initialize_ellipsoids_from_points
    return initialize(subset,config)[0]


def fit_proxy(primitives,target,points,*,family,config,timeout_s=None):
    from placement.resfit_optimizer import CoordinateDescentConfig,ParameterBounds
    from placement.resfit_coupled import coupled_block_optimize
    started=time.perf_counter();deadline=None if timeout_s is None else started+max(.001,float(timeout_s))
    print(f'proxy-fit stage=initial_geometry parts={len(primitives)} elapsed=0.00s',flush=True)
    remaining=lambda:None if deadline is None else max(.001,deadline-time.perf_counter())
    working=deepcopy(list(primitives));initial=hard_evidence(target,working);current=initial
    render_checkpoints=[];last_checked_loss=[float('inf')]
    objective=OpaqueProxyObjective(target,resolution=int(config.get('proxy_fit_resolution',48)))
    allowance=int(config.get('proxy_fit_evaluations',192))
    gradients=objective.parameter_gradients(working)
    responsibility=np.zeros(len(working))
    for ref,value in gradients.items():responsibility[ref[0]]+=abs(value)
    priority=tuple(int(index) for index in np.argsort(-responsibility,kind='stable'))
    optimizer=CoordinateDescentConfig(iterations=int(config.get('proxy_fit_iterations',2)),initial_step=.15,
        max_objective_evaluations=allowance,max_elapsed_s=remaining(),
        bounds=ParameterBounds(min_radius=float(config.get('min_radius',1e-4)),
            max_radius=float(config.get('max_radius') or 1e4)))
    def checkpoint(scored):
        nonlocal working,current
        if len(render_checkpoints)>=8 or scored.best_loss>last_checked_loss[0]*.9:return
        if deadline is not None and time.perf_counter()>=deadline:return
        last_checked_loss[0]=scored.best_loss
        candidate=hard_evidence(target,scored.primitives)
        accepted=admissible(candidate,current)
        render_checkpoints.append({'evaluation':scored.objective_evaluations,'accepted':accepted,
            'surrogate_loss':scored.best_loss,'original_area_iou_mean':candidate['mean']})
        if accepted:working=deepcopy(list(scored.primitives));current=candidate
        print(f'proxy-fit evaluations={scored.objective_evaluations} mesh_checks={len(render_checkpoints)} elapsed={time.perf_counter()-started:.2f}s',flush=True)
    print(f'proxy-fit parts={len(working)} evaluations=0 elapsed={time.perf_counter()-started:.2f}s',flush=True)
    result=coupled_block_optimize(working,objective,optimizer,on_progress=checkpoint,priority_parts=priority,
        parameter_filter=lambda ref:ref[1] not in {'opacity','density','confidence'})
    proposal=hard_evidence(target,result.primitives)
    accepted=admissible(proposal,current)
    if accepted:working=list(result.primitives);current=proposal
    history=[{'stage':'coupled_opaque_union','accepted':accepted,'evaluations':result.objective_evaluations,
        'original_size_checkpoints':render_checkpoints,
        'termination':result.termination_reason,'surrogate_initial':result.initial_result.total,
        'surrogate_best':result.best_loss,'parameter_visits':[list(ref) for ref in sorted(set(result.parameter_visits),key=str)]}]
    # Structural edits use actual unique observed foreground and original-size
    # evidence. They cannot claim an invisible opacity change as simplification.
    if len(working)>1 and (deadline is None or time.perf_counter()<deadline):
        contributions=unique_foreground(target,working)
        for index in np.argsort(contributions):
            if contributions[index]>2 or len(working)<=1:break
            proposal_parts=working[:int(index)]+working[int(index)+1:]
            candidate=hard_evidence(target,proposal_parts)
            if admissible(candidate,current,allow_equal=True):
                history.append({'stage':'unique_coverage_prune','accepted':True,'removed':int(index),
                    'unique_observed_pixels':int(contributions[index])})
                working=proposal_parts;current=candidate
                break
    if len(working)>1 and (deadline is None or time.perf_counter()<deadline):
        replacement=missing_region_seed(target,current,points,family=family)
        if replacement is not None:
            index=int(np.argmin(unique_foreground(target,working)))
            proposal_parts=deepcopy(working);proposal_parts[index]=replacement
            candidate=hard_evidence(target,proposal_parts)
            accepted=admissible(candidate,current)
            history.append({'stage':'missing_component_reallocation','accepted':accepted,'replaced':index})
            if accepted:working=proposal_parts;current=candidate
    return tuple(working),{'variant':'fitted_opaque_union_v1','fit_history':history,
        'optimization_performed':True,
        'objective_evaluations':result.objective_evaluations,'initial_parts':len(primitives),'retained_parts':len(working),
        'evidence_gradient_priority':list(priority),'evidence_gradient_per_part':responsibility.tolist(),
        'original_size_mesh_gate':True,'opacity_and_confidence_optimized':False,
        'mesh_isosurface_sigma':1.,'surrogate':'analytic opaque ellipsoid union; smooth balanced pixel and worst-view error',
        'initial_area_iou_mean':initial['mean'],'retained_area_iou_mean':current['mean'],
        'elapsed_s':time.perf_counter()-started,'single_solid_qualification':False}
