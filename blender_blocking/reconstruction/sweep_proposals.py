"""Stable local-frame section hypotheses from explicitly completed row evidence."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np


def generalized_sweep_programs(target,seed,*,maximum_sections=16):
    from primitives.shape_program import ShapeNode
    from primitives.generalized_sweep import GeneralizedSweepPrimitive
    from .projection_contract import calibrated_profile
    from .adaptive_geometry import adaptive_slices
    if target.bounds is None:return []
    try:
        profile=calibrated_profile(SimpleNamespace(target=target,config={'num_samples':64}))
    except ValueError:
        return []  # No circular fallback invents an unobserved axis dimension.
    active=(np.asarray(profile.rx)>0.)&(np.asarray(profile.ry)>0.)
    for records in profile.meta['row_evidence'].values():
        active &= np.array([row['within_viewport'] for row in records],bool)
    indices=np.flatnonzero(active)
    if len(indices)<2 or np.any(np.diff(indices)!=1):
        return []  # Birth/death/disconnected axial sections require another representation.
    # Preserve original knots first, then adapt to physical centerline/radius error.
    from geometry.profile_models import EllipticalProfileU
    t=np.asarray(profile.heights_t)[indices];span=float(t[-1]-t[0])
    if span<=0.:return []
    restricted=EllipticalProfileU(heights_t=(t-t[0])/span,rx=np.asarray(profile.rx)[indices],
        ry=np.asarray(profile.ry)[indices],world_height=profile.world_height*span,
        z0=profile.z0+profile.world_height*t[0],cx=np.asarray(profile.cx)[indices],cy=np.asarray(profile.cy)[indices])
    slices=adaptive_slices(restricted,minimum=3,maximum=min(64,int(maximum_sections)))
    z=np.array([row.z for row in slices]);cx=np.array([row.cx or 0. for row in slices]);cy=np.array([row.cy or 0. for row in slices])
    rx=np.array([row.rx for row in slices]);ry=np.array([row.ry for row in slices])
    width=2*float(rx.max());depth=2*float(ry.max());height=float(z[-1]-z[0]);middle=.5*float(z[0]+z[-1])
    if min(width,depth,height)<=0.:return []
    knots=np.column_stack(((z-middle)/height,cx/width,cy/depth,rx/width,ry/depth))
    center=np.asarray(target.bounds.center,float).copy();center[2]=middle
    proposals=[]
    for exponent in (.5,1.,2.):
        parameters={'section_knots_normalized':knots.tolist(),'section_exponent':exponent,
            'width_world':width,'depth_world':depth,'height_world':height,
            **dict(zip(('x','y','z'),center.tolist())),'rotation':np.eye(3).tolist()}
        part=GeneralizedSweepPrimitive.from_program_parameters(parameters)
        node=ShapeNode('editable_sweep','add','generalized_sweep',parameters=parameters)
        proposals.append(replace(seed,root_nodes=(node,),constraints=(),residual_patches=(),metadata={
            **seed.metadata,'proposal':'local_frame_completed_profile_sweep','parts':1,
            'section_family':'superellipse','section_exponent':exponent,
            'section_topology':'one stable convex loop; axial births/deaths refused',
            'evidence_completion':profile.meta,'global_frame_initializer':'world axial profile; proper local pose is editable',
            'mesh_field_disagreement':part.mesh_field_disagreement(24),
            'full_mask_admission_required':True,'field_semantics':'signed zero-set field; not Euclidean distance'}))
    return proposals
