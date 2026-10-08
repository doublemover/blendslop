"""Observed ray and subcell targets in the fixed DVX z,y,x cube convention."""
from __future__ import annotations
import itertools
import time
import numpy as np

DEPTH_AXIS = {"front":1,"side":2,"top":0}


def cube_axes(center,scale,resolution):
    center=np.asarray(center,float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale<=0. or resolution<1:
        raise ValueError('fixed cube requires finite center, positive scale and resolution')
    return [center[a]-scale+(np.arange(resolution)+.5)/resolution*2.*scale for a in range(3)]


def global_thin_ray_screen(vertices,scale,resolution):
    """A global PCA-width diagnostic, never a local wall/neck thickness proof."""
    vertices=np.asarray(vertices,float)
    centered=vertices-vertices.mean(axis=0)
    _,_,frame=np.linalg.svd(centered,full_matrices=False)
    widths=np.ptp(centered@frame.T,axis=0)
    voxel_width=2.*float(scale)/int(resolution)
    return {'global_pca_widths':widths.tolist(),'physical_voxel_width':voxel_width,
        'obviously_subvoxel_thin':bool(widths.min()<voxel_width),
        'global_envelope_below_two_voxels':bool(widths.min()<2.*voxel_width),
        'scope':'global PCA envelope only; local thin features or walls remain unqualified',
        'ray_operator':'max depth of box-filtered volume; not exact projected coverage'}


def _pixel_box_integral(values,x,y):
    """Continuous summed-area integral of piecewise constant pixel cells."""
    values=np.asarray(values,float);height,width=values.shape
    cumulative=np.pad(values,((1,0),(1,0))).cumsum(axis=0).cumsum(axis=1)
    # The extra zero row/column permits exact evaluation at the far image edge.
    cumulative=np.pad(cumulative,((0,1),(0,1)),mode='edge')
    padded=np.pad(values,((0,1),(0,1)))
    xx,yy=np.meshgrid(np.clip(x,0.,width),np.clip(y,0.,height),indexing='xy')
    ix,iy=np.floor(xx).astype(int),np.floor(yy).astype(int)
    fx,fy=xx-ix,yy-iy
    return (cumulative[iy,ix]+fx*(cumulative[iy,ix+1]-cumulative[iy,ix])+
        fy*(cumulative[iy+1,ix]-cumulative[iy,ix])+fx*fy*padded[iy,ix])


def exact_pixel_box_rays(target,center,scale,resolution,*,use_pixel_maps=True):
    from ..projection_contract import pixel_cell_viewport
    from ..visibility import valid_evidence
    from ..pixel_evidence import observed_pixel_evidence
    cube_axes(center,scale,resolution)  # Validate the fixed physical transform.
    edges=[center[axis]-scale+np.arange(resolution+1)/resolution*2.*scale for axis in range(3)]
    rays={}
    for constraint in target.constraints:
        (a,b),(u0,u1,v0,v1)=pixel_cell_viewport(target,constraint)
        if use_pixel_maps:
            evidence=observed_pixel_evidence(constraint)
        else:
            from types import SimpleNamespace
            mask=np.asarray(getattr(constraint.mask,'mask',constraint.mask),bool);known=valid_evidence(constraint)
            evidence=SimpleNamespace(foreground=mask.astype(float),valid=known,weights=known.astype(float))
        reference=evidence.foreground;valid=evidence.valid;height,width=reference.shape
        x=(edges[a]-u0)/(u1-u0)*width
        y=(v1-edges[b])/(v1-v0)*height
        area=(x[1]-x[0])*(y[0]-y[1])
        if not area>0.:raise ValueError('ray pixel box has invalid physical footprint')
        integrate=lambda values:-np.diff(np.diff(_pixel_box_integral(values,x,y),axis=0),axis=1)
        known=np.clip(integrate(valid),0.,area)
        foreground=np.clip(integrate(reference*valid),0.,known)
        reliability=np.clip(integrate(evidence.weights),0.,known)
        rays[constraint.view]={'foreground':np.divide(foreground,known,out=np.zeros_like(foreground),where=known>0.),
            'valid':known/area,'depth_axis_zyx':DEPTH_AXIS[constraint.view],
            'weights':reliability/area,
            'evidence_weight_semantics':'spatial confidence times boundary reliability; separate from observed fraction',
            'layout':'world-increasing plane axes; original pixel-cell integration includes screen vertical flip',
            'filter':'exact original pixel-cell area in one-voxel uniform box',
            'normalized_box_half_width':1./resolution,
            'voxelizer_footprint_scope':'DVX cf default box support; general depth-ray surrogate agreement is separate'}
    return rays


def prepare_ray_targets(target,center,scale,resolution,*,quadrature=2,filter_mode='pixel_cell_box_exact',use_pixel_maps=True):
    from ..projection_contract import AXES,project_vertices
    from ..visibility import valid_evidence
    if quadrature not in (1,2,3):
        raise ValueError('target quadrature must be 1, 2 or 3 samples per axis')
    if filter_mode=='pixel_cell_box_exact':
        return exact_pixel_box_rays(target,np.asarray(center,float),scale,resolution,use_pixel_maps=use_pixel_maps)
    if filter_mode!='subcell_quadrature':raise ValueError('unsupported ray filter mode')
    axes=cube_axes(center,scale,resolution)
    cell=2.*scale/resolution
    offsets=((np.arange(quadrature)+.5)/quadrature-.5)*cell
    rays={}
    for constraint in target.constraints:
        a,b=AXES[constraint.view]
        uu,vv=np.meshgrid(axes[a],axes[b],indexing='xy')
        foreground=np.zeros(uu.size,float);known=np.zeros(uu.size,float)
        reference=np.asarray(getattr(constraint.mask,'mask',constraint.mask),bool)
        valid=valid_evidence(constraint);h,w=reference.shape
        for du,dv in itertools.product(offsets,repeat=2):
            points=np.broadcast_to(np.asarray(center,float),(uu.size,3)).copy()
            points[:,a],points[:,b]=uu.ravel()+du,vv.ravel()+dv
            xy=np.floor(project_vertices(target,constraint,points)+.5).astype(int)
            inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            ids=np.flatnonzero(inside);x,y=xy[ids].T
            observed=valid[y,x]
            known[ids]+=observed;foreground[ids]+=observed & reference[y,x]
        rays[constraint.view]={'foreground':np.divide(foreground,known,out=np.zeros_like(foreground),where=known>0).reshape(resolution,resolution),
            'valid':(known/quadrature**2).reshape(resolution,resolution),
            'depth_axis_zyx':DEPTH_AXIS[constraint.view],
            'filter':'legacy deterministic subcell quadrature',
            'layout':'world-increasing plane axes; original pixel lookup includes screen vertical flip'}
    return rays


def prepare_coverage_target(target,center,scale,resolution,*,quadrature=2,progress=None):
    from ..visibility import point_support
    if quadrature not in (1,2,3):
        raise ValueError('target quadrature must be 1, 2 or 3 samples per axis')
    axes=cube_axes(center,scale,resolution)
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    cell=2.*scale/resolution
    offsets=((np.arange(quadrature)+.5)/quadrature-.5)*cell
    foreground=np.zeros(len(points),float);known=np.zeros(len(points),float)
    started=time.perf_counter()
    combinations=list(itertools.product(offsets,repeat=3))
    for index,offset in enumerate(combinations):
        for start in range(0,len(points),65536):
            stop=min(start+65536,len(points))
            supported,observed=point_support(target,points[start:stop]+offset)
            foreground[start:stop]+=supported & (observed>0)
            known[start:stop]+=observed>0
        if progress is not None:
            progress(index+1,len(combinations),time.perf_counter()-started)
    coverage=np.divide(foreground,known,out=np.zeros_like(foreground),where=known>0)
    return {'target_zyx':coverage.reshape((resolution,)*3).transpose(2,1,0),
            'target_valid_zyx':(known/quadrature**3).reshape((resolution,)*3).transpose(2,1,0),
            'filter':'deterministic subcell quadrature; unknown subcells omitted',
            'voxelizer_footprint_match':'pending actual pinned-runtime forward check'}


def ray_reference_loss(grid,rays):
    """NumPy contract oracle for max ray coverage, not the DVX voxelizer."""
    grid=np.clip(np.asarray(grid,float),0.,1.)
    total,weight=0.,0.
    for record in rays.values():
        predicted=np.max(grid,axis=int(record['depth_axis_zyx']))
        valid=np.asarray(record.get('weights',record['valid']),float)
        total+=float(np.sum((predicted-np.asarray(record['foreground']))**2*valid))
        weight+=float(valid.sum())
    if weight<=0.:
        raise ValueError('ray loss has no observed ray footprint')
    return total/weight


def union_coverage_reference(part_grids):
    return 1.-np.prod(1.-np.clip(np.asarray(part_grids,float),0.,1.),axis=0)
