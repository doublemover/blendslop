"""Evidence-supported planar extrusions with original pixel-cell boundaries."""
from __future__ import annotations
from dataclasses import replace
import numpy as np


def mask_polygons(target,constraint):
    from shapely import box,union_all
    from .visibility import valid_evidence
    from .projection_contract import pixel_cell_viewport
    mask=np.asarray(getattr(constraint.mask,'mask',constraint.mask),bool)
    if not valid_evidence(constraint).all():
        raise ValueError('initial polygon extrusion needs a fully observed planar outline')
    h,w=mask.shape
    _,(u0,u1,v0,v1)=pixel_cell_viewport(target,constraint)
    rectangles=[]
    for row in range(h):
        changes=np.diff(np.pad(mask[row],(1,1)).astype(int))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            rectangles.append(box(u0+a/w*(u1-u0),v1-(row+1)/h*(v1-v0),
                                  u0+b/w*(u1-u0),v1-row/h*(v1-v0)))
    if not rectangles:
        return []
    shape=union_all(rectangles)
    polygons=list(shape.geoms) if shape.geom_type=='MultiPolygon' else [shape]
    return [p.simplify(0.,preserve_topology=True) for p in polygons]


def planar_extrusion_programs(target,seed,*,maximum_thickness_ratio=.2,maximum_components=4):
    """Use measured axis thickness, never a semantic object class or hidden truth."""
    from primitives.shape_program import ShapeNode
    from .projection_contract import AXES
    lo,hi=np.asarray(target.bounds.to_min_max(),float)
    spans=hi-lo;center=(lo+hi)*.5
    if not np.all(spans>0.):
        return []
    normal=int(np.argmin(spans))
    if spans[normal]>maximum_thickness_ratio*spans.max():
        return []
    proposals=[]
    for constraint in target.constraints:
        axes=AXES[constraint.view]
        if normal in axes:
            continue
        polygons=mask_polygons(target,constraint)
        if not polygons or len(polygons)>maximum_components:
            continue
        frame=np.column_stack((np.eye(3)[:,axes[0]],np.eye(3)[:,axes[1]],
                               np.cross(np.eye(3)[:,axes[0]],np.eye(3)[:,axes[1]])))
        nodes=[]
        for index,polygon in enumerate(polygons):
            outline=np.asarray(polygon.exterior.coords[:-1],float)-center[list(axes)]
            holes=[(np.asarray(h.coords[:-1],float)-center[list(axes)]).tolist() for h in polygon.interiors]
            sizes=np.ptp(outline,axis=0)
            nodes.append(ShapeNode('polygon_'+str(index),'add','polygon_extrusion',parameters={
                'outer':outline.tolist(),'holes':holes,'width_world':float(sizes[0]),
                'depth_world':float(sizes[1]),'height_world':float(spans[normal]),
                **dict(zip(('x','y','z'),center.tolist())),'rotation':frame.tolist(),
                'outline_source':'known foreground pixel-cell boundaries','thickness_source':'other observed axis bounds'}))
        proposals.append(replace(seed,root_nodes=tuple(nodes),constraints=(),residual_patches=(),metadata={
            **seed.metadata,'proposal':'planar_polygon_extrusion','source_view':constraint.view,
            'parts':len(nodes),'hole_count':sum(len(n.parameters['holes']) for n in nodes),
            'thickness_is_evidence_bound':True,'full_mask_admission_required':True}))
    return proposals
