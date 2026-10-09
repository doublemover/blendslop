"""Exact lattice face-cover certificates for frozen authored box CSG sources.

The proof uses rational polygon clipping, not point or volume sampling. Inputs
are actual source arrays and a frozen family label only: candidate geometry,
metrics, reconstruction allowances and artist tolerances are excluded.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import itertools
import json
import math

import numpy as np

FAMILIES = ('thin_plate', 'concave_arch', 'asymmetric_multipart_solid')
MAX_FACES = 4096
MAX_PATCHES = 4096
MAX_CLIPS = 16384
MAX_OVERLAP_CHECKS = 65536


def _area(poly):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))) / 2


def _cross(a,b,p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _clip(poly, start, end):
    """Intersect a convex closed polygon with an exact left half-plane."""
    if not poly:return []
    result=[]
    for previous,current in zip(poly[-1:]+poly[:-1],poly):
        a,b=_cross(start,end,previous),_cross(start,end,current)
        if (a>=0)!=(b>=0):
            t=a/(a-b)
            result.append(tuple(previous[i]+t*(current[i]-previous[i]) for i in (0,1)))
        if b>=0:result.append(current)
    return result


def _intersection(poly, other):
    for a,b in zip(other,other[1:]+other[:1]):
        poly=_clip(poly,a,b)
        if not poly:break
    return poly


def _ccw(poly):
    signed=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))
    return poly if signed>0 else list(reversed(poly))


def _sqrt_upper(value):
    """A float upper endpoint checked against the exact rational square."""
    result=math.sqrt(float(value))
    while F.from_float(result)**2<value:result=math.nextafter(result,math.inf)
    return result


def _recipe(family):
    from blender_blocking.synthetic.quality_contracts import quality_workload
    if family not in FAMILIES:raise ValueError('frozen rectilinear source family unsupported')
    row=deepcopy(next(x for x in quality_workload()['cases'] if x['name']==family))
    p=row['parameters']
    if family=='thin_plate':
        dimensions=p['dimensions']
        parts=[{'type':'box','center':[0.,0.,0.],
                'size':[dimensions[k] for k in ('width','depth','height')]}]
    else:
        if p.get('builder')!='compound':raise ValueError('authored source must be fixed box CSG')
        parts=p['parts']
    boxes=[]
    if not 1<=len(parts)<=8:raise ValueError('authored box count exceeds proof budget')
    for index,part in enumerate(parts):
        if (part.get('type')!='box' or set(part)-{'type','center','size','boolean'}
                or part.get('boolean','union') not in ('union','subtract')
                or (index==0 and part.get('boolean')=='subtract')):
            raise ValueError('only fixed axis-aligned authored box union/subtraction is supported')
        center=np.asarray(part['center'],float);size=np.asarray(part['size'],float)
        if (center.shape!=(3,) or size.shape!=(3,) or not np.isfinite(center).all()
                or not np.isfinite(size).all() or (size<=0).any()):
            raise ValueError('authored box coordinates must be finite and positive')
        # Use the frozen binary64 endpoint declaration. Subsequent geometry and
        # cover calculations remain exact in its rational representation.
        lower=tuple(F.from_float(float(x)) for x in center-size/2)
        upper=tuple(F.from_float(float(x)) for x in center+size/2)
        boxes.append((lower,upper,part.get('boolean','union')))
    return row,boxes


def _boundary(boxes):
    grid=[sorted({v[axis] for lower,upper,_ in boxes for v in (lower,upper)}) for axis in range(3)]
    shape=tuple(len(axis)-1 for axis in grid)
    if math.prod(shape)>4096:raise ValueError('authored lattice cell budget exceeded')
    occupied=set()
    for index in itertools.product(*(range(n) for n in shape)):
        point=tuple((grid[a][i]+grid[a][i+1])/2 for a,i in enumerate(index))
        state=False
        for lower,upper,operation in boxes:
            inside=all(lower[a]<point[a]<upper[a] for a in range(3))
            state=(state and not inside) if operation=='subtract' else (state or inside)
        if state:occupied.add(index)
    patches={}
    for index in sorted(occupied):
        for axis,sign in itertools.product(range(3),(-1,1)):
            neighbor=list(index);neighbor[axis]+=sign
            if tuple(neighbor) in occupied:continue
            u,v=(axis+1)%3,(axis+2)%3
            plane=grid[axis][index[axis]+(sign>0)]
            u0,u1=grid[u][index[u]:index[u]+2];v0,v1=grid[v][index[v]:index[v]+2]
            rectangle=[(u0,v0),(u1,v0),(u1,v1),(u0,v1)]
            key=(axis,plane,sign,index[u],index[v])
            patches[key]={'rectangle':rectangle,'pieces':[],'area':(u1-u0)*(v1-v0)}
    if not patches or len(patches)>MAX_PATCHES:raise ValueError('analytic boundary patch budget exceeded')
    return grid,patches


def _certificate(family, reference):
    row,boxes=_recipe(family)
    grid,patches=_boundary(boxes)
    vertices=np.asarray(reference.vertices,float);faces=np.asarray(reference.faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not 4<=len(vertices)<=4096
            or not np.isfinite(vertices).all() or faces.ndim!=2 or faces.shape[1]!=3
            or not 4<=len(faces)<=MAX_FACES or not np.issubdtype(faces.dtype,np.integer)
            or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('source finite indexed-triangle budget invalid')
    from blender_blocking.reconstruction.native_geometry import GeometryArrays
    captured=GeometryArrays.capture(vertices,faces)
    if captured.content_hash!=getattr(reference,'content_hash',None):
        raise ValueError('source array identity differs from its declared content hash')
    if len(np.unique(faces))!=len(vertices):raise ValueError('source has unused vertices')
    scale=max(1.,max(abs(float(x)) for axis in grid for x in axis))
    association_budget=F.from_float(float(4*np.finfo(np.float32).eps*scale))
    exact=[tuple(F.from_float(float(x)) for x in vertex) for vertex in vertices]
    snapped=[];max_shift_squared=F(0)
    for vertex in exact:
        snap=tuple(min(grid[a],key=lambda x:abs(x-vertex[a])) for a in range(3))
        if any(abs(snap[a]-vertex[a])>association_budget for a in range(3)):
            raise ValueError('source vertex is outside fixed numerical lattice association budget')
        max_shift_squared=max(max_shift_squared,sum((snap[a]-vertex[a])**2 for a in range(3)))
        snapped.append(snap)
    clips=0;overlap_checks=0;normal_tangent_squared=F(0)
    lookup={}
    for key,patch in patches.items():lookup.setdefault(key[:3],[]).append((key,patch))
    for face in faces:
        ids=list(map(int,face))
        if len(set(ids))!=3:raise ValueError('source triangle index degeneracy')
        points=[snapped[i] for i in ids]
        axes=[a for a in range(3) if len({p[a] for p in points})==1]
        if len(axes)!=1:raise ValueError('snapped source triangle is not a nondegenerate lattice-plane facet')
        axis=axes[0];u,v=(axis+1)%3,(axis+2)%3
        projected=[(p[u],p[v]) for p in points]
        signed=_cross(projected[0],projected[1],projected[2])
        if not signed:raise ValueError('source triangle collapses after numerical snapping')
        sign=1 if signed>0 else -1
        choices=lookup.get((axis,points[0][axis],sign),[])
        if not choices:raise ValueError('source facet has no outward analytic boundary plane')
        polygon=_ccw(projected);area=_area(polygon);covered=F(0)
        for key,patch in choices:
            rectangle=patch['rectangle']
            if (max(p[0] for p in polygon)<=rectangle[0][0] or min(p[0] for p in polygon)>=rectangle[1][0]
                    or max(p[1] for p in polygon)<=rectangle[0][1] or min(p[1] for p in polygon)>=rectangle[2][1]):continue
            clips+=1
            if clips>MAX_CLIPS:raise ValueError('triangle-to-boundary clipping budget exceeded')
            clipped=_intersection(polygon,rectangle)
            if not clipped or not _area(clipped):continue
            for previous in patch['pieces']:
                overlap_checks+=1
                if overlap_checks>MAX_OVERLAP_CHECKS:
                    raise ValueError('exact overlap-intersection budget exceeded')
                overlap=_intersection(clipped,previous)
                if overlap and _area(overlap)>0:raise ValueError('positive-area overlap on analytic boundary patch')
            patch['pieces'].append(clipped);covered+=_area(clipped)
        if covered!=area:raise ValueError('source facet extends outside the analytic CSG boundary')
        original=[exact[i] for i in ids]
        a=[original[1][k]-original[0][k] for k in range(3)]
        b=[original[2][k]-original[0][k] for k in range(3)]
        cross=[a[(k+1)%3]*b[(k+2)%3]-a[(k+2)%3]*b[(k+1)%3] for k in range(3)]
        dot=sign*cross[axis]
        if dot<=0:raise ValueError('actual source facet orientation does not preserve its snapped outward normal')
        normal_tangent_squared=max(normal_tangent_squared,sum(cross[k]**2 for k in range(3) if k!=axis)/(dot*dot))
    for patch in patches.values():
        if sum((_area(p) for p in patch['pieces']),F(0))!=patch['area']:
            raise ValueError('analytic boundary patch lacks complete nonoverlapping source-face coverage')
    shift=_sqrt_upper(max_shift_squared);tangent=_sqrt_upper(normal_tangent_squared)
    angle=math.degrees(math.atan(tangent))
    # The exact tangent bound is authoritative; trig display gets an outward
    # floating-point allowance separate from any reconstruction tolerance.
    if tangent:
        for _ in range(8):angle=math.nextafter(angle,math.inf)
    return {'protocol':'exact_rectilinear_source_cover_v1','status':'certified','family':family,
        'reference_geometry_hash':reference.content_hash,
        'frozen_parameters_sha256':hashlib.sha256(json.dumps(row['parameters'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'maximum_source_facet_distance_world':shift,'maximum_normal_angle_degrees':angle,
        'normal_tangent_upper_bound':tangent,'normal_correspondence':'each source facet to its proved outward authored plane; edge/corner nearest-normal correspondence is not asserted',
        'distance_correspondence':'bidirectional continuous facet/boundary cover via matching barycentric points; max vertex displacement bounds every paired point',
        'maximum_vertex_rounding_shift_world':shift,'numerical_grid_association_budget_world':float(association_budget),
        'numerical_budget_scope':'fixed4float32eps×authored coordinate scale; grid association only, not reconstruction allowance or artist tolerance',
        'source_triangles_proved':len(faces),'boundary_patches_proved':len(patches),'triangle_patch_clips':clips,'positive_area_overlap_checks':overlap_checks,
        'coverage_proof':'exact rational triangle-to-lattice-cell clipping; every source facet assigned fully; every boundary patch fully covered; all positive-area overlaps forbidden',
        'orientation_proof':'every snapped facet outward; exact actual cross dot with that plane strictly positive',
        'arithmetic':'exact Fraction from frozen binary64 plane declarations and actual binary64 source coordinates; square-root upper endpoints rationally checked',
        'artist_surface_limits':None,'candidate_boundary_qualification':'not supplied'}


def rectilinear_reference_certificate(family, reference):
    """Return a complete independent source certificate or an explicit reason."""
    try:
        return _certificate(family,reference)
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return {'protocol':'exact_rectilinear_source_cover_v1','status':'unsupported','family':family,
                'reference_geometry_hash':getattr(reference,'content_hash',None),
                'reason':type(exc).__name__+': '+str(exc),'artist_surface_limits':None,
                'candidate_boundary_qualification':'not supplied'}
