"""Exact patch-cover proof for one frozen authored BEVEL8 rounded cuboid.

The ideal is the radius-.15 Minkowski offset of [.65,.40,.50] inset
half-extents. Custom/weighted shading normals are not geometric normals.
Numerical construction displacement is separate from tessellation and artist
allowance. Existing source diagnostics were exposed; no blind history is claimed.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction as F
import itertools
import math
import numpy as np

PROTOCOL = 'authored_rounded_box_patch_cover_v1'
FROZEN_SOURCE_HASH = 'adafdbd9610c2951be09d956221f147b78a72984469d1fd2c3f12d67ca3a42f0'
IDEAL_CORE = tuple(F(x) for x in ('.65', '.40', '.50'))
RADIUS = F('.15')
NATIVE_CORE = tuple(F.from_float(float.fromhex(x)) for x in
                   ('0x1.4ccccc0000000p-1', '0x1.99999a0000000p-2', '0x1.fffffe0000000p-2'))
SNAP_MAX = F('.0000001')
MAX_VERTICES, MAX_FACES = 512, 1024


def _dot(a, b):
    return sum((x*y for x,y in zip(a,b)), F(0))


def _sub(a, b):
    return tuple(x-y for x,y in zip(a,b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _sqrt(value, upper):
    if value < 0:raise ValueError('negative exact square')
    result=math.sqrt(float(value))
    if not math.isfinite(result):raise ValueError('construction arithmetic exhausted')
    # libm is only a starting guess; the exact rational square verifies bounds.
    for _ in range(4):
        square=F.from_float(result)**2
        if (square>=value if upper else square<=value):return F.from_float(result)
        result=math.nextafter(result, math.inf if upper else -math.inf)
    raise ValueError('directed square-root construction exhausted')


def _upper_float(value):
    result=float(value)
    if not math.isfinite(result):raise ValueError('bound conversion exhausted')
    for _ in range(4):
        if F.from_float(result)>=value:return result
        result=math.nextafter(result,math.inf)
    raise ValueError('bound conversion exhausted')


def _norm_range(points):
    """Exact filled-triangle squared radius, including edge/face interiors."""
    candidates=[_dot(p,p) for p in points]
    for a,b in zip(points,points[1:]+points[:1]):
        edge=_sub(b,a);den=_dot(edge,edge)
        if den:
            t=max(F(0),min(F(1),-_dot(a,edge)/den))
            point=tuple(x+t*y for x,y in zip(a,edge));candidates.append(_dot(point,point))
    a,b,c=points;u,v=_sub(b,a),_sub(c,a)
    uu,uv,vv=_dot(u,u),_dot(u,v),_dot(v,v);au,av=_dot(a,u),_dot(a,v)
    den=uu*vv-uv*uv
    if den:
        s,t=(uv*av-vv*au)/den,(uv*au-uu*av)/den
        if s>=0 and t>=0 and s+t<=1:
            point=tuple(x+s*y+t*z for x,y,z in zip(a,u,v));candidates.append(_dot(point,point))
    return min(candidates),max(_dot(p,p) for p in points)


def _area2(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _cover(triangles, polygon):
    """Exact positive triangle chain with a once-only convex boundary.

    For any point off the finite edge set, the sum of positive triangle winding
    numbers equals the winding number of the uncancelled boundary. Complete
    once-only polygon boundary gives multiplicity one inside/zero outside:
    this proves no positive-area overlap or gap, rather than area alone.
    """
    edges=defaultdict(list)
    for tri in triangles:
        if len(set(tri))!=3 or not _area2(*tri):raise ValueError('collapsed chart triangle')
        for point in tri:
            if any(_area2(a,b,point)<0 for a,b in zip(polygon,polygon[1:]+polygon[:1])):
                raise ValueError('chart triangle leaves analytic patch')
        if _area2(*tri)<0:tri=(tri[0],tri[2],tri[1])
        for a,b in zip(tri,tri[1:]+tri[:1]):edges[tuple(sorted((a,b)))].append((a,b))
    intervals=defaultdict(list)
    for uses in edges.values():
        if len(uses)==2 and uses[0]==tuple(reversed(uses[1])):continue
        if len(uses)!=1:raise ValueError('chart overlap/duplicate/nonconforming edge')
        a,b=uses[0];assigned=False
        for i,(start,end) in enumerate(zip(polygon,polygon[1:]+polygon[:1])):
            if _area2(start,end,a) or _area2(start,end,b):continue
            axis=0 if end[0]!=start[0] else 1
            ta,tb=(a[axis]-start[axis])/(end[axis]-start[axis]),(b[axis]-start[axis])/(end[axis]-start[axis])
            if not 0<=ta<tb<=1:raise ValueError('chart boundary reverses or leaves patch')
            intervals[i].append((ta,tb));assigned=True;break
        if not assigned:raise ValueError('uncovered interior chart boundary')
    for i in range(len(polygon)):
        cursor=F(0)
        for a,b in sorted(intervals[i]):
            if a!=cursor:raise ValueError('patch boundary gap or positive overlap')
            cursor=b
        if cursor!=1:raise ValueError('patch boundary incomplete')
    return {'triangles':len(triangles),'exact_edges':len(edges),'complete_nonoverlapping_cover':True}


def _closed(vertices, faces):
    edges=defaultdict(list);links=[defaultdict(list) for _ in vertices]
    for face in faces:
        if len(set(face))!=3:raise ValueError('degenerate source triangle indices')
        a,b,c=face
        for u,v in ((a,b),(b,c),(c,a)):edges[min(u,v),max(u,v)].append(1 if u<v else -1)
        for center,u,v in ((a,b,c),(b,c,a),(c,a,b)):
            links[center][u].append(v);links[center][v].append(u)
    if any(sorted(uses)!=[-1,1] for uses in edges.values()):raise ValueError('source not closed with consistent outward indexing')
    if len(vertices)-len(edges)+len(faces)!=2:raise ValueError('source is not a closed sphere-topology mesh')
    seen=set();pending=[0];adj=defaultdict(set)
    for u,v in edges:adj[u].add(v);adj[v].add(u)
    while pending:
        u=pending.pop()
        if u not in seen:seen.add(u);pending.extend(adj[u]-seen)
    if len(seen)!=len(vertices):raise ValueError('source has disconnected/loose vertices')
    for link in links:
        if not link or any(len(ns)!=2 for ns in link.values()):raise ValueError('source vertex link is not a cycle')
        seen=set();pending=[next(iter(link))]
        while pending:
            u=pending.pop()
            if u not in seen:seen.add(u);pending.extend(set(link[u])-seen)
        if len(seen)!=len(link):raise ValueError('source vertex link has multiple cycles')


def _edge_cover(rows, snapped, directions, axes, core, intervals):
    """Two identical angular rings with exact two-triangle cell inventories."""
    other=next(i for i in range(3) if i not in axes)
    ids={i for face in rows for i in face};rays={};labels={}
    for i in ids:
        q=tuple(abs(directions[i][axis]) for axis in axes);den=sum(q)
        if den<=0:raise ValueError('edge ray missing')
        t=q[1]/den;z=snapped[i][other]
        if z not in (-core[other],core[other]):raise ValueError('edge has extra axial ring')
        label=(t,1 if z>0 else -1)
        if label in labels.values():raise ValueError('duplicate edge ring vertex')
        labels[i]=label
        if t in rays and rays[t]!=q:raise ValueError('edge angular rings differ')
        rays[t]=q
    ts=sorted(rays)
    if len(ts)!=intervals+1 or ts[0]!=0 or ts[-1]!=1 or len(ids)!=2*len(ts):
        raise ValueError('edge angular lattice incomplete')
    cells=defaultdict(list)
    for face in rows:
        tri=tuple(labels[i] for i in face);angles=sorted(set(x[0] for x in tri))
        if len(angles)!=2 or ts.index(angles[1])!=ts.index(angles[0])+1:
            raise ValueError('edge triangle skips angular cell')
        cells[ts.index(angles[0])].append(frozenset(tri))
    for cell in range(intervals):
        lo,hi=ts[cell:cell+2];a,b,c,d=(lo,-1),(lo,1),(hi,-1),(hi,1)
        pairs={frozenset((frozenset((a,b,c)),frozenset((b,c,d)))),
               frozenset((frozenset((a,b,d)),frozenset((a,c,d))))}
        if len(cells[cell])!=2 or frozenset(cells[cell]) not in pairs:
            raise ValueError('edge cell gap/overlap/diagonal mismatch')
    return {'triangles':len(rows),'angular_intervals':intervals,'axial_rings':2,'complete_nonoverlapping_cover':True}


def _proof(vertices, faces, *, core, ideal_core, radius, edge_intervals, snap_max=SNAP_MAX):
    """Portable proof core; public API below restricts the original source hash."""
    vertices=np.asarray(vertices);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)>MAX_VERTICES or len(vertices)<4
            or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>MAX_FACES
            or not np.isfinite(vertices).all() or np.abs(vertices).max()>16
            or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('source exceeds bounded array contract')
    raw=[tuple(F.from_float(float(x)) for x in v) for v in vertices]
    faces=[tuple(map(int,f)) for f in faces];_closed(raw,faces)
    snapped=[];max_snap2=F(0)
    for point in raw:
        snap=tuple((c if x>=0 else -c) if abs(abs(x)-c)<=snap_max else x for x,c in zip(point,core))
        max_snap2=max(max_snap2,_dot(_sub(point,snap),_sub(point,snap)));snapped.append(snap)
    directions=[tuple((1 if x>=0 else -1)*max(F(0),abs(x)-c) for x,c in zip(point,core)) for point in snapped]
    patches=defaultdict(list);max_radial=F(0);max_vertex_radial=F(0);max_tangent2=F(0)
    for face in faces:
        points=[raw[i] for i in face];normal=_cross(_sub(points[1],points[0]),_sub(points[2],points[0]))
        sp=[snapped[i] for i in face];sn=_cross(_sub(sp[1],sp[0]),_sub(sp[2],sp[0]))
        qs=[directions[i] for i in face]
        axes=tuple(i for i in range(3) if any(q[i] for q in qs))
        if not axes:raise ValueError('source face has no analytic patch')
        signs=tuple(1 if sp[0][i]>0 else -1 for i in axes)
        if any(any((1 if point[i]>0 else -1)!=sign for point in sp) for i,sign in zip(axes,signs)):
            raise ValueError('source face crosses analytic patch sign seam')
        if len(axes)==1:
            ds=[tuple(F(signs[0]) if i==axes[0] else F(0) for i in range(3))]*3
            if any(abs(q[axes[0]])!=abs(qs[0][axes[0]]) for q in qs):raise ValueError('flat source face not planar')
        else:ds=qs
        for d in ds:
            dot=_dot(normal,d)
            if dot<=0 or _dot(sn,d)<=0:raise ValueError('source face is not positively outward oriented')
            cross=_cross(normal,d);max_tangent2=max(max_tangent2,_dot(cross,cross)/(dot*dot))
        for q in qs:
            squared=_dot(q,q)
            max_vertex_radial=max(max_vertex_radial,radius-_sqrt(squared,False),
                                  _sqrt(squared,True)-radius)
        low2,high2=_norm_range(qs)
        if low2<=0:raise ValueError('analytic correspondence crosses a zero radial direction')
        max_radial=max(max_radial,radius-_sqrt(low2,False),_sqrt(high2,True)-radius)
        patches[axes,signs].append(face)
    expected={(axes,signs) for n in (1,2,3) for axes in itertools.combinations(range(3),n)
              for signs in itertools.product((-1,1),repeat=n)}
    if set(patches)!=expected:raise ValueError('required26 analytic patches missing')
    covers=[]
    for (axes,signs),rows in sorted(patches.items()):
        if len(axes)==1:
            other=[i for i in range(3) if i not in axes]
            chart=[tuple(tuple(snapped[i][axis] for axis in other) for i in face) for face in rows]
            a,b=(core[i] for i in other);polygon=[(-a,-b),(a,-b),(a,b),(-a,b)]
            cover=_cover(chart,polygon);kind='planar_rectangle'
        elif len(axes)==2:
            cover=_edge_cover(rows,snapped,directions,axes,core,edge_intervals);kind='cylindrical_strip'
        else:
            chart=[]
            for face in rows:
                points=[]
                for i in face:
                    q=tuple(abs(x) for x in directions[i]);den=sum(q)
                    points.append((q[0]/den,q[1]/den))
                chart.append(tuple(points))
            cover=_cover(chart,[(F(0),F(0)),(F(1),F(0)),(F(0),F(1))]);kind='spherical_octant'
        covers.append({'kind':kind,'axes':list(axes),'signs':list(signs),**cover})
    shift2=_dot(_sub(core,ideal_core),_sub(core,ideal_core))
    snap,shift=_sqrt(max_snap2,True),_sqrt(shift2,True)
    distance=max_radial+snap+shift
    from .torus_reference import _pi_interval
    pi_lower,_=_pi_interval();normal_degrees=_sqrt(max_tangent2,True)*180/pi_lower
    if normal_degrees>=90:raise ValueError('normal cone bound is not positively oriented')
    return {'status':'certified','protocol':PROTOCOL,'complete_analytic_patches':26,'patch_cover':covers,
            'maximum_source_facet_distance_world':_upper_float(distance),
            'maximum_normal_angle_degrees':_upper_float(normal_degrees),
            'radial_facet_distance_bound_world':_upper_float(max_radial),
            'maximum_vertex_radial_construction_error_world':_upper_float(max_vertex_radial),
            'radial_bound_includes_vertex_construction':True,
            'maximum_vertex_snapping_shift_world':_upper_float(snap),
            'native_to_ideal_core_shift_world':_upper_float(shift),
            'affine_construction_shift_bound_world':_upper_float(snap+shift),
            'construction_distance_bound_world':_upper_float(max_vertex_radial+snap+shift),
            'construction_scope':'native vertex radial error plus snapping/core shift, source-only; radial error already included in the facet radial bound, not added twice',
            'normal_correspondence':'same patch radial/axial correspondence; actual source cross-product normals, not custom shading normals or nearest-normal theorem',
            'distance_correspondence':'bidirectional continuous facet/analytic patch coverage; filled-triangle radial bounds plus affine construction shifts',
            'arithmetic':'exact binary64 rational coordinates/cover; rationally checked directed sqrt and Machin pi; atan(t)<=t',
            'source_closed_oriented_indexing':True,'positive_actual_normal_cones':True,
            'self_intersection_qualification':'not a native boundary qualification; actual source indexing/outward cones and analytic cover only',
            'candidate_boundary_qualified':False,'artist_limits':None,'aggregate_accepted':False,'sampled':False}


def rounded_box_reference_certificate(reference):
    """Refuse changed source bytes, holes/winding/geometry or proof exhaustion."""
    try:
        from blender_blocking.reconstruction.native_geometry import GeometryArrays
        actual=GeometryArrays.capture(reference.vertices,reference.faces)
        if reference.content_hash!=actual.content_hash or actual.content_hash!=FROZEN_SOURCE_HASH:
            raise ValueError('source differs from the original frozen indexed rounded-box identity')
        if len(actual.vertices)!=488 or len(actual.faces)!=972:raise ValueError('frozen source inventory differs')
        result=_proof(actual.vertices,actual.faces,core=NATIVE_CORE,ideal_core=IDEAL_CORE,
                      radius=RADIUS,edge_intervals=8)
        result['reference_geometry_hash']=actual.content_hash
        result['interpretation']={'radius_exact_decimal':'.15','inset_half_extents_exact_decimal':['.65','.40','.50'],'geometry':'authored BEVEL8; weighted normals separate'}
        result['historical_exposure']='source diagnostics were inspected before freeze; no candidate values are proof inputs'
        return result
    except (ValueError,TypeError,OverflowError,IndexError,ZeroDivisionError) as exc:
        return {'status':'unsupported','protocol':PROTOCOL,'reason':str(exc),
                'reference_geometry_hash':getattr(reference,'content_hash',None),
                'candidate_boundary_qualified':False,'artist_limits':None,'aggregate_accepted':False}
