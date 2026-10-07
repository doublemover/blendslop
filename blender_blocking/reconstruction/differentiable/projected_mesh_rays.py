"""Filtered opaque mesh projections with active-topology vertex pullbacks.

GEOS resolves the projected triangle union. Each boundary vertex is mapped to
an original projected vertex or an edge intersection; Torch differentiates that
fixed active topology and the pinned DVX 2D box integral. Topology events are
nonsmooth, and unsupported carriers fail rather than inventing gradients.
"""
from __future__ import annotations
import itertools
import time
import numpy as np

AXES={'front':(0,2),'side':(1,2),'top':(0,1)}


def projected_boundary_carriers(projected,faces,*,maximum_faces=2048):
    from shapely import Polygon,union_all
    from shapely.geometry.polygon import orient
    projected=np.asarray(projected,float);faces=np.asarray(faces,np.int64)
    if (projected.ndim!=2 or projected.shape[1]!=2 or not np.isfinite(projected).all() or
        faces.ndim!=2 or faces.shape[1]!=3 or len(faces)>maximum_faces or
        np.any(faces<0) or np.any(faces>=len(projected))):
        raise ValueError('projected mesh requires finite 2D vertices and at most 2048 valid triangles')
    polygons=[]
    for face in faces:
        points=projected[face]
        first,second=points[1]-points[0],points[2]-points[0]
        area=first[0]*second[1]-first[1]*second[0]
        if area!=0.:polygons.append(Polygon(points))
    if not polygons:raise ValueError('mesh has no nonzero projected area')
    union=union_all(polygons)
    if union.is_empty or not union.is_valid:
        raise ValueError('projected opaque union is invalid')
    components=list(union.geoms) if union.geom_type=='MultiPolygon' else [union]
    if len(components)>32:raise ValueError('projected component allowance exceeded')
    loops=[]
    for polygon in components:
        polygon=orient(polygon,sign=1.)
        loops.extend([np.asarray(polygon.exterior.coords[:-1],float),
                      *[np.asarray(ring.coords[:-1],float) for ring in polygon.interiors]])
    if sum(map(len,loops))>4096:raise ValueError('projected boundary allowance exceeded')
    lookup={}
    for index,point in enumerate(projected):lookup.setdefault(tuple(point),index)
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    starts=projected[edges[:,0]];vectors=projected[edges[:,1]]-starts
    lengths=np.einsum('ij,ij->i',vectors,vectors)
    scale=max(1.,float(np.abs(projected).max()));tolerance=np.finfo(float).eps*scale*256.
    carriers=[];coordinates=[];connectivity=[];maximum_reconstruction_error=0.
    for loop in loops:
        offset=len(coordinates)
        for point in loop:
            key=tuple(point)
            if key in lookup:
                carriers.append(('vertex',int(lookup[key])))
            else:
                delta=point-starts
                cross=vectors[:,0]*delta[:,1]-vectors[:,1]*delta[:,0]
                fraction=np.divide(np.einsum('ij,ij->i',delta,vectors),lengths,
                    out=np.full(len(edges),np.inf),where=lengths>0.)
                active=np.flatnonzero((np.abs(cross)<=tolerance*np.sqrt(lengths))&
                    (fraction>=-tolerance)&(fraction<=1.+tolerance)&(lengths>0.))
                found=None
                for first,second in itertools.combinations(active,2):
                    a,b=starts[first],starts[second];r,s=vectors[first],vectors[second]
                    denominator=r[0]*s[1]-r[1]*s[0]
                    if abs(denominator)<=np.finfo(float).eps*np.sqrt(lengths[first]*lengths[second])*32.:continue
                    diff=b-a;factor=(diff[0]*s[1]-diff[1]*s[0])/denominator
                    rebuilt=a+factor*r;error=float(np.max(np.abs(rebuilt-point)))
                    if error<=tolerance*8.:
                        found=('intersection',*map(int,edges[first]),*map(int,edges[second]))
                        maximum_reconstruction_error=max(maximum_reconstruction_error,error);break
                if found is None:
                    raise ValueError('projected union boundary has no supported vertex/edge-intersection carrier')
                carriers.append(found)
            coordinates.append(point)
        connectivity.extend((offset+index,offset+(index+1)%len(loop)) for index in range(len(loop)))
    return np.asarray(coordinates),np.asarray(connectivity,np.int64),carriers,{
        'components':len(components),'loops':len(loops),'boundary_vertices':len(coordinates),
        'maximum_carrier_reconstruction_error':maximum_reconstruction_error,
        'gradient_scope':'fixed active projected-union topology; coincident ties use deterministic subgradients'}


def torch_boundary(projected,faces,*,maximum_faces=2048):
    import torch
    coordinates,edges,carriers,report=projected_boundary_carriers(projected.detach().numpy(),faces,
        maximum_faces=maximum_faces)
    values=[]
    cross=lambda first,second:first[0]*second[1]-first[1]*second[0]
    for carrier in carriers:
        if carrier[0]=='vertex':values.append(projected[carrier[1]])
        else:
            _,a,b,c,d=carrier
            start=projected[a];r=projected[b]-start;s=projected[d]-projected[c]
            factor=cross(projected[c]-start,s)/cross(r,s)
            values.append(start+factor*r)
    boundary=torch.stack(values)
    error=float(np.max(np.abs(boundary.detach().numpy()-coordinates)))
    tolerance=512.*torch.finfo(projected.dtype).eps
    if not np.isfinite(error) or error>tolerance:
        raise ValueError('Torch projected boundary differs from resolved opaque union')
    return boundary,torch.tensor(edges,dtype=torch.int64),{**report,'torch_boundary_error':error}


def projected_mesh_grids(vertices,faces,resolution,views,*,maximum_faces=2048):
    import dvx.torch as dvx
    started=time.perf_counter();result={};reports={}
    face_array=faces.detach().numpy() if hasattr(faces,'detach') else np.asarray(faces,np.int64)
    for index,view in enumerate(views):
        if view not in AXES:raise ValueError('projected rays require canonical front/side/top cameras')
        print(f'DVX mesh-ray view={view} stage={index+1}/{len(views)} faces={len(face_array)} elapsed={time.perf_counter()-started:.2f}s',flush=True)
        boundary,edges,report=torch_boundary(vertices[:,AXES[view]],face_array,maximum_faces=maximum_faces)
        result[view]=dvx.voxelize(int(resolution),boundary,edges,method='cf')
        reports[view]=report
    return result,reports
