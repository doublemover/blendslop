"""Editable local polygon extrusion with explicit holes and an exact capped field.

Optional cap triangulation uses Shapely 2.1 / GEOS constrained Delaunay. Missing
or unsupported triangulation fails clearly; no unconstrained cap fills a hole.
"""
from __future__ import annotations
import numpy as np
from .primitive_protocol import MeshData,normalize_rotation


def signed_area(loop):
    loop=np.asarray(loop,float)
    return .5*float(np.sum(loop[:,0]*np.roll(loop[:,1],-1)-loop[:,1]*np.roll(loop[:,0],-1)))


def checked_polygon(outer,holes=()):
    try:
        from shapely import Polygon,constrained_delaunay_triangles
    except ImportError as exc:
        raise RuntimeError('polygon extrusion requires optional Shapely>=2.1 constrained triangulation') from exc
    loops=[]
    for index,raw in enumerate((outer,*holes)):
        loop=np.asarray(raw,float)
        if loop.ndim!=2 or loop.shape[1]!=2 or len(loop)<3 or not np.isfinite(loop).all():
            raise ValueError('polygon loops must be finite Mx2 arrays with at least three vertices')
        if np.array_equal(loop[0],loop[-1]):
            loop=loop[:-1]
        if len(loop)<3 or np.any(np.all(loop==np.roll(loop,1,axis=0),axis=1)):
            raise ValueError('polygon loop has repeated/collapsed edges')
        if (signed_area(loop)>0.) != (index==0):
            loop=loop[::-1]
        loops.append(loop.copy())
    polygon=Polygon(loops[0],loops[1:])
    if not polygon.is_valid or polygon.is_empty or polygon.area<=0.:
        raise ValueError('polygon is invalid, self-crossing or has unsupported touching holes')
    return polygon,loops


def triangulate_polygon(outer,holes=()):
    from shapely import constrained_delaunay_triangles
    polygon,loops=checked_polygon(outer,holes)
    vertices=np.concatenate(loops)
    lookup={tuple(point):index for index,point in enumerate(vertices)}
    faces=[]
    for triangle in constrained_delaunay_triangles(polygon).geoms:
        points=np.asarray(triangle.exterior.coords[:-1],float)
        if len(points)!=3 or any(tuple(p) not in lookup for p in points):
            raise ValueError('constrained cap introduced an unsupported new vertex')
        ids=[lookup[tuple(point)] for point in points]
        if signed_area(points)<0.:
            ids[1],ids[2]=ids[2],ids[1]
        faces.append(tuple(ids))
    if not faces:
        raise ValueError('constrained cap triangulation is empty')
    points=vertices[np.asarray(faces)]
    areas=.5*np.abs((points[:,1,0]-points[:,0,0])*(points[:,2,1]-points[:,0,1])-
                       (points[:,1,1]-points[:,0,1])*(points[:,2,0]-points[:,0,0]))
    scale=max(float(np.ptp(vertices,axis=0).max())**2,polygon.area)
    if abs(float(areas.sum())-polygon.area)>np.finfo(float).eps*scale*max(32,len(faces)*8):
        raise ValueError('constrained cap does not cover the declared polygon area')
    return vertices,tuple(faces),loops


def polygon_signed_distance(points,loops):
    points=np.asarray(points,float)
    distance=np.full(len(points),np.inf);inside=np.zeros(len(points),bool)
    for loop_index,loop in enumerate(loops):
        a=loop;b=np.roll(loop,-1,axis=0);edge=b-a
        squared=np.sum(edge*edge,axis=1)
        if np.any(squared<=0.):
            raise ValueError('polygon distance cannot use collapsed edges')
        loop_inside=np.zeros(len(points),bool)
        for start in range(0,len(points),512):
            p=points[start:start+512]
            delta=p[:,None,:]-a[None,:,:]
            fraction=np.clip(np.einsum('nmi,mi->nm',delta,edge)/squared,0.,1.)
            projected=a[None,:,:]+fraction[...,None]*edge[None,:,:]
            distance[start:start+len(p)]=np.minimum(distance[start:start+len(p)],
                np.sqrt(np.sum((p[:,None,:]-projected)**2,axis=2)).min(axis=1))
            crossing=(a[:,1]>p[:,None,1])!=(b[:,1]>p[:,None,1])
            denominator=np.where(edge[:,1]!=0.,edge[:,1],1.)
            x=a[:,0]+(p[:,None,1]-a[:,1])*edge[:,0]/denominator
            loop_inside[start:start+len(p)]=np.count_nonzero(crossing & (p[:,None,0]<x),axis=1)%2==1
        inside=loop_inside if loop_index==0 else inside & ~loop_inside
    return np.where(inside,-distance,distance)


class PolygonExtrusionPrimitive:
    def __init__(self,outer,holes=(),*,center=(0.,0.,0.),rotation=None,height=.1,scale_xy=(1.,1.)):
        _,loops=checked_polygon(outer,holes)
        self.outer,self.holes=loops[0],tuple(loops[1:])
        self.center=np.asarray(center,float)
        self.rotation=normalize_rotation(np.eye(3) if rotation is None else rotation)
        self.height=float(height);self.scale_xy=np.asarray(scale_xy,float)
        if (self.center.shape!=(3,) or not np.isfinite(self.center).all() or self.scale_xy.shape!=(2,) or
            not np.isfinite(self.scale_xy).all() or np.any(self.scale_xy<=0.) or
            not np.isfinite(self.height) or self.height<=0.):
            raise ValueError('polygon extrusion requires finite pose and positive scales/height')

    def local_loops(self):
        return [loop*self.scale_xy for loop in (self.outer,*self.holes)]

    def sdf_batch(self,points):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('extrusion field points must be finite Nx3')
        if not len(points):
            return np.empty(0)
        local=(points-self.center)@self.rotation
        profile=polygon_signed_distance(local[:,:2],self.local_loops())
        q=np.column_stack((profile,np.abs(local[:,2])-self.height*.5))
        return np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(q.max(axis=1),0.)

    def to_mesh_data(self,resolution=16):
        loops=self.local_loops()
        points,caps,loops=triangulate_polygon(loops[0],loops[1:])
        count=len(points)
        vertices=np.vstack((np.column_stack((points,np.full(count,-self.height*.5))),
                            np.column_stack((points,np.full(count,self.height*.5)))))
        faces=[(c,b,a) for a,b,c in caps]+[(a+count,b+count,c+count) for a,b,c in caps]
        offset=0
        for loop in loops:
            for index in range(len(loop)):
                a=offset+index;b=offset+(index+1)%len(loop)
                faces.append((a,b,b+count,a+count))
            offset+=len(loop)
        return MeshData(vertices@self.rotation.T+self.center,tuple(faces))

    def sample_surface(self,n):
        if n<=0:
            return np.empty((0,3))
        mesh=self.to_mesh_data()
        faces=np.asarray([(f[0],f[i],f[i+1]) for f in mesh.faces for i in range(1,len(f)-1)])
        a,b,c=(mesh.vertices[faces[:,i]] for i in range(3))
        areas=.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
        indices=np.searchsorted(np.cumsum(areas),(np.arange(n)+.5)/n*areas.sum())
        root=np.sqrt(np.mod((np.arange(n)+.5)*.6180339887498949,1.))
        second=np.mod((np.arange(n)+.5)*.4142135623730951,1.)
        return (1.-root[:,None])*a[indices]+root[:,None]*((1.-second[:,None])*b[indices]+second[:,None]*c[indices])

    def to_mesh(self,resolution=16):
        return self.to_mesh_data(resolution)

    def to_dict(self):
        return {'type':'polygon_extrusion','outer':self.outer.tolist(),'holes':[h.tolist() for h in self.holes],
                'center':self.center.tolist(),'rotation':self.rotation.tolist(),'height':self.height,
                'scale_xy':self.scale_xy.tolist()}

    @classmethod
    def from_dict(cls,parameters):
        return cls(parameters['outer'],parameters.get('holes',()),center=parameters.get('center',(0.,0.,0.)),
            rotation=parameters.get('rotation'),height=parameters['height'],scale_xy=parameters.get('scale_xy',(1.,1.)))
