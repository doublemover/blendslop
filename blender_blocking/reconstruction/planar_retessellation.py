"""Exact-plane retessellation using original vertices and boundary chains.

No tolerance groups nearly coplanar faces, welds a gap or moves a coordinate.
Original meshes remain separate artifacts and actual output qualification is new.
"""
from __future__ import annotations
from collections import defaultdict
import math,time
import numpy as np
from .native_geometry import GeometryArrays


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _plane(vertices,face):
    a,b,c=(vertices[int(i)] for i in face)
    normal=_cross(tuple(b[i]-a[i] for i in range(3)),tuple(c[i]-a[i] for i in range(3)))
    if not any(normal):
        raise ValueError('exact planar retessellation cannot repair degenerate input faces')
    divisor=math.gcd(*normal)
    normal=tuple(value//divisor for value in normal)
    return (*normal,-sum(normal[i]*a[i] for i in range(3)))


def retessellate_coplanar(data,*,timeout_s=5.,progress=None):
    from blender_blocking.evaluation.triangle_contacts import integer_vertices
    from blender_blocking.primitives.polygon_extrusion import triangulate_polygon
    from .grouped_solids import solid_guard
    started=time.perf_counter();original_guard=solid_guard(data)
    if not original_guard['valid_solid']:
        return data,{'status':'unchanged','reason':'input topology/orientation/volume screen failed'}
    exact=integer_vertices(data.vertices)
    keys=[];incidences=defaultdict(list);vertex_planes=defaultdict(set)
    def check(stage,count,total):
        if progress is not None:
            progress(stage,count,total,time.perf_counter()-started)
        if timeout_s is not None and time.perf_counter()-started>=timeout_s:
            raise TimeoutError('planar retessellation allowance exhausted; original retained')
    try:
        for index,face in enumerate(data.faces):
            key=_plane(exact,face);keys.append(key)
            for vertex in face:
                vertex_planes[int(vertex)].add(key)
            for a,b in zip(face,np.roll(face,-1)):
                incidences[tuple(sorted((int(a),int(b))))].append(index)
            if index%10000==0:
                check('exact_planes',index+1,len(data.faces))
        neighbors=[[] for _ in data.faces]
        for edge,faces in incidences.items():
            if len(faces)!=2:
                raise ValueError('planar input requires closed two-face edge incidence')
            a,b=faces
            if keys[a]==keys[b]:
                neighbors[a].append(b);neighbors[b].append(a)
        remaining=set(range(len(data.faces)));outputs=[];patches=changed=0
        first_order=iter(range(len(data.faces)))
        while remaining:
            first=next(first_order)
            while first not in remaining:
                first=next(first_order)
            remaining.remove(first);queue=[first];patch=[first]
            while queue:
                face=queue.pop()
                for other in neighbors[face]:
                    if other in remaining:
                        remaining.remove(other);queue.append(other);patch.append(other)
            patches+=1
            if patches%256==0:
                check('planar_patches',len(data.faces)-len(remaining),len(data.faces))
            if len(patch)<=2:
                outputs.extend(data.faces[patch]);continue
            patch_set=set(patch);boundary=[]
            for face_index in patch:
                face=data.faces[face_index]
                for a,b in zip(face,np.roll(face,-1)):
                    if any(other not in patch_set for other in incidences[tuple(sorted((int(a),int(b))))]):
                        boundary.append((int(a),int(b)))
            successors={a:b for a,b in boundary}
            if len(successors)!=len(boundary) or len({b for a,b in boundary})!=len(boundary):
                outputs.extend(data.faces[patch]);continue
            loops=[];unvisited=set(successors)
            valid=True
            while unvisited:
                start=min(unvisited);loop=[];vertex=start
                while vertex in unvisited:
                    unvisited.remove(vertex);loop.append(vertex);vertex=successors[vertex]
                if vertex!=start or len(loop)<3:
                    valid=False;break
                loops.append(loop)
            if not valid:
                outputs.extend(data.faces[patch]);continue
            normal=np.asarray(keys[first][:3],dtype=object)
            dropped=int(np.argmax(np.abs(normal)));axes=[i for i in range(3) if i!=dropped]
            direction=int(np.sign(normal[dropped]))*(-1 if dropped==1 else 1)
            simplified=[]
            for loop in loops:
                again=True
                while again and len(loop)>3:
                    again=False;retained=[]
                    for i,vertex in enumerate(loop):
                        before,after=loop[i-1],loop[(i+1)%len(loop)]
                        a,b,c=(exact[j] for j in (before,vertex,after))
                        first_edge=tuple(b[k]-a[k] for k in range(3));second_edge=tuple(c[k]-b[k] for k in range(3))
                        removable=(not any(_cross(first_edge,second_edge)) and
                            sum(first_edge[k]*second_edge[k] for k in range(3))>0 and
                            len(vertex_planes[vertex])<=2 and
                            vertex_planes[vertex].issubset(vertex_planes[before]) and
                            vertex_planes[vertex].issubset(vertex_planes[after]))
                        if removable:
                            again=True
                        else:
                            retained.append(vertex)
                    if len(retained)<3:
                        valid=False;break
                    loop=retained
                simplified.append(loop)
            if not valid:
                outputs.extend(data.faces[patch]);continue
            def twice_area(loop):
                points=[exact[i] for i in loop]
                return sum(a[axes[0]]*b[axes[1]]-a[axes[1]]*b[axes[0]]
                           for a,b in zip(points,points[1:]+points[:1]))*direction
            outer=[loop for loop in simplified if twice_area(loop)>0]
            holes=[loop for loop in simplified if twice_area(loop)<0]
            if len(outer)!=1:
                outputs.extend(data.faces[patch]);continue
            ordered=[outer[0],*holes]
            source_ids=[i for loop in ordered for i in loop]
            projected=[data.vertices[loop][:,axes] for loop in ordered]
            try:
                cap_vertices,caps,_=triangulate_polygon(projected[0],projected[1:])
                # checked_polygon normalizes loop direction; map by exact
                # original projected coordinates rather than positional order.
                mapping={tuple(data.vertices[i,axes]):i for i in source_ids}
                if len(mapping)!=len(source_ids):
                    raise ValueError('coincident boundary vertices cannot be welded by retessellation')
                candidate=[]
                for face in caps:
                    ids=[mapping[tuple(cap_vertices[i])] for i in face]
                    a,b,c=(exact[i] for i in ids)
                    cross=_cross(tuple(b[k]-a[k] for k in range(3)),tuple(c[k]-a[k] for k in range(3)))
                    if sum(cross[k]*int(normal[k]) for k in range(3))<0:
                        ids[1],ids[2]=ids[2],ids[1]
                    candidate.append(tuple(ids))
                area=sum(twice_area(list(face)) for face in candidate)
                original_area=sum(twice_area(list(data.faces[i])) for i in patch)
                if area!=original_area or any(twice_area(list(face))<=0 for face in candidate):
                    raise ValueError('retessellated triangles do not preserve exact oriented patch area')
                if len(candidate)>=len(patch):
                    outputs.extend(data.faces[patch]);continue
                outputs.extend(candidate);changed+=1
            except (ValueError,RuntimeError,ImportError):
                outputs.extend(data.faces[patch])
        check('final_boundary',len(data.faces),len(data.faces))
        if not changed:
            return data,{'status':'unchanged','reason':'no reducible exact planar patches','planar_patches':patches}
        faces=np.asarray(outputs,int);used=np.unique(faces)
        inverse=np.full(len(data.vertices),-1,int);inverse[used]=np.arange(len(used))
        result=GeometryArrays.capture(data.vertices[used],inverse[faces])
        guard=solid_guard(result)
        if not guard['valid_solid']:
            raise ValueError('retessellated output failed closed orientation/topology/volume screen')
        return result,{'status':'retessellated','input_hash':data.content_hash,'output_hash':result.content_hash,
            'input_faces':len(data.faces),'output_faces':len(result.faces),'input_vertices':len(data.vertices),
            'output_vertices':len(result.vertices),'planar_patches':patches,'changed_patches':changed,
            'coordinate_changes':0,'exact_patch_area_preserved':True,'geometric_tolerance_used':False,
            'scope':'exact planes and original boundary vertices; final boundary qualification remains separate',
            'single_solid_qualified':False,'elapsed_s':time.perf_counter()-started}
    except (TimeoutError,ValueError) as exc:
        return data,{'status':'unchanged','reason':str(exc),'elapsed_s':time.perf_counter()-started}
