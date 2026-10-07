"""Opt-in native CPU query batches. No candidate selection depends on these proxies."""
from __future__ import annotations
import numpy as np
from .native_geometry import NativeOwnedGeometry, GeometryCache


def _query_batch(target, positions, *, directions=None, near=0., far=100., proximity=False, _resident=None):
    import bpy
    p=np.asarray(positions,dtype=np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("query positions must be finite Nx3")
    if len(p)>262144:raise ValueError("query batch exceeds bounded point budget")
    if not proximity:
        d=np.asarray(directions,dtype=np.float32)
        if d.shape!=p.shape or not np.isfinite(d).all():raise ValueError("directions must match positions")
        norm=np.linalg.norm(d,axis=1)
        if (norm<=0).any() or not (0<=near<far):raise ValueError("directions/near/far invalid")
        d=d/norm[:,None];p=p+d*near
    if _resident is not None and _resident._state is not None:
        state = _resident._state
        if state['count'] != len(p) or state['proximity'] != proximity:
            _resident.close()
    state = None if _resident is None else _resident._state
    if state is None:
        target_owner=target if isinstance(target,NativeOwnedGeometry) else NativeOwnedGeometry(target,"QueryTarget")
        own_target=target_owner is not target
        was_linked=bool(target_owner.obj.users_collection)
        target_owner.attach()
        group=None;mesh=None;obj=None;temporary=None;evaluated=None
    else:
        target_owner, own_target, was_linked = state['target_owner'], state['own_target'], state['was_linked']
        group, mesh, obj = state['group'], state['mesh'], state['obj']
        temporary = evaluated = None
    try:
        if state is None:
            mesh=bpy.data.meshes.new("QueryPoints");mesh.vertices.add(len(p));mesh.vertices.foreach_set("co",p.ravel());mesh.update()
            obj=bpy.data.objects.new("QueryPoints",mesh);bpy.context.collection.objects.link(obj)
            if not proximity:
                attr=mesh.attributes.new("query_direction","FLOAT_VECTOR","POINT")
                attr.data.foreach_set("vector",d.ravel())
            group=bpy.data.node_groups.new("NativeQueryBatch","GeometryNodeTree")
            group.interface.new_socket(name="Geometry",in_out="INPUT",socket_type="NodeSocketGeometry")
            group.interface.new_socket(name="Geometry",in_out="OUTPUT",socket_type="NodeSocketGeometry")
            inputs=group.nodes.new("NodeGroupInput");output=group.nodes.new("NodeGroupOutput")
            info=group.nodes.new("GeometryNodeObjectInfo");info.transform_space="RELATIVE"
            info.inputs["Object"].default_value=target_owner.obj
            info.inputs["As Instance"].default_value=False
            position=group.nodes.new("GeometryNodeInputPosition")
            query=group.nodes.new("GeometryNodeProximity" if proximity else "GeometryNodeRaycast")
            if proximity:
                query.target_element="FACES"
                group.links.new(info.outputs["Geometry"],query.inputs["Geometry"])
                group.links.new(position.outputs["Position"],query.inputs["Sample Position"])
                outputs=(("native_distance","FLOAT",query.outputs["Distance"]),
                         ("native_position","FLOAT_VECTOR",query.outputs["Position"]),
                         ("native_hit","BOOLEAN",query.outputs["Is Valid"]))
            else:
                direction=group.nodes.new("GeometryNodeInputNamedAttribute");direction.data_type="FLOAT_VECTOR"
                direction.inputs["Name"].default_value="query_direction"
                group.links.new(info.outputs["Geometry"],query.inputs["Target Geometry"])
                group.links.new(position.outputs["Position"],query.inputs["Source Position"])
                group.links.new(direction.outputs["Attribute"],query.inputs["Ray Direction"])
                query.inputs["Ray Length"].default_value=float(far-near)
                outputs=(("native_distance","FLOAT",query.outputs["Hit Distance"]),
                         ("native_position","FLOAT_VECTOR",query.outputs["Hit Position"]),
                         ("native_normal","FLOAT_VECTOR",query.outputs["Hit Normal"]),
                         ("native_hit","BOOLEAN",query.outputs["Is Hit"]))
            geometry=inputs.outputs["Geometry"]
            for name,dtype,socket in outputs:
                store=group.nodes.new("GeometryNodeStoreNamedAttribute");store.domain="POINT";store.data_type=dtype
                store.inputs["Name"].default_value=name
                group.links.new(geometry,store.inputs["Geometry"]);group.links.new(socket,store.inputs["Value"])
                geometry=store.outputs["Geometry"]
            group.links.new(geometry,output.inputs["Geometry"])
            modifier=obj.modifiers.new("NativeQueryBatch","NODES");modifier.node_group=group
            if _resident is not None:
                _resident._state = dict(target_owner=target_owner, own_target=own_target,
                    was_linked=was_linked, group=group, mesh=mesh, obj=obj,
                    count=len(p), proximity=proximity)
        else:
            outputs = state['outputs']
            query = state['query']
            mesh.vertices.foreach_set('co', p.ravel())
            if not proximity:
                mesh.attributes['query_direction'].data.foreach_set('vector', d.ravel())
                query.inputs['Ray Length'].default_value = float(far-near)
            mesh.update()
        if _resident is not None:
            _resident.stats['calls'] += 1
            _resident.stats['graph_builds' if state is None else 'query_bulk_updates'] += 1
            _resident._state.update(outputs=outputs, query=query)
        # No deprecated modifier ID-property socket assignments: interface/RNA sockets above.
        bpy.context.view_layer.update()
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());temporary=evaluated.to_mesh()
        result={}
        for name,dtype,_ in outputs:
            attr=temporary.attributes.get(name)
            if attr is None:raise RuntimeError("native output attribute missing: "+name)
            vector=dtype=="FLOAT_VECTOR"
            values=np.empty(len(p)*3 if vector else len(p),dtype=np.float32 if dtype!="BOOLEAN" else np.bool_)
            attr.data.foreach_get("vector" if vector else "value",values)
            result[name]=values.reshape(-1,3) if vector else values
        hit=result['native_hit']
        if not proximity:result['native_distance']=np.where(hit,result['native_distance']+near,np.inf)
        else:result['native_distance']=np.where(hit,result['native_distance'],np.inf)
        result['contract']={"version":"native_point_queries_v1","units":"input_world_units",
                            "near":near,"far":far,"no_hit":"infinite_distance; hit_false",
                            "coverage":"point samples; not raster edge/antialias coverage",
                            "coordinate_frame":"identity query object; target world-coordinate mesh",
                            "implementation":"GeometryNodesProximity" if proximity else "GeometryNodesRaycast"}
        return result
    except BaseException:
        if _resident is not None and _resident._state is not None:
            if temporary is not None:
                evaluated.to_mesh_clear()
                temporary = None
            _resident.close()
            obj = mesh = group = None
            own_target = False
            was_linked = True
        raise
    finally:
        if temporary is not None:evaluated.to_mesh_clear()
        if _resident is None or _resident._state is None:
            if obj is not None:bpy.data.objects.remove(obj,do_unlink=True)
            if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:bpy.data.node_groups.remove(group)
            if own_target:target_owner.release()
            elif not was_linked:target_owner.detach()



def raycast_batch(target,origins,directions,*,near=0.,far=100.):
    return _query_batch(target,origins,directions=directions,near=near,far=far)


def proximity_batch(target,positions):
    return _query_batch(target,positions,proximity=True)


def scalar_raycast(data,origins,directions,*,near=0.,far=100.,cache=None):
    from mathutils import Vector
    tree=(cache or GeometryCache()).scalar_bvh(data)
    origins=np.asarray(origins,float);directions=np.asarray(directions,float)
    distances=[];hits=[]
    for p,d in zip(origins,directions):
        norm=np.linalg.norm(d)
        if norm<=0:raise ValueError("zero direction")
        d=d/norm;hit,_,_,distance=tree.ray_cast(Vector(p+d*near),Vector(d),far-near)
        hits.append(hit is not None);distances.append(distance+near if hit is not None else np.inf)
    return {"native_hit":np.asarray(hits,bool),"native_distance":np.asarray(distances)}


def orthographic_pixel_rays(bounds,width,height,*,horizontal_axis,vertical_axis,depth_axis,depth,direction_sign):
    """Pixel centers, top-down image rows; caller supplies the near-plane depth."""
    if len({horizontal_axis,vertical_axis,depth_axis})!=3 or direction_sign not in (-1,1):
        raise ValueError("orthographic axes and direction invalid")
    xmin,xmax,ymin,ymax=map(float,bounds)
    u=xmin+(np.arange(width)+.5)/width*(xmax-xmin)
    v=ymax-(np.arange(height)+.5)/height*(ymax-ymin)
    uu,vv=np.meshgrid(u,v,indexing="xy")
    origins=np.zeros((width*height,3));directions=np.zeros_like(origins)
    origins[:,horizontal_axis]=uu.ravel();origins[:,vertical_axis]=vv.ravel();origins[:,depth_axis]=depth
    directions[:,depth_axis]=direction_sign
    return origins,directions


class ResidentQueryBatch:
    """Retain target geometry and GN query graph across same-size fit batches."""
    def __init__(self, target):
        self.target = target
        self._state = None
        self.stats = {'calls': 0, 'graph_builds': 0, 'target_coordinate_updates': 0,
                      'target_topology_rebuilds': 0, 'query_bulk_updates': 0}

    def update_target(self, target):
        if self._state is None:
            self.target = target
            return
        from .native_geometry import geometry_arrays
        action = self._state['target_owner'].update(geometry_arrays(target))
        if action == 'coordinates_updated':
            self.stats['target_coordinate_updates'] += 1
        elif action == 'topology_rebuilt':
            self.stats['target_topology_rebuilds'] += 1
        self.target = target

    def proximity(self, positions):
        try:
            return _query_batch(self.target, positions, proximity=True, _resident=self)
        except BaseException:
            self.close()
            raise

    def raycast(self, origins, directions, *, near=0., far=100.):
        try:
            return _query_batch(self.target, origins, directions=directions,
                                near=near, far=far, _resident=self)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._state is None:
            return
        import bpy
        state, self._state = self._state, None
        bpy.data.objects.remove(state['obj'], do_unlink=True)
        if state['mesh'].users == 0:
            bpy.data.meshes.remove(state['mesh'])
        if state['group'].users == 0:
            bpy.data.node_groups.remove(state['group'])
        if state['own_target']:
            state['target_owner'].release()
        elif not state['was_linked']:
            state['target_owner'].detach()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
