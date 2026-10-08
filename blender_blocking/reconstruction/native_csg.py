"""Separately opt-in native CSG algorithms; these can change reconstruction geometry."""
from __future__ import annotations
import numpy as np
from .native_geometry import NativeOwnedGeometry, evaluated_arrays, GeometryCache


def boolean_mesh(left,right,*,operation="INTERSECT",solver="EXACT",qualification=None):
    import bpy
    if operation not in ("INTERSECT","UNION","DIFFERENCE") or solver not in ("EXACT","MANIFOLD"):
        raise ValueError("unsupported native Boolean configuration")
    if solver=="MANIFOLD":
        q=qualification or {}
        identity = q.get('_toolchain_identity')
        from .native_qualification import toolchain_identity
        if not identity or not q.get('_qualification_python'):
            raise ValueError('Manifold qualification must declare its toolchain identity and interpreter')
        if identity != toolchain_identity(q['_qualification_python']):
            raise ValueError('Manifold receipt belongs to a different toolchain')
        if not identity:
            raise ValueError("Manifold qualification must declare its toolchain identity")
        for data in (left,right):
            r=q.get(data.content_hash,{})
            if not (r.get('manifold_validated') is True and r.get('self_intersections')==0 and
                    r.get('geometry_content_hash')==data.content_hash and
                    r.get('toolchain_identity')==identity):
                raise ValueError("Manifold solver requires actual hash-matched native manifold qualification")
    a=NativeOwnedGeometry(left,"CSGLeft");b=NativeOwnedGeometry(right,"CSGRight")
    try:
        a.attach();b.attach()
        mod=a.obj.modifiers.new("ExperimentalCSG","BOOLEAN")
        mod.operation=operation;mod.solver=solver;mod.object=b.obj
        bpy.context.view_layer.update()
        result=evaluated_arrays(a.obj)
        return result,{"algorithm":"native_boolean","operation":operation,"solver":solver,
                       "same_output_optimization":False}
    finally:
        a.release();b.release()


def sdf_grid_mesh(left,right,*,operation="INTERSECT",resolution=128):
    import bpy
    if resolution not in (128,256):raise ValueError("SDF ladder is bounded to 128/256")
    if operation not in ("INTERSECT","UNION","DIFFERENCE"):raise ValueError("invalid SDF operation")
    cache=GeometryCache()
    for data in (left,right):
        report=cache.topology_report(data)
        if not report['watertight'] or report['connected_components']!=1:
            raise ValueError("experimental SDF inputs require one watertight component")
    combined=np.vstack([left.vertices,right.vertices])
    longest=float(np.ptp(combined,axis=0).max())
    voxel=longest/resolution
    a=NativeOwnedGeometry(left,"SDFLeft");b=NativeOwnedGeometry(right,"SDFRight")
    host=None;mesh=None;group=None
    try:
        a.attach();b.attach()
        mesh=bpy.data.meshes.new("SDFHost");host=bpy.data.objects.new("SDFHost",mesh)
        bpy.context.collection.objects.link(host)
        group=bpy.data.node_groups.new("ExperimentalSDFCSG","GeometryNodeTree")
        group.interface.new_socket(name="Geometry",in_out="OUTPUT",socket_type="NodeSocketGeometry")
        output=group.nodes.new("NodeGroupOutput");grids=[]
        for owner in (a,b):
            info=group.nodes.new("GeometryNodeObjectInfo");info.transform_space="RELATIVE"
            info.inputs['Object'].default_value=owner.obj;info.inputs['As Instance'].default_value=False
            node=group.nodes.new("GeometryNodeMeshToSDFGrid")
            node.inputs['Voxel Size'].default_value=voxel;node.inputs['Band Width'].default_value=3
            group.links.new(info.outputs['Geometry'],node.inputs['Mesh']);grids.append(node.outputs['SDF Grid'])
        combine=group.nodes.new("GeometryNodeSDFGridBoolean");combine.operation=operation
        # UNION/INTERSECT expose one multi-input socket named Grid. Its RNA
        # identifier remains Grid 2; DIFFERENCE also enables the first input.
        grid_input = next(socket for socket in combine.inputs if socket.identifier == 'Grid 2')
        if operation == 'DIFFERENCE':
            first = next(socket for socket in combine.inputs if socket.identifier == 'Grid 1')
            group.links.new(grids[0], first)
            group.links.new(grids[1], grid_input)
        else:
            for grid in grids:
                group.links.new(grid, grid_input)
        extract=group.nodes.new("GeometryNodeGridToMesh");extract.inputs['Adaptivity'].default_value=0
        extract.inputs['Threshold'].default_value=0.0  # SDF zero level, not the volume-density default.
        group.links.new(combine.outputs['Grid'],extract.inputs['Grid'])
        group.links.new(extract.outputs['Mesh'],output.inputs['Geometry'])
        mod=host.modifiers.new("ExperimentalSDFCSG","NODES");mod.node_group=group
        bpy.context.view_layer.update();result=evaluated_arrays(host)
        return result,{"algorithm":"native_sdf_grid_csg","operation":operation,"resolution":resolution,
                       "voxel_size":voxel,"band_width_voxels":3,"same_output_optimization":False,
                       "limitations":"new discretization; topology, holes, thin parts and volume must be requalified"}
    finally:
        if host is not None:bpy.data.objects.remove(host,do_unlink=True)
        if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
        if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        a.release();b.release()
