"""Cheap numeric proposal geometry, with actual rendering left to admission."""
from __future__ import annotations
from dataclasses import replace
import numpy as np


def whole_program_geometry(program, *, resolution=16):
    from primitives.analytic_primitives import EllipsoidPrimitive
    from primitives.superfrustum import SuperFrustum
    from .native_geometry import GeometryArrays
    from .program_transforms import rotation_matrix, position_vector, DIMENSION_KEYS
    if len(program.root_nodes) != 1:
        from .grouped_solids import concatenate
        return concatenate([whole_program_geometry(replace(program,root_nodes=(node,)),resolution=resolution)
                            for node in program.root_nodes])
    node = program.root_nodes[0]; params = node.parameters
    center = position_vector(params)
    sizes = np.asarray([params.get(k, 1.) for k in DIMENSION_KEYS], float)
    frame = rotation_matrix(params)
    if node.primitive_type == 'generalized_sweep':
        from primitives.generalized_sweep import GeneralizedSweepPrimitive
        mesh=GeneralizedSweepPrimitive.from_program_parameters(params).to_mesh_data(resolution)
        vertices,faces=mesh.vertices,mesh.faces
    elif node.primitive_type == 'deformed_superquadric':
        from primitives.deformed_superquadric import DeformedSuperquadricPrimitive
        mesh = DeformedSuperquadricPrimitive.from_program_parameters(params).to_mesh_data(resolution)
        vertices, faces = mesh.vertices, mesh.faces
    elif node.primitive_type == "polygon_extrusion":
        from primitives.polygon_extrusion import PolygonExtrusionPrimitive
        outline=np.asarray(params['outer'],float)
        scaling=sizes[:2]/np.ptp(outline,axis=0)
        mesh=PolygonExtrusionPrimitive(outline,params.get('holes',()),center=center,rotation=frame,
                                       height=sizes[2],scale_xy=scaling).to_mesh_data()
        vertices,faces=mesh.vertices,mesh.faces
    elif node.primitive_type == "box":
        local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)*sizes*.5
        vertices = local @ frame.T + center
        faces = ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    elif node.primitive_type == "ellipsoid":
        mesh = EllipsoidPrimitive(center=center, radii=sizes*.5, rotation=frame).to_mesh_data(resolution)
        vertices, faces = mesh.vertices, mesh.faces
    elif node.primitive_type in {"cylinder", "frustum"}:
        axis = frame[:,2]
        mesh = SuperFrustum(position=center,
            orientation=(float(np.arctan2(axis[1],axis[0])),float(np.arccos(np.clip(axis[2],-1.,1.)))),
            radius_bottom=float(params.get('radius_bottom',sizes[0]*.5)),
            radius_top=float(params.get('radius_top',sizes[0]*.5)),height=sizes[2]).to_mesh_data(resolution)
        vertices, faces = mesh.vertices, mesh.faces
    else:
        raise ValueError("whole-program primitive is not supported by numeric screening")
    triangles = np.asarray([(f[0], f[i], f[i+1]) for f in faces for i in range(1,len(f)-1)], int)
    return GeometryArrays.capture(vertices,triangles)


def coarse_target(target, *, width=64):
    """Center-consistent coverage and conservative observed fraction."""
    import cv2
    from .visibility import valid_evidence
    from .projection_contract import pixel_cell_viewport
    from .types import OrthographicCameraSpec, Bounds2D
    constraints = []
    for c in target.constraints:
        mask = np.asarray(getattr(c.mask, "mask", c.mask), bool)
        h,w = mask.shape; output = (int(width),max(8,round(h/w*width)))
        valid = valid_evidence(c)
        known_fraction = cv2.resize(valid.astype(np.float32),output,interpolation=cv2.INTER_AREA)
        coverage = cv2.resize((mask & valid).astype(np.float32),output,interpolation=cv2.INTER_AREA)
        _, (u0,u1,v0,v1) = pixel_cell_viewport(target,c)
        camera = getattr(c,"camera",None)
        if getattr(camera,"bounds",None) is None:
            # The legacy viewport intervals locate endpoint pixel centers.
            # Convert them to complete pixel-cell bounds before direct resizing.
            camera = OrthographicCameraSpec(c.view, {'front':'Y','side':'X','top':'Z'}[c.view],
                bounds=Bounds2D(u0,v0,u1,v1))
        constraints.append(replace(c,mask=coverage >= .5,valid_mask=known_fraction >= 1.-1e-6,camera=camera))
    return replace(target,constraints=tuple(constraints))


def screen_whole_programs(target, programs):
    from .projected_metrics import projected_mesh_metrics
    from .feature_evidence import mesh_empty_features
    coarse = coarse_target(target)
    ranked = []
    for program in programs:
        data = whole_program_geometry(program)
        metrics = projected_mesh_metrics(coarse,data.vertices,data.faces)
        observed = [r for r in metrics.values() if r.get('required',True) and r.get('area_iou') is not None]
        features = mesh_empty_features(coarse,data)
        if not observed or not features['passed']:
            continue
        key = (min(r['area_iou'] for r in observed),float(np.mean([r['area_iou'] for r in observed])))
        ranked.append((key,program,{'coarse_min_iou':key[0],'coarse_mean_iou':key[1],
            'known_empty_features':features,'screen_scope':'numeric coarse proposal; actual final render required'}))
    return sorted(ranked,key=lambda row:row[0],reverse=True)
