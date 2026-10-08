"""Full union silhouette objective for editable primitive meshes, no confidence-as-IoU."""
from __future__ import annotations
import numpy as np


def mesh_union_silhouette_hook(target, resolution=64):
    import cv2
    from reconstruction.mesh_io import mesh_arrays_from_object
    from reconstruction.projection_contract import project_vertices
    from blender_blocking.reconstruction.visibility import valid_evidence
    records=[]
    for c in target.constraints:
        mask=np.asarray(c.mask,bool)
        h,w=mask.shape
        out_w=max(16, int(resolution)); out_h=max(16,round(h/w*out_w))
        reference=cv2.resize(mask.astype(np.uint8),(out_w,out_h),interpolation=cv2.INTER_NEAREST).astype(bool)
        valid=cv2.resize(valid_evidence(c).astype(np.uint8),(out_w,out_h),interpolation=cv2.INTER_NEAREST).astype(bool)
        if valid.any():
            records.append((c,reference,valid))
    def hook(primitives):
        meshes=[mesh_arrays_from_object(p.to_mesh_data(10)) for p in primitives]
        overlaps=[]
        for constraint,ref,valid in records:
            canvas=np.zeros(ref.shape,np.uint8)
            for verts,faces in meshes:
                xy=project_vertices(target,constraint,verts,output_shape=ref.shape)
                polygons=[np.round(xy[list(face)]).astype(np.int32) for face in faces if len(face)>=3]
                # Draw faces independently: OpenCV's even/odd multi-contour rule would punch false holes in overlapping faces.
                for polygon in polygons:
                    cv2.fillConvexPoly(canvas,polygon,1)
            pred=canvas.astype(bool) & valid
            ref=ref & valid
            union=np.count_nonzero(ref|pred)
            overlaps.append(np.count_nonzero(ref&pred)/union if union else 1.)
        return {'mesh_union':float(2.-np.mean(overlaps)-min(overlaps))} if overlaps else {}
    return hook
