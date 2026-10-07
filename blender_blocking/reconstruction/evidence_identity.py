"""Content identity for masks, visibility and fixed projection conventions."""
from __future__ import annotations
import hashlib
import json
import numpy as np


def target_evidence_hash(target):
    from .projection_contract import pixel_cell_viewport
    from .visibility import valid_evidence
    digest=hashlib.sha256()
    for constraint in sorted(target.constraints,key=lambda c:c.view):
        mask=np.asarray(getattr(constraint.mask,'mask',constraint.mask),bool)
        valid=valid_evidence(constraint)
        digest.update(json.dumps({'view':constraint.view,'shape':mask.shape,
            'viewport':pixel_cell_viewport(target,constraint),'projection_convention':'complete_pixel_cells_v2',
            'camera':constraint.camera.to_dict() if hasattr(constraint.camera,'to_dict') else None},sort_keys=True).encode())
        digest.update(np.ascontiguousarray(mask & valid).tobytes())
        digest.update(np.ascontiguousarray(valid).tobytes())
        for name in ('foreground_prob','confidence','boundary_uncertainty'):
            values=getattr(getattr(constraint,'uncertainty',None),name,None)
            digest.update(name.encode())
            if values is not None:
                values=np.asarray(values,dtype=np.float64)
                # Legacy nonspatial summaries keep their own identity; actual
                # maps must not hash unobserved private/unreliable pixel values.
                if values.shape==mask.shape:values=np.where(valid,values,0.)
                digest.update(str(values.shape).encode())
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
