"""A genuinely rendered orthographic crop selected only from fitting residuals."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .controlled_measurement import coverage_and_mask


def residual_crop_request(reference, candidate, camera, *, pixel_span=160):
    ref, _ = coverage_and_mask(reference)
    cand, _ = coverage_and_mask(candidate)
    if ref.shape != cand.shape or ref.shape[0] != ref.shape[1]:
        raise ValueError('initial residual crop requires matched square acquisitions')
    size = ref.shape[0]
    if (isinstance(pixel_span, bool) or not isinstance(pixel_span, int)
            or not 16 <= pixel_span < size):
        raise ValueError('crop span must be a bounded smaller pixel-cell interval')
    matrix = np.asarray(camera['matrix_world'], float)
    scale = float(camera['ortho_scale'])
    if (camera.get('projection') != 'ORTHO' or camera.get('shift_x', 0.) != 0.
            or camera.get('shift_y', 0.) != 0. or matrix.shape != (4, 4)
            or not np.isfinite(matrix).all() or not np.isfinite(scale) or scale <= 0.
            or tuple(camera['resolution']) != (size, size)
            or camera.get('pixel_aspect', [1., 1.]) != [1., 1.]):
        raise ValueError('initial crop requires a finite unshifted square orthographic camera')
    residual = np.abs(ref-cand)
    if not residual.any():
        return {'status': 'unneeded', 'reason': 'no fitting-view coverage residual'}
    gy, gx = np.gradient(ref)
    diagonal = np.minimum(np.abs(gx), np.abs(gy))
    score = residual*diagonal
    if not score.any():
        score = residual
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    x0 = int(np.clip(x-pixel_span//2, 0, size-pixel_span))
    y0 = int(np.clip(y-pixel_span//2, 0, size-pixel_span))
    center = np.array([(x0+pixel_span/2)/size-.5,
                       .5-(y0+pixel_span/2)/size])*scale
    cropped = deepcopy(camera)
    matrix[:3, 3] += matrix[:3, :2] @ center
    cropped['matrix_world'] = matrix.tolist()
    cropped['ortho_scale'] = scale*pixel_span/size
    # Output resolution is a separate acquisition choice; never enlarge pixels
    # of the old image and call them newly observed information.
    return {'status': 'requested', 'camera': cropped,
        'source_pixel_cell_bounds': [x0, y0, x0+pixel_span, y0+pixel_span],
        'selected_pixel': [int(x), int(y)], 'selection_score': float(score[y, x]),
        'selection': 'fitting-view alpha residual weighted by diagonal reference edge gradient',
        'source_geometry_available_required': True,
        'scope': 'crop renders only; independent validation views are untouched'}
