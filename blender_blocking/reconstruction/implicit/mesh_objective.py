"""Extracted zero-set edge pullbacks and original-pixel opaque support.

The active marching-cubes and projected-union topology is recomputed each
forward pass. Unsupported interior carriers fail; derivatives never substitute
a fixed-mesh deformation for the actual saved-field extraction.
"""
import importlib.util
from pathlib import Path

import numpy as np

_PROJECTED = None


def projected_module():
    global _PROJECTED
    if _PROJECTED is None:
        path = Path(__file__).resolve().parents[1]/'differentiable'/'projected_mesh_rays.py'
        spec = importlib.util.spec_from_file_location('blendslop_implicit_projected_boundary', path)
        _PROJECTED = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PROJECTED)
    return _PROJECTED


def torch_extracted_zero_mesh(field, model, *, maximum_faces=2048):
    import torch
    from skimage.measure import marching_cubes
    values = field.detach().numpy().transpose(2, 1, 0)
    indices, faces, _, _ = marching_cubes(values, 0., spacing=(1.,)*3,
        method='lewiner', gradient_direction='descent', allow_degenerate=False)
    if not len(faces) or len(faces) > maximum_faces:
        raise ValueError('implicit projected extraction exceeds bounded triangle allowance')
    n = field.shape[0]
    rounded = np.rint(indices)
    integral = np.abs(indices-rounded) <= 4*np.finfo(np.float32).eps*n
    if np.any(integral.sum(axis=1) != 2):
        raise ValueError('implicit marching vertex has an unsupported interior or exact-grid carrier')
    axis = np.argmin(integral, axis=1)
    first = rounded.astype(np.int64)
    rows = np.arange(len(first))
    first[rows, axis] = np.floor(indices[rows, axis]).astype(np.int64)
    second = first.copy()
    second[rows, axis] += 1
    if np.any(first < 0) or np.any(second >= n):
        raise ValueError('implicit zero-set edge escapes fixed field domain')
    a = torch.tensor((first[:, 2]*n+first[:, 1])*n+first[:, 0], dtype=torch.int64)
    b = torch.tensor((second[:, 2]*n+second[:, 1])*n+second[:, 0], dtype=torch.int64)
    va, vb = field.reshape(-1)[a], field.reshape(-1)[b]
    if bool(((va*vb >= 0) | (va == vb)).any()):
        raise ValueError('implicit active extraction carrier does not cross the zero set')
    fraction = va/(va-vb)
    direction = torch.zeros((len(first), 3), dtype=field.dtype)
    direction[torch.arange(len(first)), torch.tensor(axis)] = 1.
    grid_coordinates = torch.tensor(first, dtype=field.dtype)+fraction[:, None]*direction
    spacing = torch.tensor(model.voxel_size_xyz.copy(), dtype=field.dtype)
    origin = torch.tensor(model.origin_xyz.copy(), dtype=field.dtype)
    vertices = origin+grid_coordinates*spacing
    exact_world = model.origin_xyz+indices*model.voxel_size_xyz
    error = float(np.max(np.abs(vertices.detach().numpy()-exact_world)))
    tolerance = 16*np.finfo(np.float32).eps*max(float(np.abs(exact_world).max()), float(spacing.max()))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError('implicit differentiable extraction differs from authoritative zero-set mesh')
    return vertices, faces.astype(np.int64), {'vertex_count': len(indices), 'face_count': len(faces),
        'maximum_world_reconstruction_error': error,
        'gradient_scope': 'fixed active marching-cubes grid-edge carriers; topology recomputed each evaluation',
        'native_qualification': False}


def original_pixel_mesh_predictions(field, model, targets):
    import torch
    import dvx.torch as dvx
    if str(dvx.__version__) != '0.1.1':
        raise RuntimeError('implicit extracted-mesh objective requires pinned DVX0.1.1')
    vertices, faces, extraction = torch_extracted_zero_mesh(field, model)
    predictions, reports = {}, {}
    axes = {'front': (0, 2), 'side': (1, 2), 'top': (0, 1)}
    for view, row in targets.items():
        n = int(row['square_resolution'])
        if not 8 <= n <= 192:
            raise ValueError('original-pixel implicit projection requires a bounded 8..192 square support')
        viewport = np.asarray(row['padded_world_bounds'], float)
        center = torch.tensor([(viewport[0]+viewport[1])*.5, (viewport[2]+viewport[3])*.5], dtype=field.dtype)
        half = torch.tensor([(viewport[1]-viewport[0])*.5, (viewport[3]-viewport[2])*.5], dtype=field.dtype)
        projected = (vertices[:, axes[view]]-center)/half
        if bool((torch.abs(projected) >= 1.).any()):
            raise ValueError('implicit projected mesh is outside fixed padded camera support')
        boundary, edges, report = projected_module().torch_boundary(projected, faces)
        predictions[view] = dvx.voxelize(n, boundary, edges, method='cf')
        reports[view] = report
    return predictions, {'extraction': extraction, 'per_view_boundary': reports,
        'operator': 'extracted zero-set opaque triangle union with original pixel-cell closed-form box support'}
