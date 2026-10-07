"""Local rigid-frame edits shared by program proposals and refinement."""
from __future__ import annotations
from copy import deepcopy
import numpy as np

POSITION_KEYS = ("x", "y", "z")
DIMENSION_KEYS = ("width_world", "depth_world", "height_world")


def position_vector(parameters):
    return np.asarray([parameters.get(key,parameters.get('location_'+key,0.)) for key in POSITION_KEYS],float)


def _write_position(parameters,position):
    parameters.update(zip(POSITION_KEYS,np.asarray(position,float).tolist()))
    for key,value in zip(POSITION_KEYS,position):
        if 'location_'+key in parameters:parameters['location_'+key]=float(value)


def rotation_matrix(parameters):
    if "rotation" in parameters:
        matrix = np.asarray(parameters["rotation"], dtype=float)
    elif 'rotation_row_major' in parameters:
        values=np.asarray(parameters['rotation_row_major'],float)
        if values.size!=9:raise ValueError('row-major rotation requires nine values')
        matrix=values.reshape(3,3)
    else:
        from scipy.spatial.transform import Rotation
        matrix = Rotation.from_euler("xyz", parameters.get("rotation_euler", (0., 0., 0.))).as_matrix()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("program rotation must be a finite 3x3 matrix")
    if (not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0.)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-8, rtol=0.)):
        raise ValueError("program rotation must describe a proper orthonormal frame")
    return matrix


def local_pose_edit(parameters, *, translation=None, rotation_increment=None):
    """Translate and compose tangent rotations in the existing local frame."""
    result = deepcopy(dict(parameters))
    frame = rotation_matrix(parameters)
    if translation is not None:
        position = position_vector(parameters)
        position += frame @ np.asarray(translation, float)
        _write_position(result,position)
    if rotation_increment is not None and np.any(rotation_increment):
        from scipy.spatial.transform import Rotation
        result["rotation"] = (frame @ Rotation.from_rotvec(rotation_increment).as_matrix()).tolist()
        if 'rotation_row_major' in result:result['rotation_row_major']=np.asarray(result['rotation']).ravel().tolist()
        result.pop("rotation_euler", None)
    return result


def reflect_parameters(parameters, *, axis=0, plane=0.):
    """Reflect supported local geometry while retaining a proper frame.

    The second reflection is in local X. Boxes and centered analytic solids are
    unchanged by that local sign flip. Profile/sweep centers and polygon loops
    are reflected explicitly; unsupported absolute point clouds fail closed.
    """
    result = deepcopy(dict(parameters))
    if 'points_world' in result:
        raise ValueError('absolute point-cloud programs require a geometry-specific world reflection')
    reflection = np.eye(3); reflection[axis, axis] = -1.
    local_reflection = np.diag([-1., 1., 1.])
    position=position_vector(parameters);position[axis]=2.*plane-position[axis]
    _write_position(result,position)
    result["rotation"] = (reflection @ rotation_matrix(parameters) @ local_reflection).tolist()
    if 'rotation_row_major' in result:result['rotation_row_major']=np.asarray(result['rotation']).ravel().tolist()
    if 'bend_angle' in result:
        result['bend_angle'] = -float(result['bend_angle'])
    if 'section_knots_normalized' in result:
        knots=np.asarray(result['section_knots_normalized'],float).copy();knots[:,1]*=-1.
        result['section_knots_normalized']=knots.tolist()
    if 'outer' in result:
        for key in ('outer','holes'):
            if key not in result:continue
            loops=[result[key]] if key=='outer' else result[key]
            transformed=[]
            for loop in loops:
                values=np.asarray(loop,float).copy();values[:,0]*=-1.;transformed.append(values.tolist())
            result[key]=transformed[0] if key=='outer' else transformed
    if 'profile_curve' in result:
        curve=[dict(row) for row in result['profile_curve']]
        for row in curve:
            if 'center_offset_world' in row:row['center_offset_world']*=-1.
        result['profile_curve']=curve
    result.pop("rotation_euler", None)
    return result
