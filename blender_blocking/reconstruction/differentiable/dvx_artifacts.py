"""Replay editable DVX initialization separately from the retained deformation."""
from pathlib import Path
import numpy as np


def replay_dvx_artifacts(payload, primitives):
    from ..mesh_io import combine_primitive_meshes
    from ..grouped_solids import oriented_generated_mesh
    from ..native_geometry import GeometryArrays
    from ...evaluation.comparable_geometry import read_obj

    metadata = payload['metadata']
    if metadata.get('contract') != 'deformed_multipart_with_editable_initialization':
        raise ValueError('not a DVX initialization/deformation artifact contract')
    with np.load(metadata['deformation_state_path'], allow_pickle=False) as values:
        stored_seed = GeometryArrays.capture(values['seed_vertices'], values['faces'])
        retained = GeometryArrays.capture(values['retained_vertices'], values['faces'])
        if 'proposed_vertices' in values:
            proposed = GeometryArrays.capture(values['proposed_vertices'], values['faces'])
            if proposed.content_hash != metadata['proposed_content_hash']:
                raise ValueError('DVX proposed deformation does not match proposal identity')
    if stored_seed.content_hash != metadata['seed_content_hash']:
        raise ValueError('DVX stored seed does not match seed identity')
    if primitives:
        seed = oriented_generated_mesh(combine_primitive_meshes(primitives,
            resolution=int(metadata['initialization_resolution'])))
        if seed.vertices.shape != stored_seed.vertices.shape or not np.array_equal(seed.faces, stored_seed.faces):
            raise ValueError('DVX editable parameters do not reproduce seed connectivity')
        # from_dict re-normalizes rotations; its roundoff is not deformation.
        parameter_error = float(np.max(np.abs(seed.vertices-stored_seed.vertices)))
        parameter_tolerance = 1e-10*max(1.,float(np.max(np.abs(stored_seed.vertices))))
        if parameter_error>parameter_tolerance:
            raise ValueError('DVX editable parameters do not reproduce seed geometry')
        parameter_replay='coordinate tolerance for rotation normalization'
    else:
        if metadata.get('seed_selection',{}).get('status') != 'retained_existing_seed':
            raise ValueError('DVX empty initialization lacks an identified retained seed')
        parameter_error=parameter_tolerance=None
        parameter_replay='retained evaluated seed surface; original artist parameters remain separate'
    if metadata.get('artist_source_path'):
        import hashlib
        source=Path(metadata['artist_source_path'])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=metadata.get('artist_source_sha256'):
            raise ValueError('DVX retained artist source identity mismatch')
    if retained.content_hash != metadata['retained_content_hash']:
        raise ValueError('DVX retained deformation does not match output identity')
    errors = {}
    for name, data in (('seed_mesh_path', stored_seed), ('retained_mesh_path', retained)):
        vertices, faces = read_obj(Path(metadata[name]))
        if vertices.shape != data.vertices.shape or not np.array_equal(faces, data.faces):
            raise ValueError('DVX artifact connectivity mismatch: ' + name)
        error = float(np.max(np.abs(vertices - data.vertices)))
        if error >= 1e-7:
            raise ValueError('DVX OBJ does not match stored geometry: ' + name)
        errors[name] = error
    return {'seed': stored_seed, 'retained': retained, 'receipt': {
        'seed_content_hash': stored_seed.content_hash, 'retained_content_hash': retained.content_hash,
        'initialization_resolution': metadata['initialization_resolution'],
        'seed_parameter_replay': parameter_replay,
        'seed_parameter_max_coordinate_error': parameter_error, 'seed_parameter_tolerance': parameter_tolerance,
        'stored_geometry_identity': 'exact content hashes', 'obj_max_coordinate_errors': errors,
        'final_reconstruction_source': metadata['deformation_state_path'],
        'seed_parameters_reproduce_final_deformation': False, 'single_solid_qualified': False}}
