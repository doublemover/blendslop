"""Dedicated controlled geometry occupancy, independent of display transforms.

Opaque target override + transparent film yields filtered linear alpha. Retain
float coverage and a half-coverage mask; do not decode tone-mapped RGB. Human
preview render settings/materials are restored after the measurement session.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import numpy as np


@contextmanager
def controlled_measurement_session(*, scene=None, target_objects=None, camera=None,
                                   resolution=(512, 512), samples=64, filter_size=1.5):
    import bpy
    from .silhouette_render import silhouette_session
    if (isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 256
            or len(resolution) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
            or not 16 <= v <= 2048 for v in resolution)
            or not np.isfinite(filter_size) or not .01 <= filter_size <= 2.):
        raise ValueError('bounded positive measurement resolution/sampling/filter required')
    scene = scene or bpy.context.scene
    if camera is None:
        camera = scene.camera
    if camera is None:
        raise ValueError('measurement requires an explicitly configured camera')
    render = scene.render
    owners = [(render, key) for key in ('filepath', 'use_compositing', 'use_sequencer',
              'dither_intensity', 'filter_size', 'use_border', 'use_crop_to_border', 'use_motion_blur')
              if hasattr(render, key)]
    owners += [(render.image_settings, key) for key in ('color_depth', 'exr_codec')]
    owners += [(camera.data.dof, 'use_dof')]
    eevee = getattr(scene, 'eevee', None)
    if eevee is None or not hasattr(eevee, 'taa_render_samples'):
        raise RuntimeError('controlled EEVEE sample setting is unavailable')
    owners += [(eevee, 'taa_render_samples')]
    previous = [(owner, key, getattr(owner, key)) for owner, key in owners]
    original_world = scene.world
    measurement_world = bpy.data.worlds.new('BlendslopMeasurementWorld')
    scene.world = measurement_world
    try:
        for key in ('use_compositing', 'use_sequencer', 'use_border', 'use_crop_to_border', 'use_motion_blur'):
            if hasattr(render, key):
                setattr(render, key, False)
        render.dither_intensity = 0.
        render.filter_size = float(filter_size)
        camera.data.dof.use_dof = False
        eevee.taa_render_samples = samples
        with silhouette_session(scene=scene, target_objects=target_objects, camera=camera,
                resolution=resolution, color_mode='RGBA', transparent_bg=True,
                engine='BLENDER_EEVEE', force_material=True, ensure_light_obj=False,
                hide_non_targets=True) as session:
            render.image_settings.file_format = 'OPEN_EXR'
            render.image_settings.color_depth = '32'
            render.image_settings.exr_codec = 'ZIP'
            yield session
    finally:
        scene.world = original_world
        bpy.data.worlds.remove(measurement_world)
        for owner, key, value in previous:
            setattr(owner, key, value)


def render_controlled_measurement(session, output_path):
    """Render a linear EXR, returning top-left float coverage and its contract.

    The EXR is retained. RGB can vary with display settings; only its alpha is
    used. Opaque coverage is a filtered raster estimate, not an exact hidden edge.
    """
    import bpy
    from .silhouette_render import render_silhouette_frame
    from evaluation.canonical_artifacts import camera_record
    from evaluation.controlled_measurement import PROTOCOL, coverage_and_mask, measurement_signature
    from reconstruction.native_geometry import evaluated_arrays
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.exr' or output_path.exists():
        raise ValueError('measurement needs a fresh .exr output path')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before = sorted(evaluated_arrays(obj).content_hash for obj in session.target_objects)
    scene = session.scene
    if (scene.render.image_settings.file_format != 'OPEN_EXR' or
            scene.render.image_settings.color_depth != '32' or not scene.render.film_transparent
            or scene.render.use_compositing or scene.render.use_sequencer):
        raise ValueError('measurement session settings changed before render')
    render_silhouette_frame(session, output_path)
    image = bpy.data.images.load(str(output_path), check_existing=False)
    try:
        width, height = image.size
        if (width, height) != (scene.render.resolution_x, scene.render.resolution_y):
            raise ValueError('linear measurement image dimensions changed')
        pixels = np.empty(width*height*4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        coverage, mask = coverage_and_mask(pixels.reshape(height, width, 4)[::-1, :, 3])
    finally:
        bpy.data.images.remove(image)
    after = sorted(evaluated_arrays(obj).content_hash for obj in session.target_objects)
    if before != after:
        raise ValueError('controlled rendering changed evaluated geometry')
    contract = {'protocol': PROTOCOL, 'hard_mask_threshold': .5,
        'camera': camera_record(session.camera, resolution=(width, height),
            pixel_aspect=(scene.render.pixel_aspect_x, scene.render.pixel_aspect_y)),
        'renderer': {'engine': scene.render.engine, 'blender_version': list(bpy.app.version),
                     'build_hash': bpy.app.build_hash.decode('ascii')},
        'sampling': {'taa_render_samples': scene.eevee.taa_render_samples,
                     'filter_size': scene.render.filter_size, 'motion_blur': False, 'depth_of_field': False},
        'pixel_convention': 'top-left array; pixel (i,j) center=(j+.5,i+.5)',
        'surface_policy': 'opaque target meshes only; forced emission; no compositor/sequencer/world volume',
        'encoding': 'float32 OpenEXR linear alpha; display RGB ignored'}
    return {'contract': contract, 'contract_sha256': measurement_signature(contract),
            'geometry_hashes': before, 'geometry_unchanged': True,
            'exr_sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
            'coverage': coverage, 'mask': mask}
