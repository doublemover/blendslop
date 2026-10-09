#!/usr/bin/env python3
"""Three bounded renders proving display-independent controlled occupancy."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking')]


def main():
    import bpy
    import numpy as np
    import test_runner
    from utils.run_ownership import OwnedRun
    from integration.blender_ops.measurement_render import (controlled_measurement_session,
                                                           render_controlled_measurement)
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from reconstruction.native_geometry import evaluated_arrays
    output = Path(sys.argv[sys.argv.index('--')+1]).resolve()
    if not output.is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('validation output must stay in the isolated project temp/tasks')
    owner = OwnedRun(output, producer='controlled_measurement_validation', max_generated_bytes=16777216)
    with owner:
        scene = bpy.context.scene
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.object
        obj.scale = (.8, .6, .2)
        obj.rotation_euler.z = math.radians(13.)
        bpy.ops.object.camera_add(location=(0, 0, 4))
        camera = bpy.context.object
        camera.data.type = 'ORTHO'; camera.data.ortho_scale = 2.5
        camera.rotation_euler = (0, 0, 0); scene.camera = camera
        original_world = scene.world
        original_mats = tuple(obj.data.materials)
        records = []
        for index, (transform, exposure, gamma) in enumerate((('AgX', 0., 1.), ('Standard', 3., 2.), ('AgX', 0., 1.))):
            scene.view_settings.view_transform = transform
            scene.view_settings.look = 'None'
            scene.view_settings.exposure = exposure
            scene.view_settings.gamma = gamma
            if index == 2: obj.scale.x *= 1.12
            arrays = evaluated_arrays(obj)
            np.savez_compressed(owner.root/f'geometry-{index}.npz', vertices=arrays.vertices, faces=arrays.faces)
            owner.register_file(f'geometry-{index}.npz', 'final_output')
            with controlled_measurement_session(scene=scene, target_objects=[obj], camera=camera,
                    resolution=(128, 128), samples=64) as session:
                result = render_controlled_measurement(session, owner.root/f'measurement-{index}.exr')
            if (scene.world != original_world or tuple(obj.data.materials) != original_mats or
                    scene.view_settings.view_transform != transform or scene.view_settings.exposure != exposure
                    or scene.view_settings.gamma != gamma):
                raise ValueError('controlled acquisition changed restored human-preview state')
            for name, value in (('coverage', result['coverage']), ('mask', result['mask'])):
                np.save(owner.root/f'{name}-{index}.npy', value, allow_pickle=False)
                owner.register_file(f'{name}-{index}.npy', 'final_output')
            owner.register_file(f'measurement-{index}.exr', 'final_output')
            records.append(result)
        gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
        display_pair = compare_controlled_measurements(records[0], records[1], view='top', config=gates)
        changed_pair = compare_controlled_measurements(records[0], records[2], view='top', config=gates)
        same_coverage = bool(np.array_equal(records[0]['coverage'], records[1]['coverage']))
        fractions = int(np.count_nonzero((records[0]['coverage'] > 0.) & (records[0]['coverage'] < 1.)))
        passed = (same_coverage and fractions > 0 and display_pair['passed']
                  and changed_pair['area_iou'] < 1. and all(r['geometry_unchanged'] for r in records))
        summaries = [{key: value for key, value in record.items() if key not in ('coverage', 'mask')} for record in records]
        receipt = {'status': 'passed' if passed else 'failed', 'renders': 3,
            'resolution': [128, 128], 'display_independent_exact_coverage': same_coverage,
            'fractional_edge_pixels': fractions, 'display_pair': display_pair, 'changed_geometry_pair': changed_pair,
            'preview_state_restored': True, 'records': summaries,
            'scope': 'measurement protocol fixture only; no family acceptance, fitting or surface qualification'}
        (owner.root/'results.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
        owner.register_file('results.json', 'final_output')
        if not passed: owner.mark_failed('controlled occupancy contract fixture failed')
        print('MEASUREMENT_RESULT='+str(owner.root/'results.json'), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
