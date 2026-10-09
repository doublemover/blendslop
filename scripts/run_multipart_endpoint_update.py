#!/usr/bin/env python3
"""One observed multipart endpoint update, six candidate frames, one qualifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'blender_blocking'), str(ROOT / 'scripts'), str(ROOT)]
import test_runner

FAMILY = 'asymmetric_multipart_solid'
VIEW = 'complementary_minus15_20'
BASELINE_HASH = 'c367746d102dac33f3ad12b271c2ab00d46bd4da59a4bd8004a4bef7152eb0d1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from mathutils import Vector
    from evaluation.canonical_artifacts import CANONICAL_VIEWS, canonical_artifact_inventory, raw_surface_observation
    from evaluation.controlled_measurement import coverage_and_mask, compare_controlled_measurements, measurement_signature
    from evaluation.silhouette_eval import SilhouetteGateConfig, evaluate_silhouette_pair
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from reconstruction.grouped_solids import solid_guard
    from reconstruction.multipart_endpoint import refine_short_far_endpoint
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from reconstruction.native_qualification import qualify_geometry
    from utils.run_ownership import OwnedRun
    from run_frozen_multipart_reconstruction import compile_live_multipart
    from run_quality_coverage_check import render_masks, save_mesh
    from run_surface_quality_check import _replay_orthographic_camera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--observation-run', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    parent = args.output.resolve()
    if not parent.is_relative_to(ROOT / 'temp/tasks'):
        raise ValueError('endpoint experiment must stay within isolated temp/tasks')
    preflight = read_json(args.preflight)
    for filename, digest in preflight['input_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('endpoint frozen input changed: ' + filename)
    observation = read_json(args.observation_run)
    if observation['geometry']['candidate_before_and_after_hash'] != BASELINE_HASH:
        raise ValueError('new observation does not bind the frozen baseline')
    acquisition = args.observation_run.parent
    source_meta = read_json(acquisition / 'source/measurement.json')
    coverage, source_mask = coverage_and_mask(np.load(acquisition / 'source/coverage.npy', allow_pickle=False))
    if measurement_signature(source_meta['contract']) != source_meta['contract_sha256'] or coverage.shape != (512, 512):
        raise ValueError('retained source alpha acquisition changed')
    source_record = {**source_meta, 'coverage': coverage, 'mask': source_mask}
    prior = read_json(args.prepared_run)['cases'][FAMILY]
    prepared = Path(prior['artifact_directory'])
    wire = read_json(prepared / 'program.json')
    if wire != prior['program'] or prior['geometry_hash'] != BASELINE_HASH:
        raise ValueError('baseline saved recipe changed')
    old_frozen = read_json(prepared / 'frozen-workload.json')
    camera_declaration = read_json(acquisition / 'frozen-workload.json')['camera_declaration']
    frozen = {
        'protocol': 'multipart_one_endpoint_update_v1', 'family': FAMILY,
        'preflight': str(args.preflight), 'preflight_sha256': sha(args.preflight),
        'input_sha256': preflight['input_sha256'], 'baseline_geometry_hash': BASELINE_HASH,
        'fit_view': VIEW, 'fit_acquisition_sha256': source_meta['contract_sha256'],
        'fit_input': 'new float32 alpha + actual camera only; original interval, all other parameters fixed',
        'free_parameter': 'short_arm_far_y', 'fit_budget': {'residual_calls': 64, 'seconds': 1.},
        'renders': 6, 'candidate_old_display_mask_frames': 5, 'candidate_controlled_alpha_frames': 1,
        'optional_neutral_frames': 0, 'source_frames': 0, 'source_frames_reused': 1,
        'raw_comparisons': 1, 'reused_baseline_raw_observations': 1, 'samples_per_direction': 4096, 'seed': 61007,
        'qualification_children': 1, 'helper_timeout_seconds': 15., 'python': str(args.python),
        'old_gate_protocol': 'original display-calibrated AgX silhouettes; no relabeling',
        'new_gate_protocol': 'opaque_linear_alpha_v1', 'surface_limits': None,
        'heldout_scope': 'old oblique145 excluded from endpoint fitting and view choice; independent admission only',
        'prior_disclosure': 'original five images inspected for capability before holdout protocol was declared',
        'budget_seconds': 120, 'budget_rss_bytes': 8 * 1024 ** 3, 'threads': 2,
    }
    owner = OwnedRun(parent, producer='multipart_one_endpoint_update', max_generated_bytes=67108864,
                     shared_inputs={'observation': str(args.observation_run), 'baseline': str(args.prepared_run)})
    started = time.monotonic()
    with owner:
        output = owner.root

        def publish(relative, value):
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            owner.register_file(relative, 'final_output')

        publish('frozen-workload.json', frozen)
        try:
            program = refine_short_far_endpoint(wire, coverage, source_meta['contract']['camera'],
                                                max_evaluations=64, max_elapsed_s=1.)
            # Immutable observed recipe is frozen before ANY reference-array read.
            publish('program.json', program.to_dict())
            obj, parts = compile_live_multipart(program)
            data, geometry_files = save_mesh(obj, output)
            pointer, mesh_pointer = parts[2].as_pointer(), parts[2].data.as_pointer()
            scale = parts[2].scale.copy()
            excluded = bool(parts[2].get('blendslop_export_exclude', False))
            parts[2]['blendslop_export_exclude'] = False
            before_part = evaluated_arrays(parts[2])
            parts[2].scale.y *= 1.05
            bpy.context.view_layer.update()
            edited_part, edited = evaluated_arrays(parts[2]), evaluated_arrays(obj)
            ratio = float(np.ptp(edited_part.vertices[:, 1]) / np.ptp(before_part.vertices[:, 1]))
            parts[2].scale = scale
            parts[2]['blendslop_export_exclude'] = excluded
            bpy.context.view_layer.update()
            restored = evaluated_arrays(obj)
            edit = {'control': 'short arm depth x1.05', 'dimension_ratio': ratio,
                    'union_geometry_changed': edited.content_hash != data.content_hash,
                    'same_source_and_mesh_pointers': parts[2].as_pointer() == pointer and parts[2].data.as_pointer() == mesh_pointer,
                    'exact_indexed_restoration': restored.content_hash == data.content_hash,
                    'baseline_geometry_hash': data.content_hash, 'edited_geometry_hash': edited.content_hash,
                    'restored_geometry_hash': restored.content_hash}
            edit['passed'] = bool(abs(ratio - 1.05) <= 1e-5 and edit['union_geometry_changed']
                                  and edit['same_source_and_mesh_pointers'] and edit['exact_indexed_restoration'])
            if not edit['passed']:
                raise ValueError('specific short-depth control failed exact restoration')
            masks = {}
            for view in CANONICAL_VIEWS:
                path = Path(old_frozen['observed_masks'][view]['path'])
                if sha(path) != old_frozen['observed_masks'][view]['sha256']:
                    raise ValueError('old source mask identity changed')
                image = np.asarray(Image.open(path).convert('L'))
                if image.shape != (512, 512):
                    raise ValueError('old frozen source mask resolution changed')
                masks[view] = image < 128
            candidates, cameras = render_masks(obj, output,
                (Vector(data.vertices.min(axis=0)), Vector(data.vertices.max(axis=0))), CANONICAL_VIEWS,
                camera_records=old_frozen['cameras'], inspection_passes=('mask',))
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            views = {view: evaluate_silhouette_pair(masks[view], candidates[view], view=view, config=gates)
                     for view in CANONICAL_VIEWS}
            publish('camera-snapshots.json', cameras)
            inventory = canonical_artifact_inventory(output, geometry_hash=data.content_hash,
                camera_records=cameras, reference_camera_records=old_frozen['cameras'],
                pass_states={'mask': 'completed', 'neutral': 'unrun', 'normals': 'unrun'})
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.data.type = 'ORTHO'
            bpy.context.scene.camera = camera
            with controlled_measurement_session(target_objects=[obj], camera=camera,
                    resolution=(512, 512), samples=64, filter_size=1.5) as session:
                _replay_orthographic_camera(camera, camera_declaration)
                controlled = render_controlled_measurement(session, output / 'complementary.exr')
            if controlled['geometry_hashes'] != [data.content_hash] or evaluated_arrays(obj).content_hash != data.content_hash:
                raise ValueError('new candidate measurement changed exact indexed geometry')
            np.save(output / 'coverage.npy', controlled['coverage'], allow_pickle=False)
            Image.fromarray(np.rint(controlled['coverage'] * 255).astype(np.uint8)).save(output / 'coverage.png')
            publish('measurement.json', {k: v for k, v in controlled.items() if k not in ('coverage', 'mask')})
            new_view = compare_controlled_measurements(source_record, controlled, view=VIEW, config=gates)
            # Recipe and exact candidate exports above already exist. Only now
            # read retained GT geometry for independent raw comparison.
            with np.load(args.reference_root / FAMILY / 'evaluated-exact.npz', allow_pickle=False) as saved:
                reference_data = GeometryArrays.capture(saved['vertices'], saved['faces'])
            with np.load(prepared / 'evaluated-exact.npz', allow_pickle=False) as saved:
                baseline_data = GeometryArrays.capture(saved['vertices'], saved['faces'])
            if baseline_data.content_hash != BASELINE_HASH:
                raise ValueError('baseline geometry changed before independent raw comparison')
            baseline_raw = prior['surface_observation']
            if (baseline_raw['sample_count_per_direction'] != 4096 or baseline_raw['seed'] != 61007
                    or baseline_raw['candidate_geometry_hash'] != BASELINE_HASH
                    or baseline_raw['reference_geometry_hash'] != reference_data.content_hash):
                raise ValueError('retained baseline raw acquisition differs')
            raw = {'baseline': baseline_raw,
                   'refined': raw_surface_observation(reference_data, data, count=4096, seed=61007)}
            deltas = {key: float(raw['refined'][key] - raw['baseline'][key]) for key in (
                'symmetric_mean_distance_world', 'distance_p95_world', 'sampled_max_distance_world',
                'normal_angle_mean_degrees', 'normal_angle_p95_degrees') if key in raw['refined']}
            if 120 - (time.monotonic() - started) < 20:
                raise TimeoutError('insufficient fixed budget before one qualifier')
            boundary = qualify_geometry(data, python=args.python, timeout_s=15.,
                                        ownership_root=parent / 'qualification-children')
            if boundary.get('geometry_content_hash') != data.content_hash:
                raise ValueError('qualification receipt lost candidate geometry identity')
            all_old = all(row['passed'] for row in views.values())
            diagnostic_admission = bool(all_old and new_view['passed'] and
                new_view['coverage_l1'] < observation['observation']['coverage_l1'])
            row = {'status': 'incomplete', 'artifact_directory': str(output), **geometry_files,
                'program': program.to_dict(), 'silhouette': {'status': 'passed' if all_old else 'failed', 'views': views},
                'controlled_new_view': new_view, 'source_edit': edit, 'topology_screen': solid_guard(data),
                'boundary_qualification': boundary, 'raw_surface': raw, 'raw_surface_delta': deltas,
                'family_surface_limits': None, 'canonical_artifacts': inventory,
                'independent_endpoint_admission': diagnostic_admission, 'aggregate_accepted': False}
            publish('results.json', {'protocol': frozen['protocol'], 'status': 'measured',
                'cases': {FAMILY: row}, 'baseline_geometry_hash': BASELINE_HASH,
                'baseline_new_view': observation['observation'], 'renders': 6, 'qualification_children': 1,
                'raw_comparisons': 1, 'reused_baseline_raw_observations': 1, 'run_root': str(output), 'elapsed_seconds': time.monotonic() - started,
                'aggregate_accepted': False, 'scope': 'endpoint improvement, old/new silhouettes, raw surface, boundary and specific edit verdicts independent'})
        except Exception as exc:
            publish('results.json', {'protocol': frozen['protocol'], 'status': 'failed',
                'reason': type(exc).__name__ + ': ' + str(exc), 'run_root': str(output),
                'aggregate_accepted': False, 'scope': 'strict guard stop; no fit/render/helper fallback'})
            raise
        finally:
            for path in output.rglob('*'):
                if path.is_file() and path.name not in ('run-ownership.json', 'run-lease.json'):
                    owner.register_file(path.relative_to(output), 'final_output')
        print('MULTIPART_ENDPOINT=' + str(output / 'results.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
