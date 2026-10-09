#!/usr/bin/env python3
"""Two saved multipart controlled-alpha observations; no fitting or raw metrics."""
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
REQUIRED_CANDIDATE_HASH = 'c367746d102dac33f3ad12b271c2ab00d46bd4da59a4bd8004a4bef7152eb0d1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    import bpy
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import camera_frame_sha256
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.reference_noise import oriented_surface_identity
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from integration.blender_ops.measurement_render import (
        controlled_measurement_session, render_controlled_measurement,
    )
    from reconstruction.multipart_family import retained_multipart_program
    from reconstruction.native_geometry import GeometryArrays, evaluated_arrays
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_frozen_multipart_reconstruction import compile_live_multipart
    from run_quality_coverage_check import save_mesh
    from run_surface_quality_check import _replay_orthographic_camera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not args.output.resolve().is_relative_to(ROOT / 'temp/tasks'):
        raise ValueError('owned observation output must stay within isolated temp/tasks')
    preflight = read_json(args.preflight)
    for filename, digest in preflight['input_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('frozen observation input changed: ' + filename)
    proposal = read_json(args.proposal)
    if (proposal['saved_geometry_hash'] != REQUIRED_CANDIDATE_HASH
            or proposal['source_receipt_sha256'] != sha(args.prepared_run)):
        raise ValueError('proposal no longer binds the unchanged saved candidate receipt')
    selection = proposal['proposed_one_complementary_uncropped_view']
    camera_declaration = selection['camera']
    if (selection['azimuth_degrees'] != -15 or selection['elevation_degrees'] != 20
            or camera_declaration['resolution'] != [512, 512]):
        raise ValueError('only the frozen complementary 512 camera is authorized')
    camera_frame_sha256(camera_declaration)
    prior = read_json(args.prepared_run)['cases'][FAMILY]
    prepared = Path(prior['artifact_directory'])
    wire = read_json(prepared / 'program.json')
    if wire != prior['program'] or prior['geometry_hash'] != REQUIRED_CANDIDATE_HASH:
        raise ValueError('saved multipart recipe or indexed identity changed')
    for filename, key in [('evaluated-exact.npz', 'npz_sha256'), ('evaluated.obj', 'obj_sha256')]:
        if sha(prepared / filename) != prior[key]:
            raise ValueError('saved candidate artifact changed: ' + filename)
    with np.load(prepared / 'evaluated-exact.npz', allow_pickle=False) as arrays:
        retained_candidate = GeometryArrays.capture(arrays['vertices'], arrays['faces'])
    if retained_candidate.content_hash != REQUIRED_CANDIDATE_HASH:
        raise ValueError('saved candidate indexed array identity changed')
    source_root = args.reference_root.resolve(strict=True)
    source_case = next(case for case in read_json(source_root / 'frozen-workload.json')['cases']
                       if case['name'] == FAMILY)
    source_prior = read_json(source_root / 'results.json')['cases'][FAMILY]
    with np.load(source_root / FAMILY / 'evaluated-exact.npz', allow_pickle=False) as arrays:
        retained_source = GeometryArrays.capture(arrays['vertices'], arrays['faces'])
    if retained_source.content_hash != source_prior['geometry_hash']:
        raise ValueError('retained authored source indexed identity changed')
    # Authored dimensions below enter the source observation builder only.
    # There is no fitter in this script and the saved candidate remains unchanged.
    frozen = {
        'protocol': 'multipart_complementary_controlled_observation_v1',
        'family': FAMILY, 'view': VIEW, 'camera_declaration': camera_declaration,
        'proposal': proposal, 'proposal_sha256': sha(args.proposal),
        'preflight_sha256': sha(args.preflight), 'input_sha256': preflight['input_sha256'],
        'source_case': source_case, 'source_original_geometry_hash': retained_source.content_hash,
        'candidate_geometry_hash': REQUIRED_CANDIDATE_HASH, 'candidate_program': wire,
        'source_geometry_guard': 'exact oriented triangles and vertex multiset; indexed source order retained separately',
        'candidate_geometry_guard': 'exact original evaluated indexed identity before and after both frames',
        'acquisition': 'opaque_linear_alpha_v1; float32 EXR alpha; identical actual camera/renderer/sampling contracts',
        'resolution': [512, 512], 'samples': 64, 'filter_size': 1.5,
        'renders': 2, 'fit_calls': 0, 'raw_metric_calls': 0, 'qualification_children': 0,
        'heldout_145': 'untouched; no old masks decoded or fitting performed',
        'budget_seconds': 60, 'budget_rss_bytes': 8 * 1024 ** 3, 'threads': 2,
        'independent_surface_limits': None,
    }
    owner = OwnedRun(args.output.resolve(), producer='multipart_complementary_controlled_observation',
                     max_generated_bytes=67108864, shared_inputs={'preflight': str(args.preflight)})
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
            reference = build_quality_reference(source_case).object
            source_arrays = evaluated_arrays(reference)
            source_equivalent = oriented_surface_identity(source_arrays) == oriented_surface_identity(retained_source)
            if not source_equivalent:
                save_mesh(reference, output / 'source-guard-failed')
                raise ValueError('authored observation source changed exact oriented geometry')
            program = retained_multipart_program(wire)
            candidate, parts = compile_live_multipart(program)
            candidate_arrays = evaluated_arrays(candidate)
            if candidate_arrays.content_hash != REQUIRED_CANDIDATE_HASH:
                save_mesh(candidate, output / 'candidate-guard-failed')
                raise ValueError('saved Boolean replay changed exact indexed geometry before frames')
            for label, obj in [('source', reference), ('candidate', candidate)]:
                save_mesh(obj, output / label)
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.data.type = 'ORTHO'
            bpy.context.scene.camera = camera
            records = {}
            for label, obj in [('source', reference), ('candidate', candidate)]:
                if time.monotonic() - started > 60:
                    raise TimeoutError('fixed two-frame observation deadline')
                with controlled_measurement_session(target_objects=[obj], camera=camera,
                        resolution=(512, 512), samples=64, filter_size=1.5) as session:
                    # Same ORIGINAL declaration for each pass, never the native snapshot.
                    _replay_orthographic_camera(camera, camera_declaration)
                    record = render_controlled_measurement(session, output / label / (VIEW + '.exr'))
                expected = source_arrays.content_hash if label == 'source' else REQUIRED_CANDIDATE_HASH
                if record['geometry_hashes'] != [expected] or not record['geometry_unchanged']:
                    raise ValueError(label + ' measurement lost exact geometry binding')
                np.save(output / label / 'coverage.npy', record['coverage'], allow_pickle=False)
                np.save(output / label / 'mask.npy', record['mask'], allow_pickle=False)
                # Read-only occupancy visualization derived from retained linear alpha.
                Image.fromarray(np.rint(record['coverage'] * 255).astype(np.uint8)).save(output / label / 'coverage.png')
                publish(label + '/measurement.json', {k: v for k, v in record.items() if k not in ('coverage', 'mask')})
                records[label] = record
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            observation = compare_controlled_measurements(records['source'], records['candidate'], view=VIEW, config=gates)
            unchanged = evaluated_arrays(candidate).content_hash == REQUIRED_CANDIDATE_HASH
            if not unchanged or evaluated_arrays(reference).content_hash != source_arrays.content_hash:
                raise ValueError('controlled observation changed source/candidate geometry')
            borders = {label: bool(row['mask'][0].any() or row['mask'][-1].any()
                       or row['mask'][:, 0].any() or row['mask'][:, -1].any()) for label, row in records.items()}
            publish('program.json', wire)
            publish('results.json', {
                'protocol': frozen['protocol'], 'status': 'measured', 'run_root': str(output),
                'view': VIEW, 'observation': observation, 'rendered_frames': 2,
                'camera_declaration': camera_declaration,
                'actual_camera': records['source']['contract']['camera'],
                'matched_acquisition_sha256': records['source']['contract_sha256'],
                'geometry': {'source_original_indexed_hash': retained_source.content_hash,
                    'source_replayed_indexed_hash': source_arrays.content_hash,
                    'source_exact_oriented_equivalence': source_equivalent,
                    'candidate_before_and_after_hash': REQUIRED_CANDIDATE_HASH,
                    'candidate_exact_indexed_identity': unchanged},
                'frame_border_occupied': borders, 'endpoint_uncertainty': prior.get('fit_uncertainty'),
                'independent_surface_limits': None, 'aggregate_accepted': False,
                'scope': 'new controlled observation only; frozen old display protocol and heldout145 verdict unchanged',
                'elapsed_seconds': time.monotonic() - started,
            })
        except Exception as exc:
            publish('results.json', {'protocol': frozen['protocol'], 'status': 'failed',
                'reason': type(exc).__name__ + ': ' + str(exc), 'aggregate_accepted': False,
                'run_root': str(output), 'scope': 'stop at strict guard; no repair/fitting fallback'})
            raise
        finally:
            for path in output.rglob('*'):
                if path.is_file() and path.name not in ('run-ownership.json', 'run-lease.json'):
                    owner.register_file(path.relative_to(output), 'final_output')
        print('MULTIPART_COMPLEMENTARY=' + str(output / 'results.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
