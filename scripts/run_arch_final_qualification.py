#!/usr/bin/env python3
"""Missing five-view passes and one boundary verdict for the exact saved arch."""
from __future__ import annotations
import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'blender_blocking'), str(ROOT)]
import run_triangle_final_qualification as shared

FAMILY = 'concave_arch'
SOURCE = '3efc8d0c7b67f6a229531ed3a7e9e5a51ae82dc03890a54d157a79af33a2d9a1'
CANDIDATE = 'e2b54f3511ccca10d456c3c52ac78ff2deb13915bf36905544e1943294b95e6c'
PROTOCOL = 'retained_arch_final_qualification_v1'
ORIGINAL_PRODUCER = 'bounded_arch_exterior_checkpoint'
REUSE = {'source': {'oblique_145_40': 'source-heldout/oblique_145_40'},
         'candidate': {'oblique_145_40': 'refined-heldout/oblique_145_40'}}
RUNTIME_FILES = tuple(p for p in shared.RUNTIME_FILES
                      if p != 'blender_blocking/test_triangle_final_qualification.py') + (
    'scripts/run_arch_final_qualification.py', 'blender_blocking/test_arch_final_qualification.py',
    'blender_blocking/primitives/polygon_extrusion.py',
)


def require_scope(plan):
    expected = {'protocol': PROTOCOL, 'family': FAMILY, 'views': list(shared.CANONICAL_VIEWS),
        'passes': list(shared.INSPECTION_PASSES), 'resolution': [512, 512], 'native_frames': 28,
        'retained_alpha_passes': 2, 'logical_inspection_passes': 30, 'fits': 0,
        'raw_comparisons': 0, 'semantic_transactions': 0, 'qualifier_children': 1,
        'qualifier_timeout_seconds': 15, 'work_seconds': 85, 'join_seconds': 5, 'threads': 2,
        'committed_limit_bytes': shared.MEMORY, 'rss_limit_bytes': shared.MEMORY,
        'artifact_limit_bytes': 268435456, 'pass_write_reservation_bytes': 8388608,
        'artist_surface_limits': None, 'aggregate_accepted': False,
        'retained_measurement_producer': ORIGINAL_PRODUCER,
        'gates': {'minimum_area_iou': .7, 'minimum_boundary_iou': .8, 'maximum_signed_distance_loss': .05}}
    if any(plan.get(k) != v or (type(v) in (int, bool) and type(plan.get(k)) is not type(v))
           for k, v in expected.items()):
        raise ValueError('arch final frozen scope differs')
    entry = plan['case']
    if entry['candidate_geometry_hash'] != CANDIDATE or entry['source_geometry_hash'] != SOURCE:
        raise ValueError('arch final selected source/body differs')
    if {role: {view: row['label'] for view, row in rows.items()}
            for role, rows in plan['reuse'].items()} != REUSE:
        raise ValueError('exact two retained ob145 alpha slots required')


def require_existing_edit(wire, edited_wire, edit):
    """Reuse the original actual transaction, never claim a fresh pointer edit."""
    nodes = wire.get('root_nodes', [])
    if (wire.get('program_id') != 'frozen-family-concave_arch' or len(nodes) != 1
            or nodes[0].get('primitive_type') != 'polygon_extrusion'
            or nodes[0]['parameters'].get('holes') != []
            or len(nodes[0]['parameters'].get('outer', [])) != 8):
        raise ValueError('retained arch recipe layout differs')
    depth = nodes[0]['parameters'].get('height_world')
    if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not math.isfinite(depth) or depth <= 0:
        raise ValueError('retained physical depth is invalid')
    expected = deepcopy(wire)
    expected['root_nodes'][0]['parameters']['height_world'] = depth * 1.05
    if edited_wire != expected:
        raise ValueError('existing depth edit changed outline, pose or other recipe fields')
    response = edit.get('response', {})
    checks = response.get('checks', {})
    if (edit.get('control') != 'extrusion_depth_world x1.05'
            or response.get('control') != 'extrusion_depth_world'
            or response.get('expected_multiplier') != 1.05
            or type(response.get('expected_multiplier')) is not float
            or response.get('passed') is not True
            or any(checks.get(k) is not True for k in ('fixed_all_outline_vertices_xy',
                        'unchanged_triangle_inventory', 'depth_response'))
            or any(edit.get(k) is not True for k in ('passed', 'geometry_changed',
                        'same_source_pointer', 'original_mesh_restored', 'exact_indexed_restoration'))):
        raise ValueError('body-specific physical depth response/restoration is unavailable')
    return deepcopy(edit)


def verify_checkpoint(plan):
    entry, root = plan['case'], Path(plan['checkpoint_root'])
    manifest, lease = shared.read_json(plan['checkpoint_manifest']), shared.read_json(plan['checkpoint_lease'])
    receipt = shared.read_json(plan['checkpoint_receipt'])
    if (receipt.get('protocol') != 'bounded_arch_exterior_checkpoint_v1'
            or receipt.get('status') != 'measured' or manifest['state'] != 'succeeded'
            or lease['status'] != 'released' or manifest['producer'] != ORIGINAL_PRODUCER
            or manifest['run_root'] != str(root.resolve())
            or manifest['run_id'] != lease['run_id'] or manifest['owner_token'] != lease['owner_token']):
        raise ValueError('retained arch checkpoint ownership/layout differs')
    row = receipt['cases'][FAMILY]
    raw = row['refined_raw_surface']
    wire = shared.read_json(entry['candidate_program'])
    if (row.get('status') != 'measured' or row['refined_geometry_hash'] != CANDIDATE
            or row['program'] != wire or raw.get('candidate_geometry_hash') != CANDIDATE
            or raw.get('reference_geometry_hash') != SOURCE
            or raw.get('protocol') != 'shared_world_area_sample_to_triangle_v1'
            or raw.get('sample_count_per_direction') != 4096 or type(raw.get('sample_count_per_direction')) is not int
            or raw.get('seed') != 61007 or type(raw.get('seed')) is not int
            or row.get('source_original_geometry_hash') != SOURCE or row.get('source_replayed_geometry_hash') != SOURCE
            or row.get('source_exact_oriented_equivalence') is not True
            or row.get('fixed_inputs_geometry_stable') is not True
            or row.get('artist_surface_limits') is not None or row.get('aggregate_accepted') is not False):
        raise ValueError('retained arch body/program/raw/source bindings differ')
    require_existing_edit(wire, shared.read_json(entry['semantic_edited_program']), row['semantic_edit_restoration'])
    artifacts = {item['path']: item for item in manifest['artifacts']}
    for binding in (plan['checkpoint_receipt'], plan['historical_frozen_workload'], plan['crop_request'],
                    entry['candidate_program'], entry['candidate_npz'], entry['candidate_obj'], entry['semantic_edited_program']):
        relative = Path(binding['path']).relative_to(root).as_posix()
        if artifacts[relative]['sha256'] != binding['sha256']:
            raise ValueError('checkpoint original owned artifact bytes differ')
    historical = shared.read_json(plan['historical_frozen_workload'])
    if historical['cases'][0]['prior_view_exposure'] != plan['prior_view_exposure']:
        raise ValueError('original held-out exposure history differs')
    crop = shared.read_json(plan['crop_request'])
    if (crop.get('status') != 'requested' or crop.get('fit_view') != 'oblique_35_28_expanded'
            or crop.get('heldout_views') != ['oblique_145_40']
            or crop.get('prior_view_exposure') != plan['prior_view_exposure']):
        raise ValueError('original genuine crop/held-out history differs')
    return row


def validate_plan(plan):
    require_scope(plan)
    needed = {str((ROOT / p).resolve()) for p in RUNTIME_FILES}
    needed.update(plan[k]['path'] for k in ('reference_workload', 'reference_receipt', 'checkpoint_receipt',
        'checkpoint_manifest', 'checkpoint_lease', 'historical_frozen_workload', 'crop_request'))
    needed.update(plan['case'][k]['path'] for k in ('source_npz', 'candidate_program', 'candidate_npz',
        'candidate_obj', 'semantic_edited_program'))
    needed.update(row['path'] for row in plan['case']['original_masks'].values())
    needed.add(plan['qualification_python'])
    needed.update(row['path'] for row in plan['qualification_package_files'])
    needed.update(row['path'] for row in plan['geometry_package_files'])
    for rows in plan['reuse'].values():
        for entry in rows.values(): needed.update(row['path'] for row in entry['files'].values())
    if not needed.issubset(plan['input_sha256']) or len(plan['input_sha256']) > 80:
        raise ValueError('arch final inputs not all frozen within bounded inventory')
    if sum(Path(p).stat().st_size for p in plan['input_sha256']) > 536870912:
        raise ValueError('frozen input total byte bound exceeded')
    for path, digest in plan['input_sha256'].items(): shared.verify_file({'path': path, 'sha256': digest})
    entry = plan['case']
    authored = {row['name']: row for row in shared.read_json(plan['reference_workload'])['cases']}
    source = shared.read_json(plan['reference_receipt'])['cases'][FAMILY]
    if (entry['source_case'] != authored[FAMILY] or entry['source_geometry_hash'] != source['geometry_hash']
            or entry['original_cameras'] != source['reference_cameras']
            or set(entry['original_cameras']) != set(shared.CANONICAL_VIEWS)):
        raise ValueError('authored arch/original camera declarations differ')
    for view in shared.CANONICAL_VIEWS:
        shared.camera_frame_sha256(entry['original_cameras'][view])
        if entry['original_masks'][view]['sha256'] != source['reference_cameras'][view]['png_sha256']:
            raise ValueError('original arch mask binding differs')
    for key, expected in (('source_npz', SOURCE), ('candidate_npz', CANDIDATE)):
        arrays = shared.load_exact(entry[key]['path'], expected)
        if (len(arrays.vertices), len(arrays.faces)) != (16, 28):
            raise ValueError('retained source/body inventory differs')
    verify_checkpoint(plan)
    for role, rows in plan['reuse'].items():
        for entry in rows.values():
            shared.load_reuse(entry, SOURCE if role == 'source' else CANDIDATE, plan['measurement_settings'],
                              family=FAMILY, producer=ORIGINAL_PRODUCER)


def execute(plan, output, plan_binding):
    import bpy
    from evaluation.canonical_artifacts import canonical_artifact_inventory
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.output_qualification import qualify_retained_output
    from synthetic.quality_references import build_quality_reference
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _compile_exact
    from run_quality_coverage_check import save_mesh
    from run_reference_neutral_inspection import shading_state, style_comparison
    started = time.monotonic()
    if list(bpy.app.version) != [5, 2, 2] or bpy.app.build_hash.decode() != 'd13f752e3b9c':
        raise ValueError('frozen measurement renderer version/build differs')
    owner = OwnedRun(output, producer='retained_arch_final_qualification', max_generated_bytes=268435456,
                    shared_inputs={**plan['input_sha256'], plan_binding['path']: plan_binding['sha256']})
    receipt = {'protocol': PROTOCOL, 'status': 'running', 'cases': {}, 'completed_native_frames': 0,
        'attempted_native_frames': 0, 'retained_alpha_passes_verified': 0, 'fits': 0, 'raw_comparisons': 0,
        'semantic_transactions': 0, 'qualifier_children_invoked': 0, 'aggregate_accepted': False,
        'environment': {'blender': bpy.app.version_string, 'python': sys.version},
        'lifecycle_scope': 'producer joins its one helper; complete tree joins belong to the separate bounded supervisor'}
    with owner:
        def publish(relative, value):
            path = owner.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(shared.encoded_json(value), encoding='utf-8')
            owner.register_file(relative, 'final_output')
        publish('frozen-workload.json', {**plan, 'plan_binding': plan_binding})
        publish('results.json', receipt)
        folder = owner.root / FAMILY
        folder.mkdir()
        row = {'status': 'blocked', 'geometry_hash': CANDIDATE, 'source_geometry_hash': SOURCE,
            'artifact_directory': str(folder / 'candidate'), 'artist_surface_limits': None, 'aggregate_accepted': False}
        receipt['cases'][FAMILY] = row
        try:
            entry = plan['case']
            old = verify_checkpoint(plan)
            original = shared.load_exact(entry['source_npz']['path'], SOURCE)
            retained = shared.load_exact(entry['candidate_npz']['path'], CANDIDATE)
            source = build_quality_reference(entry['source_case']).object
            source_folder, candidate_folder = folder / 'source', folder / 'candidate'
            rebuilt, _ = save_mesh(source, source_folder)
            equivalence = shared.reference_equivalence(original, rebuilt)
            row['source_equivalence'] = equivalence
            publish(Path(FAMILY) / 'source-equivalence.json', equivalence)
            if rebuilt.content_hash != SOURCE or equivalence['equivalent'] is not True:
                raise ValueError('source exact indexed replay differs before measurements')
            compiled, candidate, captured = _compile_exact(shared.read_json(entry['candidate_program']), retained)
            saved, hashes = save_mesh(candidate, candidate_folder)
            row.update(hashes)
            row['candidate_replay'] = {'expected_indexed_hash': CANDIDATE,
                'actual_indexed_hash': saved.content_hash, 'exact': saved.content_hash == CANDIDATE}
            if saved.content_hash != CANDIDATE or captured.content_hash != CANDIDATE:
                raise ValueError('candidate exact indexed replay differs before measurements')
            publish(Path(FAMILY) / 'candidate/program.json', shared.read_json(entry['candidate_program']))
            row.update(program=old['program'], refined_geometry_hash=CANDIDATE,
                refined_raw_surface=old['refined_raw_surface'], raw_receipt_reuse=plan['checkpoint_receipt'],
                editability={'status': 'passed', 'passed': True, 'receipt': plan['checkpoint_receipt'],
                    'geometry_hash': CANDIDATE, 'semantic_edit_restoration': old['semantic_edit_restoration'],
                    'scope': 'original body-specific actual pointer/mesh restoration; no new edit on this process'},
                surface={'status': 'unqualified', 'qualified_limits': None, 'raw': old['refined_raw_surface']},
                raw_regression_history={'baseline': old['baseline_raw_surface_reused'],
                    'refined': old['refined_raw_surface'], 'P95': 'increase retained; no new P95 cutoff'},
                prior_view_exposure=plan['prior_view_exposure'], crop_history=plan['crop_history'])
            styles = {'source': shading_state(source), 'candidate': shading_state(candidate)}
            styles['comparison'] = style_comparison(styles['source'], styles['candidate'])
            publish(Path(FAMILY) / 'shading-style.json', styles)
            source_cameras, source_measurements = shared.render_role(source, source_folder, 'source', plan,
                owner, started + 85, receipt, publish)
            candidate_cameras, candidate_measurements = shared.render_role(candidate, candidate_folder, 'candidate', plan,
                owner, started + 85, receipt, publish)
            if shading_state(source) != styles['source'] or shading_state(candidate) != styles['candidate']:
                raise ValueError('inspection altered native shading intent')
            inventories = {}
            for role, geometry, cameras, peer in (('source', rebuilt, source_cameras, candidate_cameras),
                                                   ('candidate', captured, candidate_cameras, source_cameras)):
                inventories[role] = canonical_artifact_inventory(folder / role, geometry_hash=geometry.content_hash,
                    camera_records=cameras, reference_camera_records=peer,
                    pass_states={name: 'completed' for name in shared.INSPECTION_PASSES})
                publish(Path(FAMILY) / role / 'canonical-inspection.json', inventories[role])
            packet = shared.matched_packet(inventories['source'], inventories['candidate'], source_cameras, candidate_cameras)
            packet['legacy_original_declaration'] = {view: {
                'actual_source_frame_matches_declared': shared.camera_frame_sha256(source_cameras[view]) ==
                    shared.camera_frame_sha256(entry['original_cameras'][view]),
                'original_clipping_status': 'unavailable', 'original_geometry_binding': 'legacy unavailable; no backfill'}
                for view in shared.CANONICAL_VIEWS}
            publish(Path(FAMILY) / 'matched-packet.json', packet)
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            views = {view: compare_controlled_measurements(source_measurements[view], candidate_measurements[view],
                     view=view, config=gates) for view in shared.CANONICAL_VIEWS}
            row.update(canonical_inspection=packet, presentation={'status': 'retained', 'styles': styles,
                'pixel_review': 'pending real output inspection; retained passes do not establish appearance'},
                silhouette={'status': 'passed' if all(v['passed'] for v in views.values()) else 'failed', 'views': views})
            publish('source-declarations.json', {FAMILY: shared.source_declaration(plan, source_folder, source_cameras, equivalence)})
            if receipt['completed_native_frames'] != 28 or receipt['retained_alpha_passes_verified'] != 2:
                raise ValueError('native/reused frame counts differ from the frozen plan')
            publish('results.json', receipt)
            if started + 85 - time.monotonic() < 20:
                raise TimeoutError('insufficient unchanged outer allowance for one15s boundary helper')
            receipt['qualifier_children_invoked'] += 1
            boundary = qualify_retained_output(captured, {'native_qualification_python': plan['qualification_python'],
                'native_qualification_timeout_s': 15., 'native_run_ownership_root': output / 'qualification-children'})
            if boundary.get('geometry_content_hash') != CANDIDATE:
                raise ValueError('boundary helper geometry identity differs')
            row.update(boundary=boundary, status='complete', current_twelve_row_selection='preserved')
            if evaluated_arrays(candidate).content_hash != CANDIDATE or evaluated_arrays(source).content_hash != SOURCE:
                raise ValueError('final qualification altered retained geometry')
            validate_plan(plan)
            passed = row['silhouette']['status'] == 'passed' and boundary.get('single_solid_qualified')
            receipt['status'] = 'required_independent_gates_passed' if passed else 'required_independent_gates_failed'
            if not passed: owner.mark_failed('strict silhouettes/boundary failed or unavailable')
        except Exception as exc:
            row.update(status='blocked', reason=type(exc).__name__ + ': ' + str(exc))
            receipt['status'] = 'blocked'
            owner.mark_failed(row['reason'])
            try:
                shared.encoded_json(receipt)
            except (ValueError, TypeError) as secondary:
                receipt['case_transport_error'] = type(secondary).__name__ + ': ' + str(secondary)
                receipt['cases'] = {FAMILY: {'status': 'blocked', 'geometry_hash': CANDIDATE, 'reason': row['reason']}}
        finally:
            for path in folder.rglob('*'):
                if path.is_file(): owner.register_file(path.relative_to(owner.root), 'final_output')
            receipt['elapsed_seconds'] = time.monotonic() - started
            publish('results.json', receipt)
    print('ARCH_FINAL_RESULT=' + str(owner.root / 'results.json'), flush=True)
    return 0 if receipt['status'] == 'required_independent_gates_passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    binding = {'path': str(args.plan.resolve()), 'sha256': args.plan_sha256}
    plan = shared.read_json(binding)
    validate_plan(plan)
    output = args.output.absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / 'temp/tasks'):
        raise ValueError('fresh output must stay below isolated worktree temp/tasks')
    return execute(plan, output, binding)


if __name__ == '__main__':
    raise SystemExit(main())
