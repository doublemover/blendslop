#!/usr/bin/env python3
"""Missing exact five-view and boundary gates for five saved positive bodies.

No fitting, raw resampling, semantic transaction or row promotion runs here.
Original alpha reuse retains its actual succeeded/released owner and bytes.
"""
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

PROTOCOL = 'retained_family_final_qualification_v1'
ORIGINAL_PRODUCER = 'bounded_adaptive_family_checkpoint'
TARGETS = {
    'capsule': {
        'source': '85925adc8fa76ed6f0ed29eca90fc8cd2bd86680171a4c419a6e6e20f7f4938e',
        'candidate': 'd436a5924e7f8e7fca6dab582462a0794017721f3ca51a488403aee1cb2c2b6b',
        'owner': 'adaptive-family-detail-02/owned-ipr1val3', 'primitive': 'capsule', 'metadata_family': 'capsule',
        'source_inventory': (1154, 2304), 'candidate_inventory': (4610, 9216),
        'fit_view': 'front', 'control': 'segment_height_world', 'multiplier': 1.05,
        'response_checks': ('extents_response', 'axial_segment_response'), 'reuse': False},
    'torus': {
        'source': '3d1743e165e096fc7e413237523e2402a809a7e71c6209f95d1de9f705b1738a',
        'candidate': '3afacdc32a140a1099cd304b101d8910a7e6e58701b3608f947039aaf2349417',
        'owner': 'adaptive-family-detail-02/owned-ipr1val3', 'primitive': 'torus', 'metadata_family': None,
        'source_inventory': (3072, 6144), 'candidate_inventory': (2304, 4608),
        'fit_view': 'top', 'control': 'minor_radius', 'multiplier': 1.05,
        'response_checks': ('tube_radius_x1_05', 'major_radius_fixed', 'axial_tube_height_x1_05',
                            'outer_ring_expands', 'hole_narrows'), 'reuse': True},
    'tapered_frustum': {
        'source': 'a05605528bbb310beef7f6a137ecd89521055cc1facaedc1383af8f42f5cfa1d',
        'candidate': '41cc2701f38d6328b8e72c8aa89c8169c2264eadd2c8b2c4159e3cb10226ac45',
        'owner': 'adaptive-family-detail-03/owned-_r4c2moh', 'primitive': 'frustum', 'metadata_family': 'frustum',
        'source_inventory': (192, 380), 'candidate_inventory': (192, 380),
        'fit_view': 'front', 'control': 'radius_top', 'multiplier': 1.1,
        'response_checks': ('extents_response', 'top_radius_response', 'bottom_radius_fixed'), 'reuse': True},
    'cylinder': {
        'source': 'c68d770aa57f8185001ca8211b2342bbe29646a0ed78e548755b338f96f8c38a',
        'candidate': 'ab6343943cb9a35006f1b1659d9b6cbad50520fff000ab9ca781fecf25c5ff88',
        'owner': 'adaptive-family-detail-04/owned-7hmpyfeo', 'primitive': 'cylinder', 'metadata_family': 'cylinder',
        'source_inventory': (192, 380), 'candidate_inventory': (192, 380),
        'fit_view': 'front', 'control': 'radius_world', 'multiplier': 1.05,
        'response_checks': ('extents_response',), 'reuse': True},
    'thin_plate': {
        'source': '6f0af9263586cd421c6c539c660f5d8b09c90efdf1b9bd810488fffeb24f7f50',
        'candidate': '0b2cb7665eeb1a2680383a39c79f526ceee5359f6b8b55fd54aacf4b59aea7a0',
        'owner': 'adaptive-family-detail-05/owned-hkggish9', 'primitive': 'box', 'metadata_family': 'box',
        'source_inventory': (8, 12), 'candidate_inventory': (8, 12),
        'fit_view': 'oblique_35_28', 'control': 'depth_world', 'multiplier': 1.05,
        'response_checks': ('extents_response',), 'reuse': True},
}
RUNTIME_FILES = tuple(p for p in shared.RUNTIME_FILES
                      if p != 'blender_blocking/test_triangle_final_qualification.py') + (
    'scripts/run_retained_family_final_qualification.py',
    'blender_blocking/test_retained_family_final_qualification.py',
)
MEASUREMENT_SETTINGS = {
    'protocol': 'opaque_linear_alpha_v1', 'hard_mask_threshold': .5,
    'renderer': {'engine': 'BLENDER_EEVEE', 'blender_version': [5, 2, 2], 'build_hash': 'd13f752e3b9c'},
    'sampling': {'taa_render_samples': 64, 'filter_size': 1.5, 'motion_blur': False, 'depth_of_field': False},
    'pixel_convention': 'top-left array; pixel (i,j) center=(j+.5,i+.5)',
    'surface_policy': 'opaque target meshes only; forced emission; no compositor/sequencer/world volume',
    'encoding': 'float32 OpenEXR linear alpha; display RGB ignored',
}
REPLAY_CLIPS = {'clip_start': .10000000149011612, 'clip_end': 1000.}
STYLE_POLICY = 'authored source and exact saved candidate recipe; no smoothing/style edits; normal material restored'


def target(family):
    if family not in TARGETS:
        raise ValueError('unsupported retained positive family')
    return TARGETS[family]


def reuse_slots(family):
    spec = target(family)
    if not spec['reuse']:
        return {'source': {}, 'candidate': {}}
    return {'source': {spec['fit_view']: 'source-fit', 'oblique_145_40': 'source-heldout/oblique_145_40'},
            'candidate': {'oblique_145_40': 'refined-heldout/oblique_145_40'}}


def scope(family):
    spec = target(family)
    reused = 3 if spec['reuse'] else 0
    return {'protocol': PROTOCOL, 'family': family, 'views': list(shared.CANONICAL_VIEWS),
        'passes': list(shared.INSPECTION_PASSES), 'resolution': [512, 512], 'native_frames': 30-reused,
        'retained_alpha_passes': reused, 'logical_inspection_passes': 30, 'fits': 0,
        'raw_comparisons': 0, 'semantic_transactions': 0, 'qualifier_children': 1,
        'qualifier_timeout_seconds': 15, 'work_seconds': 85, 'join_seconds': 5, 'threads': 2,
        'committed_limit_bytes': shared.MEMORY, 'rss_limit_bytes': shared.MEMORY,
        'artifact_limit_bytes': 268435456, 'pass_write_reservation_bytes': 8388608,
        'artist_surface_limits': None, 'aggregate_accepted': False,
        'retained_measurement_producer': ORIGINAL_PRODUCER,
        'measurement_settings': deepcopy(MEASUREMENT_SETTINGS), 'replay_clips': deepcopy(REPLAY_CLIPS),
        'style_policy': STYLE_POLICY, 'current_twelve_row_selection': 'preserved',
        'gates': {'minimum_area_iou': .7, 'minimum_boundary_iou': .8, 'maximum_signed_distance_loss': .05}}


def require_scope(plan):
    family = plan['family']
    spec = target(family)
    if any(plan.get(k) != v or (type(v) in (int, bool) and type(plan.get(k)) is not type(v))
           for k, v in scope(family).items()):
        raise ValueError('retained family frozen scope differs')
    entry = plan['case']
    if entry['candidate_geometry_hash'] != spec['candidate'] or entry['source_geometry_hash'] != spec['source']:
        raise ValueError('selected retained family source/body differs')
    if {role: {view: row['label'] for view, row in rows.items()}
            for role, rows in plan['reuse'].items()} != reuse_slots(family):
        raise ValueError('original family alpha slots differ')


def require_existing_edit(family, wire, edited_wire, edit):
    """Bind the real original recipe edit, including coupled dimensions and style."""
    spec = target(family)
    nodes = wire.get('root_nodes', [])
    if (wire.get('program_id') != 'frozen-family-'+family or len(nodes) != 1
            or nodes[0].get('primitive_type') != spec['primitive']
            or nodes[0].get('operation') != 'add' or nodes[0].get('children') != []
            or wire.get('constraints') != [] or wire.get('residual_patches') != []
            or wire.get('metadata', {}).get('family') != spec['metadata_family']):
        raise ValueError('retained family recipe layout differs')
    parameters = nodes[0]['parameters']
    control = spec['control']
    key = 'radius_bottom' if family == 'cylinder' else control
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('retained physical control is invalid')
    expected = deepcopy(wire)
    changed = expected['root_nodes'][0]['parameters']
    changed[key] = float(value) * spec['multiplier']
    if family == 'capsule':
        changed['height_world'] = 2*parameters['radius_world'] + changed['segment_height_world']
    elif family == 'tapered_frustum':
        changed['width_world'] = changed['depth_world'] = 2*max(changed['radius_bottom'], changed['radius_top'])
    elif family == 'cylinder':
        if (parameters['radius_top'] != value or parameters['width_world'] != 2*value
                or parameters['depth_world'] != 2*value):
            raise ValueError('retained cylinder cap/dimension aliases differ')
        changed['radius_top'] = changed['radius_bottom']
        changed['width_world'] = changed['depth_world'] = 2*changed['radius_bottom']
    if edited_wire != expected:
        raise ValueError('existing family edit changed pose, style, metadata or other recipe controls')
    response = edit.get('response', {})
    if (edit.get('control') != control+' x'+str(spec['multiplier'])
            or response.get('passed') is not True
            or any(response.get('checks', {}).get(k) is not True for k in spec['response_checks'])
            or any(edit.get(k) is not True for k in ('passed', 'geometry_changed', 'same_source_pointer',
                        'original_mesh_restored', 'exact_indexed_restoration'))):
        raise ValueError('body-specific physical response/restoration is unavailable')
    return deepcopy(edit)


def require_owned_binding(root, artifacts, binding):
    path = shared.verify_file(binding)
    relative = path.relative_to(root).as_posix()
    if artifacts[relative]['sha256'] != binding['sha256']:
        raise ValueError('checkpoint original owned artifact bytes differ')


def verify_checkpoint(plan):
    family = plan['family']
    spec = target(family)
    entry, root = plan['case'], Path(plan['checkpoint_root'])
    manifest, lease = shared.read_json(plan['checkpoint_manifest']), shared.read_json(plan['checkpoint_lease'])
    receipt = shared.read_json(plan['checkpoint_receipt'])
    if (receipt.get('protocol') != 'bounded_adaptive_family_checkpoint_v1'
            or receipt.get('status') != 'measured' or manifest['state'] != 'succeeded'
            or lease['status'] != 'released' or manifest['producer'] != ORIGINAL_PRODUCER
            or manifest['run_root'] != str(root.resolve())
            or manifest['run_id'] != lease['run_id'] or manifest['owner_token'] != lease['owner_token']
            or Path(plan['checkpoint_manifest']['path']) != root/'run-ownership.json'
            or Path(plan['checkpoint_lease']['path']) != root/'run-lease.json'):
        raise ValueError('retained family checkpoint ownership/layout differs')
    row = receipt['cases'][family]
    raw = row['refined_raw_surface']
    wire = shared.read_json(entry['candidate_program'])
    if (row.get('status') != 'measured' or row['refined_geometry_hash'] != spec['candidate']
            or row['program'] != wire or raw.get('candidate_geometry_hash') != spec['candidate']
            or raw.get('reference_geometry_hash') != spec['source']
            or raw.get('protocol') != 'shared_world_area_sample_to_triangle_v1'
            or type(raw.get('sample_count_per_direction')) is not int or raw['sample_count_per_direction'] != 4096
            or type(raw.get('seed')) is not int or raw['seed'] != 61007
            or row.get('source_original_geometry_hash') != spec['source']
            or row.get('source_replayed_geometry_hash') != spec['source']
            or row.get('source_exact_oriented_equivalence') is not True
            or row.get('fixed_inputs_geometry_stable') is not True or row.get('bounded_checkpoint_improved') is not True
            or row.get('artist_surface_limits') is not None or row.get('aggregate_accepted') is not False
            or row.get('selected_current_row_unchanged') is not True):
        raise ValueError('retained family body/program/raw/source bindings differ')
    require_existing_edit(family, wire, shared.read_json(entry['semantic_edited_program']), row['semantic_edit_restoration'])
    artifacts = {item['path']: item for item in manifest['artifacts']}
    for binding in (plan['checkpoint_receipt'], plan['historical_frozen_workload'], entry['candidate_program'],
                    entry['candidate_npz'], entry['candidate_obj'], entry['semantic_edited_program']):
        require_owned_binding(root, artifacts, binding)
    if (Path(plan['checkpoint_receipt']['path']) != root/'results.json'
            or Path(plan['historical_frozen_workload']['path']) != root/'frozen-workload.json'
            or any(Path(entry[k]['path']) != root/family/'refined'/name for k, name in
                (('candidate_program', 'program.json'), ('candidate_npz', 'evaluated-exact.npz'), ('candidate_obj', 'evaluated.obj')))
            or Path(entry['semantic_edited_program']['path']) != root/family/'semantic-edited-program.json'):
        raise ValueError('original checkpoint artifact paths differ')
    historical = shared.read_json(plan['historical_frozen_workload'])
    old_case = next(c for c in historical['cases'] if c['family'] == family)
    if (old_case['prior_view_exposure'] != plan['prior_view_exposure']
            or old_case['cameras'] != entry['original_cameras'] or old_case['source_case'] != entry['source_case']
            or old_case['source_archive'] != entry['source_npz']['path']
            or old_case['source_geometry_hash'] != spec['source']
            or old_case['fit_view'] != spec['fit_view'] or old_case['semantic_control'] != spec['control']
            or old_case['semantic_multiplier'] != spec['multiplier']):
        raise ValueError('original family source/camera/control/exposure history differs')
    for rows in plan['reuse'].values():
        for reuse in rows.values():
            if reuse['owner_root'] != str(root) or reuse['manifest'] != plan['checkpoint_manifest'] or reuse['lease'] != plan['checkpoint_lease']:
                raise ValueError('reused alpha must retain this original succeeded family owner')
    return row


def input_bindings(plan):
    bindings = [plan[k] for k in ('reference_workload', 'reference_receipt', 'checkpoint_manifest',
        'checkpoint_lease', 'checkpoint_receipt', 'historical_frozen_workload')]
    bindings += [plan['case'][k] for k in ('source_npz', 'candidate_program', 'candidate_npz', 'candidate_obj', 'semantic_edited_program')]
    bindings += list(plan['case']['original_masks'].values())
    bindings += plan['qualification_package_files'] + plan['geometry_package_files']
    for rows in plan['reuse'].values():
        for entry in rows.values():
            bindings += [entry['manifest'], entry['lease'], *entry['files'].values()]
    return bindings


def validate_plan(plan):
    require_scope(plan)
    needed = {str((ROOT / p).resolve()) for p in RUNTIME_FILES}
    needed.update(plan[k]['path'] for k in ('reference_workload', 'reference_receipt', 'checkpoint_receipt',
        'checkpoint_manifest', 'checkpoint_lease', 'historical_frozen_workload'))
    needed.update(plan['case'][k]['path'] for k in ('source_npz', 'candidate_program', 'candidate_npz',
        'candidate_obj', 'semantic_edited_program'))
    needed.update(row['path'] for row in plan['case']['original_masks'].values())
    needed.add(plan['qualification_python'])
    needed.update(row['path'] for row in plan['qualification_package_files'])
    needed.update(row['path'] for row in plan['geometry_package_files'])
    for rows in plan['reuse'].values():
        for entry in rows.values():
            needed.update(row['path'] for row in entry['files'].values())
    if any(plan['input_sha256'].get(b['path']) != b['sha256'] for b in input_bindings(plan)):
        raise ValueError('declared binding differs from frozen input digest')
    if not needed.issubset(plan['input_sha256']) or len(plan['input_sha256']) > 88:
        raise ValueError('retained family inputs not all frozen within bounded inventory')
    if sum(Path(p).stat().st_size for p in plan['input_sha256']) > 536870912:
        raise ValueError('frozen input total byte bound exceeded')
    for path, digest in plan['input_sha256'].items():
        shared.verify_file({'path': path, 'sha256': digest})
    family, entry = plan['family'], plan['case']
    spec = target(family)
    reference = shared.read_json(plan['reference_workload'])
    receipt = shared.read_json(plan['reference_receipt'])
    authored = reference['case'] if family == 'capsule' else next(c for c in reference['cases'] if c['name'] == family)
    source = receipt if family == 'capsule' else receipt['cases'][family]
    if (entry['source_case'] != authored or entry['source_geometry_hash'] != source['geometry_hash']
            or entry['original_cameras'] != source['reference_cameras']
            or set(entry['original_cameras']) != set(shared.CANONICAL_VIEWS)
            or set(entry['original_masks']) != set(shared.CANONICAL_VIEWS)):
        raise ValueError('authored family/original camera declarations differ')
    for view in shared.CANONICAL_VIEWS:
        shared.camera_frame_sha256(entry['original_cameras'][view])
        if entry['original_masks'][view]['sha256'] != source['reference_cameras'][view]['png_sha256']:
            raise ValueError('original family mask binding differs')
    for key, hash_key, inventory in (('source_npz', 'source', 'source_inventory'),
                                    ('candidate_npz', 'candidate', 'candidate_inventory')):
        arrays = shared.load_exact(entry[key]['path'], spec[hash_key])
        if (len(arrays.vertices), len(arrays.faces)) != spec[inventory]:
            raise ValueError('retained source/body inventory differs')
    verify_checkpoint(plan)
    for role, rows in plan['reuse'].items():
        for entry in rows.values():
            shared.load_reuse(entry, spec['source' if role == 'source' else 'candidate'], plan['measurement_settings'],
                              family=family, producer=ORIGINAL_PRODUCER)


def prepare_plan(family):
    """Read-only local inventory; caller freezes/approves/writes the returned plan."""
    spec = target(family)
    root = ROOT/'temp/tasks/quality-continuation-20261008'/spec['owner']
    manifest, lease = shared.bind(root/'run-ownership.json'), shared.bind(root/'run-lease.json')
    historical = shared.bind(root/'frozen-workload.json')
    old = next(c for c in shared.read_json(historical)['cases'] if c['family'] == family)
    source_folder = Path(old['source_archive']).parent
    reference_root = source_folder if family == 'capsule' else source_folder.parent
    reference_workload, reference_receipt = shared.bind(reference_root/'frozen-workload.json'), shared.bind(reference_root/'results.json')
    packages = shared.read_json(shared.bind(ROOT/'docs/quality-arch-final-qualification-continuation-20261009/native-plan.json'))
    plan = {**scope(family), 'case': {'source_geometry_hash': spec['source'], 'candidate_geometry_hash': spec['candidate'],
        'source_case': deepcopy(old['source_case']), 'source_npz': shared.bind(old['source_archive']),
        'candidate_program': shared.bind(root/family/'refined/program.json'),
        'candidate_npz': shared.bind(root/family/'refined/evaluated-exact.npz'),
        'candidate_obj': shared.bind(root/family/'refined/evaluated.obj'),
        'semantic_edited_program': shared.bind(root/family/'semantic-edited-program.json'),
        'original_cameras': deepcopy(old['cameras']),
        'original_masks': {view: shared.bind(source_folder/(view+'-mask.png')) for view in shared.CANONICAL_VIEWS}},
        'reference_workload': reference_workload, 'reference_receipt': reference_receipt,
        'checkpoint_root': str(root), 'checkpoint_manifest': manifest, 'checkpoint_lease': lease,
        'checkpoint_receipt': shared.bind(root/'results.json'), 'historical_frozen_workload': historical,
        'qualification_python': packages['qualification_python'],
        'qualification_package_files': deepcopy(packages['qualification_package_files']),
        'geometry_package_files': deepcopy(packages['geometry_package_files']),
        'prior_view_exposure': deepcopy(old['prior_view_exposure']), 'reuse': {},
        'source_scope': 'current exact recapture; historical missing frame/clip/geometry bindings are not backfilled',
        'input_sha256': {}}
    for role, slots in reuse_slots(family).items():
        plan['reuse'][role] = {view: {'owner_root': str(root), 'manifest': manifest, 'lease': lease,
            'label': label, 'files': {name: shared.bind(root/family/label/name)
                for name in ('measurement.json', 'coverage.npy', 'alpha.exr')}} for view, label in slots.items()}
    bindings = input_bindings(plan) + [shared.bind(plan['qualification_python'])]
    bindings += [shared.bind(ROOT/p) for p in RUNTIME_FILES]
    plan['input_sha256'] = {b['path']: b['sha256'] for b in bindings}
    validate_plan(plan)
    return plan


def execute(plan, output, plan_binding):
    import bpy
    family = plan['family']
    spec = target(family)
    source_hash, candidate_hash = spec['source'], spec['candidate']
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
    owner = OwnedRun(output, producer='retained_family_final_qualification', max_generated_bytes=268435456,
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
        folder = owner.root / family
        folder.mkdir()
        row = {'status': 'blocked', 'geometry_hash': candidate_hash, 'source_geometry_hash': source_hash,
            'artifact_directory': str(folder / 'candidate'), 'artist_surface_limits': None, 'aggregate_accepted': False}
        receipt['cases'][family] = row
        try:
            entry = plan['case']
            old = verify_checkpoint(plan)
            original = shared.load_exact(entry['source_npz']['path'], source_hash)
            retained = shared.load_exact(entry['candidate_npz']['path'], candidate_hash)
            source = build_quality_reference(entry['source_case']).object
            source_folder, candidate_folder = folder / 'source', folder / 'candidate'
            rebuilt, _ = save_mesh(source, source_folder)
            equivalence = shared.reference_equivalence(original, rebuilt)
            row['source_equivalence'] = equivalence
            publish(Path(family) / 'source-equivalence.json', equivalence)
            if rebuilt.content_hash != source_hash or equivalence['equivalent'] is not True:
                raise ValueError('source exact indexed replay differs before measurements')
            compiled, candidate, captured = _compile_exact(shared.read_json(entry['candidate_program']), retained)
            saved, hashes = save_mesh(candidate, candidate_folder)
            row.update(hashes)
            row['candidate_replay'] = {'expected_indexed_hash': candidate_hash,
                'actual_indexed_hash': saved.content_hash, 'exact': saved.content_hash == candidate_hash}
            if saved.content_hash != candidate_hash or captured.content_hash != candidate_hash:
                raise ValueError('candidate exact indexed replay differs before measurements')
            publish(Path(family) / 'candidate/program.json', shared.read_json(entry['candidate_program']))
            row.update(program=old['program'], refined_geometry_hash=candidate_hash,
                refined_raw_surface=old['refined_raw_surface'], raw_receipt_reuse=plan['checkpoint_receipt'],
                editability={'status': 'passed', 'passed': True, 'receipt': plan['checkpoint_receipt'],
                    'geometry_hash': candidate_hash, 'semantic_edit_restoration': old['semantic_edit_restoration'],
                    'scope': 'original body-specific actual pointer/mesh restoration; no new edit on this process'},
                surface={'status': 'unqualified', 'qualified_limits': None, 'raw': old['refined_raw_surface']},
                raw_regression_history={'baseline': old['baseline_raw_surface_reused'],
                    'refined': old['refined_raw_surface'], 'P95': 'original observation retained; no new P95 cutoff'},
                prior_view_exposure=plan['prior_view_exposure'])
            styles = {'source': shading_state(source), 'candidate': shading_state(candidate)}
            styles['comparison'] = style_comparison(styles['source'], styles['candidate'])
            publish(Path(family) / 'shading-style.json', styles)
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
                publish(Path(family) / role / 'canonical-inspection.json', inventories[role])
            packet = shared.matched_packet(inventories['source'], inventories['candidate'], source_cameras, candidate_cameras)
            packet['legacy_original_declaration'] = {view: {
                'actual_source_frame_matches_declared': shared.camera_frame_sha256(source_cameras[view]) ==
                    shared.camera_frame_sha256(entry['original_cameras'][view]),
                'original_clipping_status': 'unavailable', 'original_geometry_binding': 'legacy unavailable; no backfill'}
                for view in shared.CANONICAL_VIEWS}
            publish(Path(family) / 'matched-packet.json', packet)
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            views = {view: compare_controlled_measurements(source_measurements[view], candidate_measurements[view],
                     view=view, config=gates) for view in shared.CANONICAL_VIEWS}
            row.update(canonical_inspection=packet, presentation={'status': 'retained', 'styles': styles,
                'pixel_review': 'pending real output inspection; retained passes do not establish appearance'},
                silhouette={'status': 'passed' if all(v['passed'] for v in views.values()) else 'failed', 'views': views})
            publish('source-declarations.json', {family: shared.source_declaration(plan, source_folder, source_cameras, equivalence)})
            if receipt['completed_native_frames'] != plan['native_frames'] or receipt['retained_alpha_passes_verified'] != plan['retained_alpha_passes']:
                raise ValueError('native/reused frame counts differ from the frozen plan')
            publish('results.json', receipt)
            if started + 85 - time.monotonic() < 20:
                raise TimeoutError('insufficient unchanged outer allowance for one15s boundary helper')
            receipt['qualifier_children_invoked'] += 1
            boundary = qualify_retained_output(captured, {'native_qualification_python': plan['qualification_python'],
                'native_qualification_timeout_s': 15., 'native_run_ownership_root': output / 'qualification-children'})
            if boundary.get('geometry_content_hash') != candidate_hash:
                raise ValueError('boundary helper geometry identity differs')
            row.update(boundary=boundary, status='complete', current_twelve_row_selection='preserved')
            if evaluated_arrays(candidate).content_hash != candidate_hash or evaluated_arrays(source).content_hash != source_hash:
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
                receipt['cases'] = {family: {'status': 'blocked', 'geometry_hash': candidate_hash, 'reason': row['reason']}}
        finally:
            for path in folder.rglob('*'):
                if path.is_file(): owner.register_file(path.relative_to(owner.root), 'final_output')
            receipt['elapsed_seconds'] = time.monotonic() - started
            publish('results.json', receipt)
    print('RETAINED_FAMILY_FINAL_RESULT=' + str(owner.root / 'results.json'), flush=True)
    return 0 if receipt['status'] == 'required_independent_gates_passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-family', choices=tuple(TARGETS))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    if args.prepare_family:
        if any((args.plan, args.plan_sha256, args.output)):
            parser.error('read-only inventory and native acquisition arguments are mutually exclusive')
        print(shared.encoded_json(prepare_plan(args.prepare_family)), end='')
        return 0
    if not all((args.plan, args.plan_sha256, args.output)):
        parser.error('native acquisition requires --plan, --plan-sha256 and --output')
    binding = {'path': str(args.plan.resolve()), 'sha256': args.plan_sha256}
    plan = shared.read_json(binding)
    validate_plan(plan)
    output = args.output.absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT/'temp/tasks'):
        raise ValueError('fresh output must stay below isolated worktree temp/tasks')
    return execute(plan, output, binding)


if __name__ == '__main__':
    raise SystemExit(main())
