#!/usr/bin/env python3
"""Exact retained triangle: missing inspection passes and boundary only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'blender_blocking'), str(ROOT / 'scripts'), str(ROOT)]
import test_runner
from evaluation.canonical_artifacts import CANONICAL_VIEWS, INSPECTION_PASSES, camera_frame_sha256
from evaluation.controlled_measurement import coverage_and_mask, measurement_signature
from run_selected_canonical_inspection import (
    load_exact, matched_packet, reference_equivalence, require_same_frame, _normal_material, _settings,
)

FAMILY = 'rounded_triangle_dot'
CANDIDATE = 'bfe727380abc81d0215abec5687b77b83c251133605149a6f7ad2f714423a89d'
SOURCE = '2681272b491af3370176a3751e408ef36496ba592d230bce7c733b8c41cadc64'
PROTOCOL = 'retained_triangle_final_qualification_v1'
MEMORY = 8 * 1024 ** 3
REUSE = {'source': {'top': 'source-fit', 'oblique_145_40': 'source-heldout/oblique_145_40'},
         'candidate': {'oblique_145_40': 'refined-heldout/oblique_145_40'}}
RUNTIME_FILES = (
    'scripts/run_triangle_final_qualification.py', 'blender_blocking/test_triangle_final_qualification.py',
    'scripts/run_selected_canonical_inspection.py', 'scripts/run_reference_neutral_inspection.py',
    'scripts/run_family_source_edit_check.py', 'scripts/run_surface_quality_check.py',
    'scripts/run_quality_coverage_check.py', 'scripts/qualify_native_solid.py',
    'scripts/run_bounded_owned_command.py', 'blender_blocking/test_runner.py',
    'blender_blocking/primitives/shape_program.py', 'blender_blocking/primitives/shape_program_compiler.py',
    'blender_blocking/primitives/rounded_triangle.py', 'blender_blocking/primitives/primitive_protocol.py',
    'blender_blocking/reconstruction/frozen_family.py', 'blender_blocking/reconstruction/native_geometry.py',
    'blender_blocking/reconstruction/output_targets.py', 'blender_blocking/reconstruction/mesh_io.py',
    'blender_blocking/reconstruction/native_qualification.py', 'blender_blocking/reconstruction/output_qualification.py',
    'blender_blocking/reconstruction/grouped_solids.py', 'blender_blocking/metrics/topology_receipt.py',
    'blender_blocking/metrics/topology.py', 'blender_blocking/evaluation/canonical_artifacts.py',
    'blender_blocking/evaluation/reference_noise.py', 'blender_blocking/evaluation/controlled_measurement.py',
    'blender_blocking/evaluation/silhouette_eval.py', 'blender_blocking/evaluation/triangle_contacts.py',
    'blender_blocking/integration/blender_ops/measurement_render.py',
    'blender_blocking/integration/blender_ops/silhouette_render.py',
    'blender_blocking/synthetic/quality_references.py', 'blender_blocking/synthetic/quality_contracts.py',
    'blender_blocking/utils/run_ownership.py', 'blender_blocking/utils/owned_process_supervisor.py',
    'blender_blocking/utils/artifact_publication.py', 'blender_blocking/utils/json_io.py',
    'blender_blocking/synthetic/blender_builders.py', 'blender_blocking/synthetic/specs.py',
    'blender_blocking/integration/blender_ops/profile_loft_mesh.py', 'blender_blocking/geometry/profile_models.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def verify_file(binding):
    path = Path(binding['path'])
    if not path.is_file() or path.stat().st_size > 536870912 or sha(path) != binding['sha256']:
        raise ValueError('frozen input bytes differ: ' + str(path))
    return path


def read_json(binding):
    path = verify_file(binding)
    if path.stat().st_size > 1048576:
        raise ValueError('JSON input exceeds bounded size')
    return json.loads(path.read_bytes())


def non_camera(contract):
    measurement_signature(contract)
    return {key: value for key, value in contract.items() if key != 'camera'}


def load_reuse(entry, geometry_hash, expected_settings, actual_camera=None, *,
               family=FAMILY, producer='bounded_adaptive_family_checkpoint'):
    """Keep original ownership; new hard PNG is explicitly derived, not rendered."""
    import numpy as np
    manifest, lease = read_json(entry['manifest']), read_json(entry['lease'])
    root = Path(entry['owner_root'])
    if (manifest['state'] != 'succeeded' or lease['status'] != 'released'
            or manifest['producer'] != producer
            or manifest['run_root'] != str(root.resolve())
            or manifest['run_id'] != lease['run_id'] or manifest['owner_token'] != lease['owner_token']):
        raise ValueError('retained alpha ownership is not succeeded/released')
    artifacts = {row['path']: row for row in manifest['artifacts']}
    prefix = family + '/' + entry['label'] + '/'
    if set(entry['files']) != {'measurement.json', 'coverage.npy', 'alpha.exr'}:
        raise ValueError('retained alpha file inventory differs')
    for name, binding in entry['files'].items():
        if Path(binding['path']) != root / prefix / name or artifacts[prefix + name]['sha256'] != binding['sha256']:
            raise ValueError('retained alpha original owner/path/hash differs')
        verify_file(binding)
    meta = read_json(entry['files']['measurement.json'])
    if (meta['geometry_hashes'] != [geometry_hash] or meta['geometry_unchanged'] is not True
            or measurement_signature(meta['contract']) != meta['contract_sha256']
            or non_camera(meta['contract']) != expected_settings
            or meta['exr_sha256'] != entry['files']['alpha.exr']['sha256']):
        raise ValueError('retained alpha geometry/acquisition settings differ')
    if 'coverage_sha256' in meta and meta['coverage_sha256'] != entry['files']['coverage.npy']['sha256']:
        raise ValueError('retained measured coverage digest differs')
    camera = meta['contract']['camera']
    if camera['resolution'] != [512, 512] or camera['pixel_aspect'] != [1., 1.]:
        raise ValueError('retained alpha resolution/pixel aspect differs')
    if actual_camera is not None:
        require_same_frame(camera, actual_camera)
    coverage, mask = coverage_and_mask(np.load(entry['files']['coverage.npy']['path'], allow_pickle=False))
    if coverage.shape != (512, 512):
        raise ValueError('retained coverage dimensions differ')
    provenance = {'mode': 'reused_completed_original_measurement', 'owner_root': str(root),
        'manifest': entry['manifest'], 'lease': entry['lease'], 'files': deepcopy(entry['files']),
        'contract_sha256': meta['contract_sha256'], 'geometry_hash': geometry_hash,
        'actual_camera_sha256': camera_frame_sha256(camera),
        'hard_mask_derivation': 'new black-occupied PNG from retained coverage >= .5; zero new native frames'}
    return {**meta, 'coverage': coverage, 'mask': mask}, provenance


def require_scope(plan):
    expected = {'protocol': PROTOCOL, 'family': FAMILY, 'views': list(CANONICAL_VIEWS),
        'passes': list(INSPECTION_PASSES), 'resolution': [512, 512], 'native_frames': 27,
        'retained_alpha_passes': 3, 'logical_inspection_passes': 30, 'fits': 0,
        'raw_comparisons': 0, 'semantic_transactions': 0, 'qualifier_children': 1,
        'qualifier_timeout_seconds': 15, 'work_seconds': 85, 'join_seconds': 5,
        'threads': 2, 'committed_limit_bytes': MEMORY, 'rss_limit_bytes': MEMORY,
        'artifact_limit_bytes': 268435456, 'pass_write_reservation_bytes': 8388608,
        'artist_surface_limits': None, 'aggregate_accepted': False,
        'gates': {'minimum_area_iou': .7, 'minimum_boundary_iou': .8, 'maximum_signed_distance_loss': .05},
        'preservation_gates': {'support_error_max_world': .005, 'thickness_error_max_world': .005}}
    if any(plan.get(k) != v or (type(v) in (int, bool) and type(plan.get(k)) is not type(v)) for k, v in expected.items()):
        raise ValueError('triangle final frozen scope differs')
    if plan['case']['candidate_geometry_hash'] != CANDIDATE or plan['case']['source_geometry_hash'] != SOURCE:
        raise ValueError('triangle final selected source/body differs')
    if {role: {view: row['label'] for view, row in rows.items()} for role, rows in plan['reuse'].items()} != REUSE:
        raise ValueError('exact three retained mask slots required')


def verify_checkpoint(plan):
    entry, root = plan['case'], Path(plan['checkpoint_root'])
    manifest, lease = read_json(plan['checkpoint_manifest']), read_json(plan['checkpoint_lease'])
    receipt = read_json(plan['checkpoint_receipt'])
    if (receipt['protocol'] != 'bounded_adaptive_family_checkpoint_v1'
            or manifest['state'] != 'succeeded' or lease['status'] != 'released'
            or manifest['producer'] != 'bounded_adaptive_family_checkpoint'
            or manifest['run_root'] != str(root.resolve())
            or manifest['run_id'] != lease['run_id'] or manifest['owner_token'] != lease['owner_token']):
        raise ValueError('retained triangle checkpoint ownership differs')
    row = receipt['cases'][FAMILY]
    if (row['refined_geometry_hash'] != CANDIDATE or row['program'] != read_json(entry['candidate_program'])
            or row['refined_raw_surface']['candidate_geometry_hash'] != CANDIDATE
            or row['refined_raw_surface']['reference_geometry_hash'] != SOURCE
            or row['source_original_geometry_hash'] != SOURCE or row['source_replayed_geometry_hash'] != SOURCE
            or row['source_exact_oriented_equivalence'] is not True):
        raise ValueError('retained body/program/raw/source bindings differ')
    edit = row['semantic_edit_restoration']
    if (edit['control'] != 'corner_radius_world x1.1' or edit['response']['passed'] is not True
            or any(edit.get(k) is not True for k in ('passed', 'geometry_changed', 'same_source_pointer',
                                                   'original_mesh_restored', 'exact_indexed_restoration'))):
        raise ValueError('existing body-specific semantic restoration unavailable')
    artifacts = {item['path']: item for item in manifest['artifacts']}
    for binding in (plan['checkpoint_receipt'], entry['candidate_program'], entry['candidate_npz'], entry['candidate_obj']):
        relative = Path(binding['path']).relative_to(root).as_posix()
        if artifacts[relative]['sha256'] != binding['sha256']:
            raise ValueError('checkpoint owned body bytes differ')
    return row


def validate_plan(plan):
    require_scope(plan)
    needed = {str((ROOT / name).resolve()) for name in RUNTIME_FILES}
    needed.update(plan[k]['path'] for k in ('reference_workload', 'reference_receipt',
        'checkpoint_receipt', 'checkpoint_manifest', 'checkpoint_lease'))
    needed.update(plan['case'][k]['path'] for k in ('source_npz', 'candidate_program', 'candidate_npz', 'candidate_obj'))
    needed.update(row['path'] for row in plan['case']['original_masks'].values())
    needed.add(plan['qualification_python'])
    needed.update(row['path'] for row in plan['qualification_package_files'])
    for rows in plan['reuse'].values():
        for entry in rows.values():
            needed.update(row['path'] for row in entry['files'].values())
    if not needed.issubset(plan['input_sha256']) or len(plan['input_sha256']) > 72:
        raise ValueError('triangle final inputs not all frozen within bounded inventory')
    if sum(Path(p).stat().st_size for p in plan['input_sha256']) > 536870912:
        raise ValueError('frozen input total byte bound exceeded')
    for path, digest in plan['input_sha256'].items():
        verify_file({'path': path, 'sha256': digest})
    entry = plan['case']
    authored = {row['name']: row for row in read_json(plan['reference_workload'])['cases']}
    source = read_json(plan['reference_receipt'])['cases'][FAMILY]
    if (entry['source_case'] != authored[FAMILY] or entry['source_geometry_hash'] != source['geometry_hash']
            or entry['original_cameras'] != source['reference_cameras']
            or set(entry['original_cameras']) != set(CANONICAL_VIEWS)):
        raise ValueError('authored triangle/original camera declarations differ')
    for view in CANONICAL_VIEWS:
        camera_frame_sha256(entry['original_cameras'][view])
        if entry['original_masks'][view]['sha256'] != source['reference_cameras'][view]['png_sha256']:
            raise ValueError('original triangle mask binding differs')
    load_exact(entry['source_npz']['path'], SOURCE)
    candidate = load_exact(entry['candidate_npz']['path'], CANDIDATE)
    if (len(candidate.vertices), len(candidate.faces)) != (6239, 12474):
        raise ValueError('retained body inventory differs')
    verify_checkpoint(plan)
    for role, rows in plan['reuse'].items():
        for entry in rows.values():
            load_reuse(entry, SOURCE if role == 'source' else CANDIDATE, plan['measurement_settings'])


def render_role(obj, folder, role, plan, owner, deadline, receipt, publish):
    """Render missing masks and previews; bind original provenance for old alpha."""
    import bpy
    import numpy as np
    from PIL import Image
    from evaluation.canonical_artifacts import camera_record
    from integration.blender_ops.measurement_render import controlled_measurement_session, render_controlled_measurement
    from integration.blender_ops.silhouette_render import silhouette_session, render_silhouette_frame
    from reconstruction.native_geometry import evaluated_arrays
    from run_surface_quality_check import _replay_orthographic_camera
    before = evaluated_arrays(obj).content_hash
    expected = plan['case']['source_geometry_hash' if role == 'source' else 'candidate_geometry_hash']
    if before != expected:
        raise ValueError('exact role replay differs before first frame')
    actual, measurements = {}, {}
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'ORTHO'

    def frame(view):
        if time.monotonic() > deadline:
            raise TimeoutError('final inspection work deadline exceeded')
        owner.reserve_bytes(8388608)
        _replay_orthographic_camera(camera, {**plan['replay_clips'], **plan['case']['original_cameras'][view]})
        bpy.context.view_layer.update()
        return camera_record(camera, resolution=(512, 512),
            pixel_aspect=(bpy.context.scene.render.pixel_aspect_x, bpy.context.scene.render.pixel_aspect_y))

    def bind_pass(view, name, path, settings, provenance):
        record = camera_record(camera, resolution=(512, 512),
            pixel_aspect=(bpy.context.scene.render.pixel_aspect_x, bpy.context.scene.render.pixel_aspect_y))
        if evaluated_arrays(obj).content_hash != before:
            raise ValueError('inspection altered exact geometry')
        if view not in actual:
            actual[view] = {**record, 'geometry_hash': before, 'render_passes': {},
                'pass_artifacts': {}, 'executed_passes': [], 'reused_passes': []}
        require_same_frame(actual[view], record)
        actual[view]['render_passes'][name] = settings
        actual[view]['pass_artifacts'][name] = {'path': path.name, 'sha256': sha(path),
            'geometry_hash': before, 'camera_sha256': camera_frame_sha256(record),
            'actual_camera': record, 'provenance': provenance}
        key = 'reused_passes' if provenance['mode'].startswith('reused') else 'executed_passes'
        actual[view][key].append(name)
        if name == 'mask':
            actual[view]['png_sha256'] = sha(path)
        owner.register_file(path.relative_to(owner.root), 'final_output')
        publish(folder.relative_to(owner.root) / 'camera-snapshots.json', actual)
        publish('results.json', receipt)

    try:
        with controlled_measurement_session(target_objects=[obj], camera=camera, resolution=(512, 512),
                                            samples=64, filter_size=1.5) as session:
            if session.target_objects != [obj]:
                raise ValueError('unexpected measurement mesh descendants')
            for view in CANONICAL_VIEWS:
                current = frame(view)
                path = folder / (view + '-mask.exr')
                if view in plan['reuse'][role]:
                    measured, provenance = load_reuse(plan['reuse'][role][view], before,
                        plan['measurement_settings'], actual_camera=current, family=plan['family'],
                        producer=plan.get('retained_measurement_producer', 'bounded_adaptive_family_checkpoint'))
                    path.write_bytes(verify_file(plan['reuse'][role][view]['files']['alpha.exr']).read_bytes())
                    receipt['retained_alpha_passes_verified'] += 1
                else:
                    receipt['attempted_native_frames'] += 1
                    measured = render_controlled_measurement(session, path)
                    receipt['completed_native_frames'] += 1
                    provenance = {'mode': 'fresh_native_frame'}
                if (measured['geometry_hashes'] != [before] or measured['geometry_unchanged'] is not True
                        or non_camera(measured['contract']) != plan['measurement_settings']):
                    raise ValueError('measurement settings/geometry differ')
                require_same_frame(current, measured['contract']['camera'])
                owner.register_file(path.relative_to(owner.root), 'final_output')
                coverage = folder / (view + '-coverage.npy')
                np.save(coverage, measured['coverage'], allow_pickle=False)
                owner.register_file(coverage.relative_to(owner.root), 'final_output')
                png = folder / (view + '-mask.png')
                Image.fromarray(np.where(measured['mask'], 0, 255).astype(np.uint8)).save(png)
                metadata = {key: value for key, value in measured.items() if key not in ('coverage', 'mask')}
                metadata.update(coverage_sha256=sha(coverage), hard_mask_sha256=sha(png), provenance=provenance,
                    hard_mask_encoding='black occupied / white empty; actual linear alpha >= .5')
                publish(folder.relative_to(owner.root) / (view + '-measurement.json'), metadata)
                measurements[view] = measured
                bind_pass(view, 'mask', png, measured['contract'], provenance)
        normal_material = _normal_material()
        originals = list(obj.data.materials)
        try:
            with silhouette_session(target_objects=[obj], camera=camera, resolution=(512, 512), color_mode='BW',
                    transparent_bg=False, engine='BLENDER_WORKBENCH', force_material=False, ensure_light_obj=False) as session:
                if session.target_objects != [obj]:
                    raise ValueError('unexpected preview descendants')
                shading = session.scene.display.shading
                shading.light, shading.color_type = 'STUDIO', 'SINGLE'
                shading.single_color = (.65, .65, .65)
                shading.show_shadows, shading.show_cavity = True, False
                shading.background_type = 'WORLD'
                for name in ('neutral', 'normals'):
                    if name == 'normals':
                        session.scene.render.engine = 'BLENDER_EEVEE'
                        session.scene.render.image_settings.color_mode = 'RGB'
                        session.scene.eevee.taa_render_samples = 64
                        obj.data.materials.clear()
                        obj.data.materials.append(normal_material)
                    for view in CANONICAL_VIEWS:
                        frame(view)
                        path = folder / (view + '-' + name + '.png')
                        receipt['attempted_native_frames'] += 1
                        render_silhouette_frame(session, path)
                        receipt['completed_native_frames'] += 1
                        bind_pass(view, name, path, _settings(session.scene, name), {'mode': 'fresh_native_frame'})
        finally:
            obj.data.materials.clear()
            for material in originals:
                obj.data.materials.append(material)
            bpy.data.materials.remove(normal_material)
    finally:
        bpy.data.objects.remove(camera, do_unlink=True)
    if evaluated_arrays(obj).content_hash != before:
        raise ValueError('role geometry changed after inspection')
    for record in actual.values():
        record['geometry_unchanged_after_passes'] = True
    publish(folder.relative_to(owner.root) / 'camera-snapshots.json', actual)
    return actual, measurements


def source_declaration(plan, folder, cameras, equivalence):
    entry = plan['case']
    return {'authored_parameters': deepcopy(entry['source_case']['parameters']),
        'reference_npz': deepcopy(entry['source_npz']), 'reference_geometry_hash': entry['source_geometry_hash'],
        'regenerated_npz': bind(folder / 'evaluated-exact.npz'),
        'regenerated_geometry_hash': entry['source_geometry_hash'],
        'oriented_surface_sha256': equivalence['reference_oriented_surface'],
        'camera_records': bind(folder / 'camera-snapshots.json'),
        'neutral_artifacts': {view: bind(folder / (view + '-neutral.png')) for view in CANONICAL_VIEWS}}


def encoded_json(value):
    import numpy as np
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('unsupported receipt value: ' + type(item).__name__)
    return json.dumps(value, indent=2, allow_nan=False, default=scalar) + '\n'


def execute(plan, output, plan_binding):
    import bpy
    from evaluation.canonical_artifacts import canonical_artifact_inventory
    from evaluation.controlled_measurement import compare_controlled_measurements
    from evaluation.silhouette_eval import SilhouetteGateConfig
    from reconstruction.native_geometry import evaluated_arrays
    from reconstruction.output_qualification import qualify_retained_output
    from synthetic.quality_references import build_quality_reference
    from synthetic.quality_contracts import triangle_preservation
    from utils.run_ownership import OwnedRun
    from run_family_source_edit_check import _compile_exact
    from run_quality_coverage_check import save_mesh
    from run_reference_neutral_inspection import shading_state, style_comparison
    started = time.monotonic()
    if list(bpy.app.version) != [5, 2, 2] or bpy.app.build_hash.decode() != 'd13f752e3b9c':
        raise ValueError('frozen measurement renderer version/build differs')
    owner = OwnedRun(output, producer='retained_triangle_final_qualification', max_generated_bytes=268435456,
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
            path.write_text(encoded_json(value), encoding='utf-8')
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
            original = load_exact(entry['source_npz']['path'], SOURCE)
            retained = load_exact(entry['candidate_npz']['path'], CANDIDATE)
            source = build_quality_reference(entry['source_case']).object
            source_folder, candidate_folder = folder / 'source', folder / 'candidate'
            rebuilt, _ = save_mesh(source, source_folder)
            equivalence = reference_equivalence(original, rebuilt)
            row['source_equivalence'] = equivalence
            publish(Path(FAMILY) / 'source-equivalence.json', equivalence)
            if rebuilt.content_hash != SOURCE or equivalence['equivalent'] is not True:
                raise ValueError('source exact indexed replay differs before measurements')
            compiled, candidate, captured = _compile_exact(read_json(entry['candidate_program']), retained)
            saved, hashes = save_mesh(candidate, candidate_folder)
            row.update(hashes)
            row['candidate_replay'] = {'expected_indexed_hash': CANDIDATE,
                'actual_indexed_hash': saved.content_hash, 'exact': saved.content_hash == CANDIDATE}
            if saved.content_hash != CANDIDATE or captured.content_hash != CANDIDATE:
                raise ValueError('candidate exact indexed replay differs before measurements')
            publish(Path(FAMILY) / 'candidate/program.json', read_json(entry['candidate_program']))
            row.update(program=old['program'], refined_geometry_hash=CANDIDATE,
                refined_raw_surface=old['refined_raw_surface'], raw_receipt_reuse=plan['checkpoint_receipt'],
                editability={'status': 'passed', 'passed': True, 'receipt': plan['checkpoint_receipt'],
                    'geometry_hash': CANDIDATE, 'semantic_edit_restoration': old['semantic_edit_restoration']},
                surface={'status': 'unqualified', 'qualified_limits': None, 'raw': old['refined_raw_surface']},
                triangle_preservation=triangle_preservation(captured.vertices),
                prior_view_exposure=plan['prior_view_exposure'], crop_history=plan['crop_history'])
            styles = {'source': shading_state(source), 'candidate': shading_state(candidate)}
            styles['comparison'] = style_comparison(styles['source'], styles['candidate'])
            publish(Path(FAMILY) / 'shading-style.json', styles)
            source_cameras, source_measurements = render_role(source, source_folder, 'source', plan, owner,
                started + 85, receipt, publish)
            candidate_cameras, candidate_measurements = render_role(candidate, candidate_folder, 'candidate', plan, owner,
                started + 85, receipt, publish)
            if shading_state(source) != styles['source'] or shading_state(candidate) != styles['candidate']:
                raise ValueError('inspection altered native shading intent')
            inventories = {}
            for role, geometry, cameras, peer in (('source', rebuilt, source_cameras, candidate_cameras),
                                                   ('candidate', captured, candidate_cameras, source_cameras)):
                inventories[role] = canonical_artifact_inventory(folder / role, geometry_hash=geometry.content_hash,
                    camera_records=cameras, reference_camera_records=peer,
                    pass_states={name: 'completed' for name in INSPECTION_PASSES})
                publish(Path(FAMILY) / role / 'canonical-inspection.json', inventories[role])
            packet = matched_packet(inventories['source'], inventories['candidate'], source_cameras, candidate_cameras)
            packet['legacy_original_declaration'] = {view: {
                'actual_source_frame_matches_declared': camera_frame_sha256(source_cameras[view]) == camera_frame_sha256(entry['original_cameras'][view]),
                'original_clipping_status': 'unavailable', 'original_geometry_binding': 'legacy unavailable; no backfill'} for view in CANONICAL_VIEWS}
            publish(Path(FAMILY) / 'matched-packet.json', packet)
            gates = SilhouetteGateConfig(min_area_iou=.7, min_boundary_iou=.8, max_signed_distance_loss=.05)
            views = {view: compare_controlled_measurements(source_measurements[view], candidate_measurements[view],
                     view=view, config=gates) for view in CANONICAL_VIEWS}
            row.update(canonical_inspection=packet, presentation={'status': 'retained', 'styles': styles,
                'pixel_review': 'pending real output inspection; retained passes do not establish appearance'},
                silhouette={'status': 'passed' if all(v['passed'] for v in views.values()) else 'failed', 'views': views})
            publish('source-declarations.json', {FAMILY: source_declaration(plan, source_folder, source_cameras, equivalence)})
            if receipt['completed_native_frames'] != 27 or receipt['retained_alpha_passes_verified'] != 3:
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
            passed = (row['silhouette']['status'] == 'passed' and row['triangle_preservation']['passed']
                      and boundary.get('single_solid_qualified'))
            receipt['status'] = 'required_independent_gates_passed' if passed else 'required_independent_gates_failed'
            if not passed:
                owner.mark_failed('strict silhouettes/preservation/boundary failed or unavailable')
        except Exception as exc:
            row.update(status='blocked', reason=type(exc).__name__ + ': ' + str(exc))
            receipt['status'] = 'blocked'
            owner.mark_failed(row['reason'])
            try:
                encoded_json(receipt)
            except (ValueError, TypeError) as secondary:
                receipt['case_transport_error'] = type(secondary).__name__ + ': ' + str(secondary)
                receipt['cases'] = {FAMILY: {'status': 'blocked', 'geometry_hash': CANDIDATE, 'reason': row['reason']}}
        finally:
            for path in folder.rglob('*'):
                if path.is_file():
                    owner.register_file(path.relative_to(owner.root), 'final_output')
            receipt['elapsed_seconds'] = time.monotonic() - started
            publish('results.json', receipt)
    print('TRIANGLE_FINAL_RESULT=' + str(owner.root / 'results.json'), flush=True)
    return 0 if receipt['status'] == 'required_independent_gates_passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    binding = {'path': str(args.plan.resolve()), 'sha256': args.plan_sha256}
    plan = read_json(binding)
    validate_plan(plan)
    output = args.output.absolute()
    if output != output.resolve() or not output.is_relative_to(ROOT / 'temp/tasks'):
        raise ValueError('fresh output must stay below isolated worktree temp/tasks')
    return execute(plan, output, binding)


if __name__ == '__main__':
    raise SystemExit(main())
