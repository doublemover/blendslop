"""Strict exact-retained family scope, alpha ownership and original edit guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_retained_family_final_qualification as packet
from evaluation.controlled_measurement import measurement_signature
from utils.run_ownership import OwnedRun


def fixture(parent, *, producer=packet.ORIGINAL_PRODUCER, measurement_hash=None):
    """Real fresh owner and NPZ-compatible coverage; no native geometry claim."""
    family = 'thin_plate'
    spec = packet.target(family)
    owner = OwnedRun(parent, producer=producer)
    wire = {'schema_version': '1', 'program_id': 'frozen-family-'+family,
        'root_nodes': [{'primitive_type': 'box', 'operation': 'add', 'children': [],
            'parameters': {'width_world': 1.6, 'depth_world': .08, 'height_world': 1.2,
                'rotation': np.eye(3).tolist(), 'x': .01, 'y': 0., 'z': 0.}}],
        'constraints': [], 'residual_patches': [], 'metadata': {'family': spec['metadata_family'], 'historical': True}}
    edited = deepcopy(wire)
    edited['root_nodes'][0]['parameters']['depth_world'] *= 1.05
    edit = {'control': 'depth_world x1.05', 'response': {'passed': True, 'checks': {'extents_response': True}},
        **dict.fromkeys(('passed', 'geometry_changed', 'same_source_pointer',
                        'original_mesh_restored', 'exact_indexed_restoration'), True)}
    camera = {'projection': 'ORTHO', 'matrix_world': np.eye(4).tolist(), 'ortho_scale': 2.,
        'shift_x': 0., 'shift_y': 0., **packet.REPLAY_CLIPS, 'resolution': [512, 512], 'pixel_aspect': [1., 1.]}
    contract = {**deepcopy(packet.MEASUREMENT_SETTINGS), 'camera': camera}
    history = {'oblique_145_40': 'prior supports inspected; excluded current local fit'}
    source_case = {'name': family, 'parameters': {'primitive': 'box'}}
    row = {'status': 'measured', 'program': wire, 'refined_geometry_hash': spec['candidate'],
        'source_original_geometry_hash': spec['source'], 'source_replayed_geometry_hash': spec['source'],
        'source_exact_oriented_equivalence': True, 'fixed_inputs_geometry_stable': True,
        'bounded_checkpoint_improved': True, 'selected_current_row_unchanged': True,
        'artist_surface_limits': None, 'aggregate_accepted': False, 'semantic_edit_restoration': edit,
        'refined_raw_surface': {'protocol': 'shared_world_area_sample_to_triangle_v1',
            'sample_count_per_direction': 4096, 'seed': 61007,
            'candidate_geometry_hash': spec['candidate'], 'reference_geometry_hash': spec['source']}}
    with owner:
        def write(relative, value):
            path = owner.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            owner.register_file(relative, 'final_output')
            return packet.shared.bind(path)
        bindings = {'source_case': source_case, 'source_geometry_hash': spec['source'],
            'candidate_geometry_hash': spec['candidate'], 'original_cameras': dict.fromkeys(packet.shared.CANONICAL_VIEWS, camera),
            'candidate_program': write(family+'/refined/program.json', wire),
            'semantic_edited_program': write(family+'/semantic-edited-program.json', edited)}
        for key, name in (('candidate_npz', 'evaluated-exact.npz'), ('candidate_obj', 'evaluated.obj')):
            path = owner.root/family/'refined'/name
            path.write_bytes(b'synthetic-not-native-geometry')
            owner.register_file(path.relative_to(owner.root), 'final_output')
            bindings[key] = packet.shared.bind(path)
        source = owner.root/'synthetic-source.npz'
        source.write_bytes(b'synthetic-source-not-native-geometry')
        bindings['source_npz'] = packet.shared.bind(source)
        receipt = write('results.json', {'protocol': 'bounded_adaptive_family_checkpoint_v1', 'status': 'measured', 'cases': {family: row}})
        frozen = write('frozen-workload.json', {'cases': [{'family': family, 'prior_view_exposure': history,
            'cameras': bindings['original_cameras'], 'source_case': source_case,
            'source_archive': str(source), 'source_geometry_hash': spec['source'],
            'fit_view': spec['fit_view'], 'semantic_control': spec['control'], 'semantic_multiplier': spec['multiplier']}]})
        label = 'source-heldout/oblique_145_40'
        folder = owner.root/family/label
        folder.mkdir(parents=True)
        values = np.zeros((512, 512), np.float32)
        values[10, 10:13] = [.499, .5, .501]
        np.save(folder/'coverage.npy', values, allow_pickle=False)
        (folder/'alpha.exr').write_bytes(b'synthetic-linear-alpha-file-not-native-render')
        meta = {'contract': contract, 'contract_sha256': measurement_signature(contract),
            'geometry_hashes': [measurement_hash or spec['source']], 'geometry_unchanged': True,
            'exr_sha256': packet.shared.sha(folder/'alpha.exr')}
        write(family+'/'+label+'/measurement.json', meta)
        for name in ('coverage.npy', 'alpha.exr'):
            owner.register_file((folder/name).relative_to(owner.root), 'final_output')
    manifest, lease = packet.shared.bind(owner.root/'run-ownership.json'), packet.shared.bind(owner.root/'run-lease.json')
    reuse = {'owner_root': str(owner.root), 'manifest': manifest, 'lease': lease, 'label': label,
        'files': {name: packet.shared.bind(folder/name) for name in ('measurement.json', 'coverage.npy', 'alpha.exr')}}
    plan = {**packet.scope(family), 'case': bindings, 'checkpoint_root': str(owner.root),
        'checkpoint_manifest': manifest, 'checkpoint_lease': lease, 'checkpoint_receipt': receipt,
        'historical_frozen_workload': frozen, 'prior_view_exposure': history,
        'reuse': {'source': {'oblique_145_40': reuse}, 'candidate': {}}}
    return plan, reuse, contract, values, wire, edited, edit


class RetainedFamilyFinalQualificationTests(unittest.TestCase):
    def test_allowlist_scope_rejects_family_hash_producer_style_or_budget_drift(self):
        plan = {**packet.scope('thin_plate'), 'case': {
            'source_geometry_hash': packet.target('thin_plate')['source'],
            'candidate_geometry_hash': packet.target('thin_plate')['candidate']},
            'reuse': {role: {view: {'label': label} for view, label in slots.items()}
                      for role, slots in packet.reuse_slots('thin_plate').items()}}
        packet.require_scope(plan)
        changes = [('family', 'sphere'), ('native_frames', 28), ('threads', True),
            ('style_policy', 'smooth all surfaces'), ('retained_measurement_producer', 'other'),
            ('aggregate_accepted', True), ('work_seconds', 90)]
        for key, value in changes:
            changed = deepcopy(plan)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                packet.require_scope(changed)
        changed = deepcopy(plan)
        changed['case']['candidate_geometry_hash'] = packet.target('cylinder')['candidate']
        with self.assertRaisesRegex(ValueError, 'source/body'):
            packet.require_scope(changed)
        changed = deepcopy(plan)
        changed['reuse']['source']['top'] = changed['reuse']['source'].pop('oblique_35_28')
        with self.assertRaisesRegex(ValueError, 'alpha slots'):
            packet.require_scope(changed)
        self.assertEqual(packet.scope('capsule')['native_frames'], 30)
        self.assertEqual(packet.reuse_slots('capsule'), {'source': {}, 'candidate': {}})

    def test_released_original_alpha_has_real_values_and_requires_geometry_frame_and_producer(self):
        for bad in ('none', 'geometry', 'producer', 'clipping'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as parent:
                plan, reuse, contract, values, _, _, _ = fixture(parent,
                    producer='unrelated' if bad == 'producer' else packet.ORIGINAL_PRODUCER,
                    measurement_hash='f'*64 if bad == 'geometry' else None)
                actual = deepcopy(contract['camera'])
                if bad == 'clipping':
                    actual['clip_end'] = 100.
                def load():
                    return packet.shared.load_reuse(reuse, packet.target('thin_plate')['source'],
                        packet.MEASUREMENT_SETTINGS, actual, family='thin_plate', producer=packet.ORIGINAL_PRODUCER)
                if bad == 'none':
                    row, provenance = load()
                    np.testing.assert_array_equal(row['coverage'], values)
                    self.assertEqual(row['mask'][10, 10:13].tolist(), [False, True, True])
                    self.assertEqual(provenance['owner_root'], plan['checkpoint_root'])
                else:
                    with self.assertRaises(ValueError):
                        load()

    def test_same_physical_mesh_is_insufficient_if_recipe_style_pose_or_restore_changed(self):
        with tempfile.TemporaryDirectory() as parent:
            _, _, _, _, wire, edited, edit = fixture(parent)
            before = deepcopy((wire, edited, edit))
            packet.require_existing_edit('thin_plate', wire, edited, edit)
            self.assertEqual((wire, edited, edit), before)
            for bad in ('style', 'pose', 'metadata', 'control', 'restoration', 'response'):
                changed, response = deepcopy(edited), deepcopy(edit)
                if bad == 'style':
                    changed['root_nodes'][0]['parameters']['weighted_normals'] = True
                elif bad == 'pose':
                    changed['root_nodes'][0]['parameters']['x'] += .01
                elif bad == 'metadata':
                    changed['metadata']['historical'] = False
                elif bad == 'control':
                    response['control'] = 'depth_world x1.1'
                elif bad == 'restoration':
                    response['exact_indexed_restoration'] = 1
                else:
                    response['response']['checks']['extents_response'] = False
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    packet.require_existing_edit('thin_plate', wire, changed, response)

    def test_checkpoint_requires_original_owned_recipe_edit_raw_and_exposure_bindings(self):
        with tempfile.TemporaryDirectory() as parent:
            plan, _, _, _, _, _, _ = fixture(parent)
            before = deepcopy(plan)
            row = packet.verify_checkpoint(plan)
            self.assertEqual(row['refined_geometry_hash'], packet.target('thin_plate')['candidate'])
            self.assertEqual(plan, before)
            changed = deepcopy(plan)
            changed['prior_view_exposure']['oblique_145_40'] = 'claimed fully blind'
            with self.assertRaisesRegex(ValueError, 'exposure history'):
                packet.verify_checkpoint(changed)
            changed = deepcopy(plan)
            changed['family'] = 'cylinder'
            with self.assertRaises((ValueError, KeyError)):
                packet.verify_checkpoint(changed)
            path = Path(plan['case']['semantic_edited_program']['path'])
            path.write_bytes(path.read_bytes()+b' ')
            changed = deepcopy(plan)
            changed['case']['semantic_edited_program'] = packet.shared.bind(path)
            with self.assertRaisesRegex(ValueError, 'owned artifact bytes'):
                packet.verify_checkpoint(changed)

    def test_actual_five_saved_recipe_transactions_and_dimension_aliases_remain_exact(self):
        for family, spec in packet.TARGETS.items():
            root = packet.ROOT/'temp/tasks/quality-continuation-20261008'/spec['owner']
            if not root.exists():
                self.skipTest('local retained campaign is unavailable outside this worktree')
            wire = packet.shared.read_json(packet.shared.bind(root/family/'refined/program.json'))
            edited = packet.shared.read_json(packet.shared.bind(root/family/'semantic-edited-program.json'))
            response = packet.shared.read_json(packet.shared.bind(root/'results.json'))['cases'][family]['semantic_edit_restoration']
            with self.subTest(family=family):
                packet.require_existing_edit(family, wire, edited, response)
                changed = deepcopy(edited)
                changed['root_nodes'][0]['parameters']['height_world' if family != 'torus' else 'major_radius'] += .001
                with self.assertRaisesRegex(ValueError, 'other recipe controls'):
                    packet.require_existing_edit(family, wire, changed, response)


if __name__ == '__main__':
    unittest.main()
