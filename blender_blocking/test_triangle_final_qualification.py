"""Focused retained-alpha provenance/settings and frozen triangle scope guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_triangle_final_qualification as packet
from evaluation.controlled_measurement import measurement_signature
from utils.run_ownership import OwnedRun


def contract():
    return {'protocol': 'opaque_linear_alpha_v1', 'hard_mask_threshold': .5,
        'camera': {'projection': 'ORTHO', 'matrix_world': np.eye(4).tolist(), 'ortho_scale': 2.,
            'shift_x': 0., 'shift_y': 0., 'clip_start': .1, 'clip_end': 1000.,
            'resolution': [512, 512], 'pixel_aspect': [1., 1.]},
        'renderer': {'engine': 'BLENDER_EEVEE'},
        'sampling': {'taa_render_samples': 64, 'filter_size': 1.5},
        'pixel_convention': 'top-left', 'surface_policy': 'opaque', 'encoding': 'linear alpha'}


def fixture(parent):
    owner = OwnedRun(parent, producer='bounded_adaptive_family_checkpoint')
    with owner:
        folder = owner.root / packet.FAMILY / 'source-fit'
        folder.mkdir(parents=True)
        values = np.zeros((512, 512), np.float32)
        values[10, 10:13] = [.499, .5, .501]
        np.save(folder / 'coverage.npy', values, allow_pickle=False)
        (folder / 'alpha.exr').write_bytes(b'fixture-original-producer-alpha-bytes')
        c = contract()
        meta = {'contract': c, 'contract_sha256': measurement_signature(c),
            'geometry_hashes': [packet.SOURCE], 'geometry_unchanged': True,
            'exr_sha256': packet.sha(folder / 'alpha.exr')}
        (folder / 'measurement.json').write_text(json.dumps(meta))
        for name in ('measurement.json', 'coverage.npy', 'alpha.exr'):
            owner.register_file((folder / name).relative_to(owner.root), 'final_output')
    entry = {'owner_root': str(owner.root), 'manifest': packet.bind(owner.root / 'run-ownership.json'),
        'lease': packet.bind(owner.root / 'run-lease.json'), 'label': 'source-fit',
        'files': {name: packet.bind(folder / name) for name in ('measurement.json', 'coverage.npy', 'alpha.exr')}}
    return entry, meta, values


def rebind_meta(entry, meta):
    path = Path(entry['files']['measurement.json']['path'])
    path.write_text(json.dumps(meta))
    entry['files']['measurement.json'] = packet.bind(path)
    manifest_path = Path(entry['manifest']['path'])
    manifest = json.loads(manifest_path.read_text())
    relative = path.relative_to(entry['owner_root']).as_posix()
    row = next(row for row in manifest['artifacts'] if row['path'] == relative)
    row.update(sha256=packet.sha(path), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    entry['manifest'] = packet.bind(manifest_path)


class TriangleFinalQualificationTests(unittest.TestCase):
    def test_verified_reuse_keeps_exact_coverage_and_original_provenance(self):
        with tempfile.TemporaryDirectory() as parent:
            entry, meta, original = fixture(parent)
            row, origin = packet.load_reuse(entry, packet.SOURCE, packet.non_camera(meta['contract']), meta['contract']['camera'])
            np.testing.assert_array_equal(row['coverage'], original)
            self.assertEqual(row['mask'][10, 10:13].tolist(), [False, True, True])
            self.assertEqual(origin['mode'], 'reused_completed_original_measurement')
            self.assertEqual(origin['files'], entry['files'])
            self.assertEqual(origin['owner_root'], entry['owner_root'])

    def test_drift_and_active_owner_refused(self):
        with tempfile.TemporaryDirectory() as parent:
            entry, meta, _ = fixture(parent)
            settings = packet.non_camera(meta['contract'])
            Path(entry['files']['alpha.exr']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                packet.load_reuse(entry, packet.SOURCE, settings)
        with tempfile.TemporaryDirectory() as parent:
            entry, meta, _ = fixture(parent)
            lease_path = Path(entry['lease']['path'])
            lease = json.loads(lease_path.read_text()); lease['status'] = 'active'
            lease_path.write_text(json.dumps(lease)); entry['lease'] = packet.bind(lease_path)
            with self.assertRaisesRegex(ValueError, 'succeeded/released'):
                packet.load_reuse(entry, packet.SOURCE, packet.non_camera(meta['contract']))

    def test_self_consistent_metadata_cannot_change_geometry_or_sampling(self):
        for change in ('geometry', 'sampling', 'coverage_digest'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as parent:
                entry, meta, _ = fixture(parent)
                expected = deepcopy(packet.non_camera(meta['contract']))
                if change == 'geometry':meta['geometry_hashes'] = ['e' * 64]
                elif change == 'sampling':
                    meta['contract']['sampling']['taa_render_samples'] = 32
                    meta['contract_sha256'] = measurement_signature(meta['contract'])
                else:meta['coverage_sha256'] = 'e' * 64
                rebind_meta(entry, meta)
                with self.assertRaisesRegex(ValueError, 'geometry/acquisition|coverage digest'):
                    packet.load_reuse(entry, packet.SOURCE, expected)

    def test_tiny_camera_or_clip_difference_refuses_reuse_without_tolerance(self):
        with tempfile.TemporaryDirectory() as parent:
            entry, meta, _ = fixture(parent)
            for key in ('shift_x', 'clip_start', 'clip_end'):
                current = deepcopy(meta['contract']['camera']); current[key] += 1e-12
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                    packet.load_reuse(entry, packet.SOURCE, packet.non_camera(meta['contract']), current)

    def test_frozen_scope_rejects_extra_work_boolean_count_and_missing_reuse(self):
        scope = {'protocol': packet.PROTOCOL, 'family': packet.FAMILY,
            'views': list(packet.CANONICAL_VIEWS), 'passes': list(packet.INSPECTION_PASSES),
            'resolution': [512, 512], 'native_frames': 27, 'retained_alpha_passes': 3,
            'logical_inspection_passes': 30, 'fits': 0, 'raw_comparisons': 0,
            'semantic_transactions': 0, 'qualifier_children': 1, 'qualifier_timeout_seconds': 15,
            'work_seconds': 85, 'join_seconds': 5, 'threads': 2,
            'committed_limit_bytes': packet.MEMORY, 'rss_limit_bytes': packet.MEMORY,
            'artifact_limit_bytes': 268435456, 'pass_write_reservation_bytes': 8388608,
            'artist_surface_limits': None, 'aggregate_accepted': False,
            'gates': {'minimum_area_iou': .7, 'minimum_boundary_iou': .8, 'maximum_signed_distance_loss': .05},
            'preservation_gates': {'support_error_max_world': .005, 'thickness_error_max_world': .005},
            'case': {'source_geometry_hash': packet.SOURCE, 'candidate_geometry_hash': packet.CANDIDATE},
            'reuse': {role: {v: {'label': label} for v, label in rows.items()} for role, rows in packet.REUSE.items()}}
        packet.require_scope(scope)
        for key, value in [('native_frames', 28), ('qualifier_children', True), ('fits', 1),
                           ('raw_comparisons', 1), ('semantic_transactions', 1), ('aggregate_accepted', True)]:
            changed = deepcopy(scope); changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen scope'):
                packet.require_scope(changed)
        changed = deepcopy(scope); changed['reuse']['source'].pop('top')
        with self.assertRaisesRegex(ValueError, 'three retained'):
            packet.require_scope(changed)

    def test_numpy_scalar_receipts_remain_finite(self):
        self.assertEqual(json.loads(packet.encoded_json({'pass': np.bool_(True), 'count': np.int64(27)})),
                         {'pass': True, 'count': 27})
        with self.assertRaises(ValueError):packet.encoded_json({'bad': np.float64(float('nan'))})


if __name__ == '__main__':unittest.main()
