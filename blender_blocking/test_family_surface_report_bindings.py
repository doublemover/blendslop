"""Known producer layouts bind diagnostic raw metrics without selecting updates."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_family_surface_contracts.py'
SPEC = importlib.util.spec_from_file_location('family_policy_report_bindings', SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class FamilySurfaceReportBindingsTests(unittest.TestCase):
    def fixture(self, directory, *, legacy=False, arch=False, improved=True):
        family = 'asymmetric_multipart_solid' if legacy else 'concave_arch' if arch else 'capsule'
        geometry = 'a' * 64
        raw = {'candidate_geometry_hash': geometry, 'reference_geometry_hash': 'b' * 64,
               'sample_count_per_direction': 4096, 'seed': 61007,
               'symmetric_mean_distance_world': .001, 'distance_p95_world': .003}
        row = {'status': 'incomplete' if legacy else 'measured'}
        row.update({'geometry_hash': geometry, 'raw_surface': {'refined': raw}} if legacy
                   else {'refined_geometry_hash': geometry, 'refined_raw_surface': raw,
                         'bounded_checkpoint_improved': improved})
        producer = {'protocol': 'multipart_one_endpoint_update_v1' if legacy else
                    'bounded_arch_exterior_checkpoint_v1' if arch else
                    'bounded_adaptive_family_checkpoint_v1', 'status': 'measured',
                    'cases': {family: row}}
        path = Path(directory) / 'results.json'
        entry = {'candidate_geometry_hash': geometry,
                 'raw_field': ['raw_surface', 'refined'] if legacy else ['refined_raw_surface']}
        if not legacy:
            entry.update(geometry_field='refined_geometry_hash',
                         observation_scope='diagnostic_checkpoint')
        return family, producer, entry, path

    def bind(self, entry, path, producer):
        data = json.dumps(producer).encode('utf-8')
        path.write_bytes(data)
        entry['source_receipt'] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}

    def reused_baseline(self, directory):
        row = {'geometry_hash': 'a' * 64, 'aggregate_accepted': False,
               'program': {'program_id': 'triangle-support',
                           'root_nodes': [{'primitive_type': 'rounded_triangle'}]},
               'surface_observation': {'candidate_geometry_hash': 'a' * 64,
                                       'reference_geometry_hash': 'b' * 64}}
        path = Path(directory) / 'original-triangle.json'
        entry = {}; self.bind(entry, path, row)
        coverage = {'actual_rows': {}, 'families': {'rounded_triangle_dot': {
            'actual_status': 'reused', 'rerun': False, 'aggregate_accepted': False,
            'source_receipt': entry['source_receipt']}}}
        return row, path, coverage

    def test_reused_original_triangle_baseline_binds_real_producer_without_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            row, path, coverage = self.reused_baseline(directory)
            original = deepcopy(coverage)
            geometry, raw = REPORT.read_baseline_observation(coverage, 'rounded_triangle_dot')
            self.assertEqual(geometry, row['geometry_hash'])
            self.assertEqual(raw, row['surface_observation'])
            self.assertEqual(coverage, original)

    def test_reused_baseline_refuses_other_family_claims_layout_and_identity_drift(self):
        for change in ('family', 'rerun', 'accepted', 'primitive', 'program', 'identity', 'bytes'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                row, path, coverage = self.reused_baseline(directory)
                family = 'capsule' if change == 'family' else 'rounded_triangle_dot'
                declaration = coverage['families']['rounded_triangle_dot']
                if change == 'rerun': declaration['rerun'] = True
                if change == 'accepted': declaration['aggregate_accepted'] = True
                if change == 'primitive': row['program']['root_nodes'][0]['primitive_type'] = 'sphere'
                if change == 'program': row['program']['program_id'] = 'unknown'
                if change == 'identity': row['surface_observation']['candidate_geometry_hash'] = 'c' * 64
                if change == 'bytes': path.write_bytes(path.read_bytes() + b' ')
                else:
                    entry = {}; self.bind(entry, path, row)
                    declaration['source_receipt'] = entry['source_receipt']
                with self.assertRaises(ValueError): REPORT.read_baseline_observation(coverage, family)

    def test_real_legacy_layout_remains_selected_repair_raw_only(self):
        with tempfile.TemporaryDirectory() as directory:
            family, producer, entry, path = self.fixture(directory, legacy=True)
            self.bind(entry, path, producer)
            original = deepcopy(producer)
            raw, scope = REPORT.read_additional_observation(family, entry)
            self.assertEqual(raw, producer['cases'][family]['raw_surface']['refined'])
            self.assertEqual(scope['input_observation_scope'], 'selected_repair')
            self.assertIsNone(scope['producer_checkpoint_improved'])
            self.assertFalse(scope['current_row_selection_changed'])
            self.assertEqual(producer, original)

    def test_both_actual_adaptive_layouts_preserve_rejected_diagnostic_state(self):
        for arch in (False, True):
            for improved in (False, True):
                with self.subTest(arch=arch, improved=improved), tempfile.TemporaryDirectory() as directory:
                    family, producer, entry, path = self.fixture(directory, arch=arch, improved=improved)
                    self.bind(entry, path, producer)
                    raw, scope = REPORT.read_additional_observation(family, entry)
                    self.assertEqual(raw['candidate_geometry_hash'], entry['candidate_geometry_hash'])
                    self.assertEqual(scope['producer_checkpoint_improved'], improved)
                    self.assertEqual(scope['input_observation_scope'], 'diagnostic_checkpoint')
                    self.assertFalse(scope['current_row_selection_changed'])

    def test_unknown_or_crossed_selector_protocol_scope_is_refused(self):
        for change in ('protocol', 'selector', 'geometry_field', 'scope', 'checkpoint_type'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                family, producer, entry, path = self.fixture(directory)
                if change == 'protocol': producer['protocol'] = 'unrecognized_producer'
                if change == 'selector': entry['raw_field'] = ['baseline_raw_surface']
                if change == 'geometry_field': entry['geometry_field'] = 'geometry_hash'
                if change == 'scope': entry['observation_scope'] = 'selected_repair'
                if change == 'checkpoint_type': producer['cases'][family]['bounded_checkpoint_improved'] = 1
                self.bind(entry, path, producer)
                with self.assertRaises(ValueError): REPORT.read_additional_observation(family, entry)

    def test_recipe_geometry_and_raw_geometry_must_both_match_declared_identity(self):
        for field in ('recipe', 'raw', 'declared'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                family, producer, entry, path = self.fixture(directory)
                if field == 'recipe': producer['cases'][family]['refined_geometry_hash'] = 'c' * 64
                if field == 'raw': producer['cases'][family]['refined_raw_surface']['candidate_geometry_hash'] = 'c' * 64
                if field == 'declared': entry['candidate_geometry_hash'] = 'wrong'
                self.bind(entry, path, producer)
                with self.assertRaises(ValueError): REPORT.read_additional_observation(family, entry)

    def test_original_receipt_bytes_cannot_drift_after_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            family, producer, entry, path = self.fixture(directory)
            self.bind(entry, path, producer)
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'input bytes differ'):
                REPORT.read_additional_observation(family, entry)

    def test_incomplete_or_missing_adaptive_row_never_supplies_observation(self):
        for failure in ('running', 'failed', 'incomplete_row', 'missing_row', 'arch_wrong_family'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                family, producer, entry, path = self.fixture(directory)
                if failure in ('running', 'failed'): producer['status'] = failure
                if failure == 'incomplete_row': producer['cases'][family]['status'] = 'incomplete'
                if failure == 'missing_row': producer['cases'] = {}
                if failure == 'arch_wrong_family': producer['protocol'] = 'bounded_arch_exterior_checkpoint_v1'
                self.bind(entry, path, producer)
                with self.assertRaises(ValueError): REPORT.read_additional_observation(family, entry)


if __name__ == '__main__':
    unittest.main()
