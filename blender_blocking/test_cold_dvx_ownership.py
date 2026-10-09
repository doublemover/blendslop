"""Cold helper ownership using stdlib children; no DVX/Torch execution."""
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from reconstruction.differentiable.dvx_adapter import helper_call
from utils.run_ownership import OwnedRun, plan_run_reclamation


FIXTURE = """import argparse, json, os, pickle, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--probe', action='store_true')
p.add_argument('--input')
p.add_argument('--output')
a = p.parse_args()
if a.probe:
    print(json.dumps({'fixture': True, 'pythonpath': os.environ.get('PYTHONPATH'),
                      'pythonhome': os.environ.get('PYTHONHOME')}))
else:
    with Path(a.input).open('rb') as f:
        payload = pickle.load(f)
    value = {'target': payload['target'], 'sum': sum(payload['numbers'])}
    with Path(a.output).open('wb') as f:
        pickle.dump(value, f)
"""


class ColdDvxOwnershipTests(unittest.TestCase):
    def receipt(self, parent):
        roots = list(Path(parent).glob('owned-*'))
        self.assertEqual(len(roots), 1)
        root = roots[0]
        return root, json.loads((root / 'run-ownership.json').read_text())

    def child(self, *, returncode=0, stdout='', stderr=''):
        child = Mock(returncode=returncode)
        child.poll.side_effect = lambda: child.returncode
        child.communicate.return_value = (stdout, stderr)
        child.kill.side_effect = lambda: setattr(child, 'returncode', -9)
        return child

    def test_probe_and_fit_keep_receipts_and_isolate_parent_environment(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / 'cold_fixture.py'
            fixture.write_text(FIXTURE)
            children = []

            def spawn(args, **kwargs):
                rewritten = list(args)
                rewritten[2] = str(fixture)
                child = real_popen(rewritten, **kwargs)
                children.append(child)
                return child

            for payload in (None, {'target': 'actual', 'numbers': [2, 3]}):
                parent = Path(folder) / ('probe' if payload is None else 'fit')
                with patch('subprocess.Popen', side_effect=spawn), patch.dict(
                        'os.environ', {'PYTHONPATH': '/incompatible/host/wheels',
                                       'PYTHONHOME': '/incompatible/interpreter'}):
                    result = helper_call(sys.executable, payload, ownership_root=parent, timeout_s=3.)
                root, manifest = self.receipt(parent)
                self.assertEqual(manifest['state'], 'succeeded')
                self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'released')
                self.assertTrue(json.loads((root / 'helper-result.json').read_text())['child_joined'])
                self.assertIsNotNone(children[-1].poll())
                self.assertEqual(plan_run_reclamation(root)['status'], 'dry_run_ready')
                if payload is None:
                    self.assertTrue(result['fixture'])
                    self.assertIsNone(result['pythonpath'])
                    self.assertIsNone(result['pythonhome'])
                else:
                    self.assertEqual(result, {'target': 'actual', 'sum': 5})
                    self.assertTrue((root / 'input.pkl').exists())
                    self.assertTrue((root / 'output.pkl').exists())
                    self.assertNotIn('progress_paths', payload)

    def test_timeout_keeps_scored_checkpoint_and_joins_before_lease_release(self):
        child = self.child(returncode=None)
        child.communicate.side_effect = [subprocess.TimeoutExpired('fixture', .01), ('timeout log', '')]
        real_close = OwnedRun.close

        def close(owner, **kwargs):
            self.assertIsNotNone(child.returncode)
            return real_close(owner, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            def spawn(args, **kwargs):
                source = Path(args[args.index('--input') + 1])
                with (source.parent / 'progress.pkl').open('wb') as stream:
                    pickle.dump({'value': {'best_evaluation': 2, 'history': [4., 1.]}}, stream)
                return child

            with patch('subprocess.Popen', side_effect=spawn), patch.object(OwnedRun, 'close', close):
                result = helper_call(sys.executable, {'target': 'scored'}, ownership_root=folder, timeout_s=.01)
            self.assertEqual(result['best_evaluation'], 2)
            self.assertTrue(result['partial'])
            self.assertFalse(result['final_update_evaluated'])
            self.assertEqual(result['stop_reason'], 'helper_timeout')
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'failed')
            self.assertTrue((root / 'progress.pkl').exists())
            self.assertEqual(plan_run_reclamation(root)['status'], 'dry_run_ready')
            child.kill.assert_called_once()

    def test_timeout_without_scored_state_fails_and_retains_logs(self):
        child = self.child(returncode=None)
        child.communicate.side_effect = [subprocess.TimeoutExpired('fixture', .01), ('', 'timeout detail')]
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.Popen', return_value=child):
            with self.assertRaisesRegex(TimeoutError, 'no scored checkpoint'):
                helper_call(sys.executable, {'target': 'none'}, ownership_root=folder, timeout_s=.01)
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'failed')
            self.assertEqual((root / 'stderr-tail.txt').read_text(), 'timeout detail')
            self.assertTrue((root / 'input.pkl').exists())
            self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'released')

    def test_failed_child_keeps_primary_error_if_diagnostics_also_fail(self):
        child = self.child(returncode=9, stderr='primary fixture error')
        register = OwnedRun.register_file

        def fail_diagnostic(owner, relative, category):
            if relative == 'stderr-tail.txt':
                raise OSError('diagnostic fixture error')
            return register(owner, relative, category)

        with tempfile.TemporaryDirectory() as folder, patch('subprocess.Popen', return_value=child), \
                patch.object(OwnedRun, 'register_file', fail_diagnostic):
            with self.assertRaisesRegex(RuntimeError, 'primary fixture error') as caught:
                helper_call(sys.executable, {'target': 'failure'}, ownership_root=folder)
            self.assertIn('diagnostic fixture error', caught.exception.__notes__[0])
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'failed')
            self.assertTrue((root / 'input.pkl').exists())
            self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'released')

    def test_cancellation_joins_child_before_releasing_cancelled_run(self):
        child = self.child(returncode=None)
        original = KeyboardInterrupt('fixture cancellation')
        child.communicate.side_effect = [original, ('', 'joined after cancellation')]
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt) as caught:
                helper_call(sys.executable, {'target': 'cancel'}, ownership_root=folder)
            self.assertIs(caught.exception, original)
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'cancelled')
            self.assertIsNotNone(child.returncode)
            self.assertTrue(json.loads((root / 'helper-result.json').read_text())['child_joined'])
            self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'released')

    def test_unconfirmed_child_join_keeps_active_lease_and_primary_cancellation(self):
        child = self.child(returncode=None)
        original = KeyboardInterrupt('primary cancellation')
        child.communicate.side_effect = original
        child.kill.side_effect = OSError('termination unconfirmed')
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt) as caught:
                helper_call(sys.executable, {'target': 'unjoined'}, ownership_root=folder)
            self.assertIs(caught.exception, original)
            self.assertIn('termination unconfirmed', original.__notes__[0])
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'failed')
            self.assertFalse(json.loads((root / 'helper-result.json').read_text())['child_joined'])
            self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'active')
            self.assertEqual(plan_run_reclamation(root)['status'], 'blocked')

    def test_input_byte_preflight_stops_before_write_or_child(self):
        real_owner = OwnedRun

        def small_owner(parent, **kwargs):
            kwargs['max_generated_bytes'] = 1024
            return real_owner(parent, **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch('utils.run_ownership.OwnedRun', side_effect=small_owner), \
                patch('subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'budget exceeded before write'):
                helper_call(sys.executable, {'target': 'too large'}, ownership_root=folder)
            spawn.assert_not_called()
            root, manifest = self.receipt(folder)
            self.assertFalse((root / 'input.pkl').exists())
            self.assertEqual(manifest['state'], 'failed')
            self.assertEqual(json.loads((root / 'run-lease.json').read_text())['status'], 'released')

    def test_spawn_failure_retains_failure_receipt_without_child(self):
        original = OSError('fixture launch failed')
        with tempfile.TemporaryDirectory() as folder, patch('subprocess.Popen', side_effect=original):
            with self.assertRaises(OSError) as caught:
                helper_call(sys.executable, {'target': 'launch'}, ownership_root=folder)
            self.assertIs(caught.exception, original)
            root, manifest = self.receipt(folder)
            self.assertEqual(manifest['state'], 'failed')
            self.assertTrue(json.loads((root / 'helper-result.json').read_text())['child_joined'])
            self.assertEqual(plan_run_reclamation(root)['status'], 'dry_run_ready')


if __name__ == '__main__':
    unittest.main()
