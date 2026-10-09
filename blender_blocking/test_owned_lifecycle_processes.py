"""Real fresh-owner concurrency/crash and Windows locked-final contracts."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

from utils.run_ownership import plan_run_reclamation

ROOT = Path(__file__).resolve().parent
WORKER = """from pathlib import Path
import json, os, sys
sys.path.insert(0, sys.argv[1])
from utils.run_ownership import OwnedRun
owner=OwnedRun(sys.argv[2],producer='bounded_process_fixture',max_generated_bytes=4096)
(owner.root/'fixture.bin').write_bytes(b'fresh-owned-bytes')
owner.register_file('fixture.bin','diagnostic')
print(json.dumps({'root':str(owner.root),'run_id':owner.run_id,'pid':os.getpid()}),flush=True)
if sys.argv[3]=='crash':os._exit(17)
if sys.stdin.readline().strip()!='release':raise RuntimeError('fixture release handshake missing')
owner.close()
"""


def fixture_python():
    candidates = [Path(sys.executable), Path(sys.prefix)/'bin/python.exe', Path(sys.prefix)/'bin/python3', Path(sys.prefix)/'bin/python']
    for candidate in candidates:
        if candidate.name.lower().startswith('python') and candidate.is_file():
            return candidate
    raise unittest.SkipTest('bundled standard-library fixture interpreter is unavailable')


def snapshot(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


class TestOwnedLifecycleProcesses(unittest.TestCase):
    def spawn(self, parent, mode='wait'):
        child = subprocess.Popen([str(fixture_python()),'-c',WORKER,str(ROOT),str(parent),mode],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.addCleanup(self.join_fixture,child)
        received = queue.Queue()
        threading.Thread(target=lambda:received.put(child.stdout.readline()),daemon=True).start()
        try:line = received.get(timeout=5.)
        except BaseException:
            child.kill();child.communicate(timeout=5.)
            raise
        try:entry=json.loads(line)
        except Exception:
            child.kill();_,err=child.communicate(timeout=5.)
            raise AssertionError('fixture startup failed: '+err.decode('utf8',errors='replace')[-4096:])
        return child,entry,Path(entry['root'])

    @staticmethod
    def join_fixture(child):
        if child.poll() is None:child.kill()
        child.communicate(timeout=5.)

    def test_concurrent_producers_have_separate_active_leases_and_keep_shared_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);shared=parent/'shared-input.bin';shared.write_bytes(b'owner input')
            first,a,root_a=self.spawn(parent/'runs')
            second,b,root_b=self.spawn(parent/'runs')
            self.assertNotEqual(root_a,root_b);self.assertNotEqual(a['run_id'],b['run_id'])
            for root in (root_a,root_b):
                before=snapshot(root);plan=plan_run_reclamation(root)
                self.assertEqual(plan['status'],'blocked');self.assertFalse(plan['eligible'])
                self.assertEqual(before,snapshot(root))
            for child,root in ((first,root_a),(second,root_b)):
                child.communicate(input=b'release\n',timeout=5.)
                self.assertEqual(child.returncode,0)
                self.assertEqual(plan_run_reclamation(root)['status'],'dry_run_ready')
                self.assertTrue((root/'fixture.bin').is_file())
            self.assertEqual(shared.read_bytes(),b'owner input')

    def test_crashed_joined_producer_retains_active_lease_and_requires_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            child,entry,root=self.spawn(Path(folder)/'runs',mode='crash')
            child.communicate(timeout=5.)
            self.assertEqual(child.returncode,17)
            before=snapshot(root);lease=json.loads((root/'run-lease.json').read_text())
            self.assertEqual(lease['status'],'active')
            plan=plan_run_reclamation(root)
            self.assertEqual(plan['status'],'blocked');self.assertFalse(plan['eligible'])
            self.assertTrue(any('recovery' in reason for reason in plan['blockers']))
            self.assertEqual(before,snapshot(root))

    @unittest.skipUnless(os.name=='nt','actual Windows file-sharing fixture')
    def test_locked_existing_final_is_never_overwritten(self):
        from utils.artifact_publication import publish_file_no_clobber
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        create.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);final=root/'final.bin';stage=root/'new-stage.bin'
            final.write_bytes(b'published owner bytes');stage.write_bytes(b'new completed stage')
            handle=create(str(final),0x80000000,0,None,3,0x80,None)
            if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
            try:
                with self.assertRaises(OSError):publish_file_no_clobber(stage,final)
            finally:kernel.CloseHandle(handle)
            self.assertEqual(final.read_bytes(),b'published owner bytes')
            self.assertEqual(stage.read_bytes(),b'new completed stage')


class TestNumericHelperBoundedShutdown(unittest.TestCase):
    @staticmethod
    def helper_class():
        # Load this stdlib-only file directly; package initialization would
        # import unrelated reconstruction backends, outside this test scope.
        spec = importlib.util.spec_from_file_location(
            'numeric_shutdown_fixture', ROOT / 'reconstruction/differentiable/helper_session.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NumericHelperSession

    def session(self, parent, waits):
        from utils.run_ownership import OwnedRun
        helper = self.helper_class()
        session = helper.__new__(helper)
        session.owner = OwnedRun(parent, producer='warm_numeric_helper', max_generated_bytes=4096)
        session.root = session.owner.root
        session.jobs = set()
        session.closed = False
        session.log = (session.root / 'worker.log').open('ab')
        session.log.write(b'retained primary-child log\n')
        session.log.flush()
        self.addCleanup(session.log.close)
        session.process = Mock()
        session.process.poll.return_value = None
        session.process.wait.side_effect = waits
        session.fixture_metadata = {name: (session.root / name).read_bytes()
                                    for name in ('run-ownership.json', 'run-lease.json')}
        return session

    @staticmethod
    def timeouts():
        return [subprocess.TimeoutExpired('numeric-fixture', timeout) for timeout in (.5, 1., 1.)]

    def assert_active(self, session, process, log):
        self.assertIs(session.process, process)
        self.assertIs(session.log, log)
        self.assertFalse(log.closed)
        self.assertFalse(session.closed)
        self.assertFalse(session.owner.closed)
        self.assertEqual(json.loads((session.root / 'run-lease.json').read_text())['status'], 'active')
        self.assertEqual((session.root / 'worker.log').read_bytes(), b'retained primary-child log\n')
        self.assertEqual(plan_run_reclamation(session.root)['status'], 'blocked')
        for name, before in session.fixture_metadata.items():
            self.assertEqual((session.root / name).read_bytes(), before)
        self.assertEqual(len(session.owner.auxiliary_errors), 1)

    def test_final_kill_wait_has_one_second_bound_and_confirmed_join_releases(self):
        with self.subTest():
            folder = self.enterContext(tempfile.TemporaryDirectory())
            session = self.session(Path(folder) / 'runs', self.timeouts()[:2] + [0])
            process, log = session.process, session.log
            session.close()
            self.assertEqual(process.wait.call_args_list, [call(timeout=.5), call(timeout=1.), call(timeout=1.)])
            process.terminate.assert_called_once_with()
            process.kill.assert_called_once_with()
            self.assertIsNone(session.process)
            self.assertIsNone(session.log)
            self.assertTrue(log.closed)
            self.assertTrue(session.closed)
            self.assertEqual(json.loads((session.root / 'run-lease.json').read_text())['status'], 'released')
            self.assertEqual(plan_run_reclamation(session.root)['status'], 'dry_run_ready')

    def test_unconfirmed_final_join_raises_retains_live_owner_and_can_retry(self):
        for direct_stop in (False, True):
            with self.subTest(direct_stop=direct_stop):
                folder = self.enterContext(tempfile.TemporaryDirectory())
                failures = self.timeouts()
                session = self.session(Path(folder) / 'runs', failures)
                process, log = session.process, session.log
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    (session._stop if direct_stop else session.close)()
                self.assertIs(caught.exception, failures[-1])
                self.assertEqual(process.wait.call_args_list, [call(timeout=.5), call(timeout=1.), call(timeout=1.)])
                self.assert_active(session, process, log)
                self.assertEqual(session.owner.state, 'failed')
                process.wait.side_effect = [0]
                session.close()
                self.assertTrue(session.closed)
                self.assertIsNone(session.process)
                self.assertTrue(log.closed)
                self.assertEqual(json.loads((session.root / 'run-lease.json').read_text())['status'], 'released')
                history = json.loads((session.root / 'run-ownership.json').read_text())
                self.assertEqual(history['state'], 'failed')
                self.assertEqual(history['error'], repr(failures[-1]))
                self.assertEqual(len(history['auxiliary_errors']), 1)
                self.assertEqual(plan_run_reclamation(session.root)['status'], 'dry_run_ready')

    def test_close_and_context_exit_preserve_primary_on_unconfirmed_join(self):
        for context_exit in (False, True):
            for error_type in (ValueError, KeyboardInterrupt):
                with self.subTest(context_exit=context_exit, error_type=error_type):
                    folder = self.enterContext(tempfile.TemporaryDirectory())
                    session = self.session(Path(folder) / 'runs', self.timeouts())
                    process, log = session.process, session.log
                    primary = error_type('existing primary numeric failure')
                    if context_exit:
                        with self.assertRaises(error_type) as caught:
                            with session:
                                raise primary
                        self.assertIs(caught.exception, primary)
                    else:
                        session.close(error=primary)
                    self.assert_active(session, process, log)
                    self.assertTrue(any('shutdown also failed' in note and 'TimeoutExpired' in note for note in primary.__notes__))
                    self.assertEqual(process.wait.call_args_list, [call(timeout=.5), call(timeout=1.), call(timeout=1.)])
                    process.wait.side_effect = [0]
                    session.close()
                    history = json.loads((session.root / 'run-ownership.json').read_text())
                    self.assertEqual(history['state'], 'cancelled' if error_type is KeyboardInterrupt else 'failed')
                    self.assertEqual(history['error'], repr(primary))
                    self.assertEqual(len(history['auxiliary_errors']), 1)
                    self.assertEqual(json.loads((session.root / 'run-lease.json').read_text())['status'], 'released')
                    self.assertEqual(plan_run_reclamation(session.root)['status'], 'dry_run_ready')

    def test_existing_primary_survives_post_join_finalization_failure(self):
        with self.subTest():
            folder = self.enterContext(tempfile.TemporaryDirectory())
            session = self.session(Path(folder) / 'runs', [0])
            process, log = session.process, session.log
            primary = RuntimeError('existing computation failure')
            secondary = OSError('owned metadata publication failed')
            # Exercise real OwnedRun.register_file, with an ordinary I/O error
            # at its metadata publication seam. _release_owner handles this
            # internally when a primary error is already supplied.
            with patch.object(session.owner, '_atomic_json', side_effect=secondary):
                with self.assertRaises(RuntimeError) as caught:
                    with session:
                        raise primary
            self.assertIs(caught.exception, primary)
            process.wait.assert_called_once_with(timeout=.5)
            self.assertIsNone(session.process)
            self.assertTrue(log.closed)
            self.assertFalse(session.closed)
            self.assertFalse(session.owner.closed)
            self.assertEqual(json.loads((session.root / 'run-lease.json').read_text())['status'], 'active')
            self.assertTrue(any('owned metadata publication failed' in note for note in primary.__notes__))
            self.assertTrue(any('owner receipt publication did not complete' in note for note in session.owner.auxiliary_errors))
            self.assertEqual(plan_run_reclamation(session.root)['status'], 'blocked')
            for name, before in session.fixture_metadata.items():
                self.assertEqual((session.root / name).read_bytes(), before)
            session.close()
            history = json.loads((session.root / 'run-ownership.json').read_text())
            self.assertEqual(history['state'], 'failed')
            self.assertEqual(history['error'], repr(primary))
            self.assertEqual(len(history['auxiliary_errors']), 1)
            self.assertEqual(plan_run_reclamation(session.root)['status'], 'dry_run_ready')


if __name__=='__main__':unittest.main()
