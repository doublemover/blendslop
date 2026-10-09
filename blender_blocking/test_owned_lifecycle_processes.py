"""Real fresh-owner concurrency/crash and Windows locked-final contracts."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

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


if __name__=='__main__':unittest.main()
