"""Identity/snapshot-bound recovery inspection never releases crashed leases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_owned_lifecycle_processes import fixture_python
from unittest.mock import patch

from utils.run_ownership import OwnedRun
from utils.run_recovery import inspect_run_recovery

ROOT=Path(__file__).resolve().parent
CRASH="""import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from utils.run_ownership import OwnedRun
owner=OwnedRun(sys.argv[2],producer='fresh_recovery_crash_fixture')
(owner.root/'history.txt').write_bytes(b'retained crash history')
owner.register_file('history.txt','diagnostic')
print(json.dumps({'root':str(owner.root),'run_id':owner.run_id,'token':owner.owner_token}),flush=True)
os._exit(17)
"""


def snapshot(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


class RunRecoveryInspectionTests(unittest.TestCase):
    def inspect(self,owner,**kwargs):
        return inspect_run_recovery(owner.root,expected_run_id=owner.run_id,
                                    expected_owner_token=owner.owner_token,**kwargs)

    def test_active_lease_snapshot_is_bound_and_never_becomes_exit_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            with OwnedRun(folder,producer='inspection_fixture') as owner:
                (owner.root/'history.txt').write_bytes(b'keep');owner.register_file('history.txt','diagnostic')
                before=snapshot(owner.root);result=self.inspect(owner)
                self.assertEqual(result['decision'],'retain_active_lease')
                self.assertFalse(result['process_identity_verified'])
                self.assertFalse(result['descendant_joins_verified'])
                self.assertFalse(result['registered_artifacts_verified'])
                self.assertFalse(result['lease_release_supported'])
                self.assertEqual(result['snapshot']['manifest_sha256'],before['run-ownership.json'])
                again=self.inspect(owner,expected_snapshot=result['snapshot'])
                self.assertEqual(again['decision'],'retain_active_lease')
                self.assertEqual(before,snapshot(owner.root))

    def test_wrong_identity_and_stale_snapshot_cannot_authorize_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            with OwnedRun(folder,producer='inspection_fixture') as owner:
                initial=self.inspect(owner)
                wrong=inspect_run_recovery(owner.root,expected_run_id=owner.run_id,expected_owner_token='wrong token')
                self.assertEqual(wrong['decision'],'refuse_unverifiable_ownership')
                (owner.root/'new.txt').write_bytes(b'new');owner.register_file('new.txt','diagnostic')
                before=snapshot(owner.root);stale=self.inspect(owner,expected_snapshot=initial['snapshot'])
                self.assertEqual(stale['decision'],'refuse_unverifiable_ownership')
                self.assertTrue(any('snapshot changed' in value for value in stale['blockers']))
                self.assertEqual(before,snapshot(owner.root))

    def test_released_valid_run_needs_no_recovery_but_drift_and_unknowns_block(self):
        for mode in ('valid','drift','unknown','bound'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                with OwnedRun(folder,producer='inspection_fixture') as owner:
                    (owner.root/'file').write_bytes(b'recorded');owner.register_file('file','diagnostic')
                if mode=='drift':(owner.root/'file').write_bytes(b'changed')
                if mode=='unknown':(owner.root/'unknown').write_bytes(b'never adopt')
                before=snapshot(owner.root)
                result=self.inspect(owner,max_hash_bytes=1 if mode=='bound' else 268435456)
                self.assertEqual(result['decision'],'no_recovery_needed' if mode=='valid' else 'retain_unverified_released_run')
                self.assertEqual(result['registered_artifacts_verified'],mode=='valid')
                self.assertFalse(result['recovery_execution_supported'])
                self.assertEqual(before,snapshot(owner.root))

    def test_missing_unowned_and_oversized_metadata_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=inspect_run_recovery(root,expected_run_id='explicit',expected_owner_token='explicit')
            self.assertEqual(result['decision'],'refuse_unverifiable_ownership')
            with OwnedRun(root/'runs',producer='inspection_fixture') as owner:pass
            (owner.root/'run-ownership.json').write_bytes(b'x'*1048577)
            before=snapshot(owner.root);result=self.inspect(owner)
            self.assertTrue(any('bounded size' in value for value in result['blockers']))
            self.assertEqual(before,snapshot(owner.root))
            with self.assertRaises(ValueError):
                inspect_run_recovery(root,expected_run_id='',expected_owner_token='token')

    def test_metadata_change_during_inspection_is_not_a_stable_decision(self):
        import utils.run_recovery as recovery
        real=recovery._bounded_read;reads=0
        with tempfile.TemporaryDirectory() as folder:
            with OwnedRun(folder,producer='inspection_fixture') as owner:
                def read(path,limit):
                    nonlocal reads
                    value=real(path,limit);reads+=1
                    return value+b' ' if reads==3 else value
                with patch.object(recovery,'_bounded_read',side_effect=read):result=self.inspect(owner)
                self.assertEqual(result['decision'],'refuse_unverifiable_ownership')
                self.assertTrue(any('changed during' in value for value in result['blockers']))

    def test_real_crashed_joined_producer_keeps_active_lease_and_failure_history(self):
        with tempfile.TemporaryDirectory() as folder:
            child=subprocess.Popen([str(fixture_python()),'-c',CRASH,str(ROOT),folder],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                out,err=child.communicate(timeout=5.)
                self.assertEqual(child.returncode,17,err.decode(errors='replace'))
                entry=json.loads(out);root=Path(entry['root']);before=snapshot(root)
                result=inspect_run_recovery(root,expected_run_id=entry['run_id'],expected_owner_token=entry['token'])
                self.assertEqual(result['reported_pid'],child.pid)
                self.assertEqual(result['decision'],'retain_active_lease')
                self.assertFalse(result['process_identity_verified'])
                self.assertFalse(result['descendant_joins_verified'])
                self.assertEqual(json.loads((root/'run-lease.json').read_text())['status'],'active')
                self.assertEqual((root/'history.txt').read_bytes(),b'retained crash history')
                self.assertEqual(before,snapshot(root))
            finally:
                if child.poll() is None:child.kill()
                child.communicate(timeout=5.)


if __name__=='__main__':unittest.main()
