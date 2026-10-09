"""Explicit nested-owner planning guards; no historical cleanup execution."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.run_ownership import OwnedRun, plan_run_reclamation
from utils.run_reclamation import ReclamationOwner, plan_run_reclamation_group


def binding(owner):
    return ReclamationOwner(str(owner.root), owner.run_id, owner.owner_token,
        hashlib.sha256((owner.root / 'run-ownership.json').read_bytes()).hexdigest(),
        hashlib.sha256((owner.root / 'run-lease.json').read_bytes()).hexdigest())


def snapshot(parent):
    return {str(p.relative_to(parent)): p.read_bytes()
            for p in parent.rglob('*') if p.is_file()}


def nested_fixture(folder, *, active_child=False, coowned=False):
    parent = OwnedRun(folder, producer='parent-fixture')
    child = OwnedRun(parent.root / 'qualification', producer='child-fixture')
    for owner, files in ((parent, (('scratch.bin', 'disposable'), ('error.txt', 'diagnostic'))),
                         (child, (('input.bin', 'disposable'), ('mesh.obj', 'final_output')))):
        for name, category in files:
            (owner.root / name).write_bytes((name + ' retained bytes').encode())
            owner.register_file(name, category)
    if not active_child:
        child.close()
    if coowned:
        parent.register_file((child.root / 'run-lease.json').relative_to(parent.root).as_posix(), 'disposable')
    parent.close()
    return parent, child


class RunReclamationGroupTests(unittest.TestCase):
    def test_explicit_nested_owners_compose_without_adoption_or_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, child = nested_fixture(folder)
            before = snapshot(Path(folder))
            flat = plan_run_reclamation(parent.root)
            self.assertEqual(flat['status'], 'blocked')
            result = plan_run_reclamation_group([binding(parent), binding(child)])
            self.assertEqual(result['status'], 'dry_run_ready', result['blockers'])
            self.assertEqual(result['owners'][0]['flat_plan'], flat)
            self.assertEqual(len(result['owners'][0]['nested_owned_files']), 4)
            self.assertEqual({(r['run_root'], r['path']) for r in result['eligible']},
                {(str(parent.root), 'scratch.bin'), (str(child.root), 'input.bin')})
            self.assertEqual(result['verified_registered_bytes'],
                result['retained_bytes'] + result['eligible_bytes'])
            self.assertFalse(result['mutation_supported'])
            self.assertFalse(result['lease_release_supported'])
            self.assertFalse(result['process_exit_verified'])
            self.assertEqual(snapshot(Path(folder)), before)

    def test_omitted_nested_owner_and_unregistered_child_file_block_group(self):
        for mode in ('omitted', 'extra'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                parent, child = nested_fixture(folder)
                owners = [binding(parent)] if mode == 'omitted' else [binding(parent), binding(child)]
                if mode == 'extra':
                    (child.root / 'unowned.txt').write_bytes(b'owner work stays')
                before = snapshot(Path(folder))
                result = plan_run_reclamation_group(owners)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['eligible'], [])
                self.assertEqual(snapshot(Path(folder)), before)

    def test_active_child_blocks_all_owners_and_keeps_active_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, child = nested_fixture(folder, active_child=True)
            try:
                before = snapshot(Path(folder))
                result = plan_run_reclamation_group([binding(parent), binding(child)])
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['eligible_bytes'], 0)
                self.assertEqual(snapshot(Path(folder)), before)
                self.assertEqual(json.loads((child.root / 'run-lease.json').read_text())['status'], 'active')
            finally:
                child.close()

    def test_content_drift_and_metadata_snapshot_drift_stay_blocking(self):
        for mode in ('content', 'metadata'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                parent, child = nested_fixture(folder)
                owners = [binding(parent), binding(child)]
                if mode == 'content':
                    (child.root / 'input.bin').write_bytes(b'different input')
                else:
                    lease = child.root / 'run-lease.json'
                    lease.write_bytes(lease.read_bytes() + b' ')
                before = snapshot(Path(folder))
                result = plan_run_reclamation_group(owners)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['eligible'], [])
                self.assertEqual(snapshot(Path(folder)), before)

    def test_duplicate_roots_or_claiming_child_control_file_are_refused(self):
        for mode in ('duplicate', 'coowned'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                parent, child = nested_fixture(folder, coowned=mode == 'coowned')
                owners = [binding(parent), binding(parent)] if mode == 'duplicate' else [binding(parent), binding(child)]
                result = plan_run_reclamation_group(owners)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['eligible'], [])
                self.assertTrue(any('duplicate' in s or 'more than one' in s for s in result['blockers']))

    def test_wrong_owner_token_and_run_identity_do_not_adopt_metadata(self):
        for field in ('owner_token', 'run_id'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                parent, child = nested_fixture(folder)
                bad = replace(binding(child), **{field: 'not-the-original-owner'})
                result = plan_run_reclamation_group([binding(parent), bad])
                self.assertEqual(result['status'], 'blocked')
                self.assertTrue(any('identity/token' in s for s in result['blockers']))
                self.assertEqual(result['eligible'], [])

    def test_global_hash_allowance_is_shared_across_independent_owners(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, child = nested_fixture(folder)
            parent_bytes = sum(row['bytes'] for row in parent.records.values())
            child_bytes = sum(row['bytes'] for row in child.records.values())
            result = plan_run_reclamation_group([binding(parent), binding(child)],
                max_hash_bytes=max(parent_bytes, child_bytes))
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue(any('combined registered' in s for s in result['blockers']))
            self.assertEqual(result['eligible_bytes'], 0)

    def test_metadata_change_during_group_inspection_blocks_final_plan(self):
        import utils.run_reclamation as planner
        with tempfile.TemporaryDirectory() as folder:
            parent, child = nested_fixture(folder)
            owners = [binding(parent), binding(child)]
            def audit(root, **kwargs):
                result = plan_run_reclamation(root, **kwargs)
                if Path(root) == child.root:
                    lease = child.root / 'run-lease.json'
                    lease.write_bytes(lease.read_bytes() + b' ')
                return result
            with patch.object(planner, 'plan_run_reclamation', side_effect=audit):
                result = planner.plan_run_reclamation_group(owners)
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue(any('snapshot changed' in s for s in result['blockers']))
            self.assertEqual(result['eligible'], [])

    def test_invalid_owner_or_resource_bounds_fail_before_inventory(self):
        with patch('utils.run_reclamation.plan_run_reclamation') as audit:
            for owners in ([], ['wildcard-root'], [ReclamationOwner('/', 'id', 'token', 'bad', 'bad')]):
                with self.assertRaises(ValueError):
                    plan_run_reclamation_group(owners)
            good = ReclamationOwner('C:/fixture', 'id', 'token', 'a' * 64, 'b' * 64)
            with self.assertRaises(ValueError):
                plan_run_reclamation_group([good] * 17)
            for limit in (True, 0, 268435457):
                with self.assertRaises(ValueError):
                    plan_run_reclamation_group([good], max_hash_bytes=limit)
            audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
