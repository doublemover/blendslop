"""Non-destructive run ownership planning; no historical cleanup fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from utils.run_ownership import plan_run_reclamation


class RunOwnershipTests(unittest.TestCase):
    def fixture(self, root, *, lease="released"):
        records=[]
        for name, category in (("mesh.obj", "final_output"), ("error.txt", "diagnostic"),
                               ("scratch.bin", "disposable"), ("input-pointer.json", "shared_input")):
            data=(name + " content").encode()
            (root/name).write_bytes(data)
            records.append({"path": name, "category": category, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
        manifest={"schema_version":1,"run_root":str(root.resolve()),"run_id":"owned-test", "owner_token":"unique-owner",
                  "state":"succeeded","artifacts":records}
        (root/"run-ownership.json").write_text(json.dumps(manifest))
        (root/"run-lease.json").write_text(json.dumps({"run_id":"owned-test","owner_token":"unique-owner","status":lease}))
        return manifest

    def test_completed_plan_keeps_final_diagnostics_and_input_pointers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            receipt=plan_run_reclamation(root)
            self.assertEqual(receipt["status"],"dry_run_ready")
            self.assertEqual([r["path"] for r in receipt["eligible"]],["scratch.bin"])
            self.assertFalse(receipt["mutation_supported"])
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_active_or_mismatched_lease_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root,lease="active")
            self.assertFalse(plan_run_reclamation(root)["eligible"])
            self.assertTrue(plan_run_reclamation(root)["blockers"])
            (root/"run-lease.json").write_text('{}')
            self.assertTrue(plan_run_reclamation(root)["blockers"])

    def test_unknown_file_content_drift_and_missing_manifest_are_blocked(self):
        for mode in ("unknown", "drift", "missing"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                self.fixture(root)
                if mode=="unknown": (root/"owner-work.txt").write_text("retain")
                if mode=="drift": (root/"scratch.bin").write_text("changed content")
                if mode=="missing": (root/"run-ownership.json").unlink()
                self.assertFalse(plan_run_reclamation(root)["eligible"])
                self.assertTrue(plan_run_reclamation(root)["blockers"])

    def test_escape_duplicate_category_and_hash_bound_are_blocked(self):
        for mode in ("escape", "duplicate", "category", "bound"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                manifest=self.fixture(root)
                if mode=="escape": manifest["artifacts"][2]["path"]="../outside.bin"
                if mode=="duplicate": manifest["artifacts"].append(manifest["artifacts"][2])
                if mode=="category": manifest["artifacts"][2]["category"]="user-temporary"
                (root/"run-ownership.json").write_text(json.dumps(manifest))
                receipt=plan_run_reclamation(root,max_hash_bytes=1 if mode=="bound" else 1024)
                self.assertFalse(receipt["eligible"])
                self.assertTrue(receipt["blockers"])

    def test_linked_generated_path_cannot_adopt_external_file(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            root=Path(folder);manifest=self.fixture(root)
            target=Path(external)/"shared.bin";target.write_bytes(b"preserve shared bytes")
            try: (root/"linked.bin").symlink_to(target)
            except OSError: self.skipTest("host does not grant symlink creation")
            receipt=plan_run_reclamation(root)
            self.assertTrue(receipt["blockers"])
            self.assertEqual(target.read_bytes(),b"preserve shared bytes")


    def test_dot_prefixed_control_file_cannot_be_classified_disposable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture(root)
            data = (root / "run-lease.json").read_bytes()
            manifest["artifacts"].append({"path": "./run-lease.json", "category": "disposable",
                                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            (root / "run-ownership.json").write_text(json.dumps(manifest))
            self.assertTrue(plan_run_reclamation(root)["blockers"])
            self.assertFalse(plan_run_reclamation(root)["eligible"])
