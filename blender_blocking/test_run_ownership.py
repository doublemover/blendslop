"""Non-destructive run ownership planning; no historical cleanup fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from utils.run_ownership import OwnedRun, plan_run_reclamation


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


class OwnedProducerTests(unittest.TestCase):
    def test_new_run_active_then_released_keeps_shared_inputs_external(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            shared = parent / "owner-input.bin"
            shared.write_bytes(b"owner bytes")
            with OwnedRun(parent / "scratch", producer="fixture", shared_inputs={"image": str(shared)}) as owner:
                owner.reserve_bytes(3)
                (owner.root / "generated.bin").write_bytes(b"new")
                owner.register_file("generated.bin", "disposable")
                self.assertEqual(plan_run_reclamation(owner.root)["status"],"blocked")
            receipt = plan_run_reclamation(owner.root)
            self.assertEqual(receipt["status"],"dry_run_ready")
            self.assertEqual(receipt["eligible_bytes"],3)
            self.assertEqual(shared.read_bytes(),b"owner bytes")
            self.assertEqual((owner.root / "generated.bin").read_bytes(),b"new")

    def test_failure_cancel_and_publication_failure_preserve_original_error(self):
        for failure in (RuntimeError("primary failure"),KeyboardInterrupt("cancel")):
            with tempfile.TemporaryDirectory() as folder:
                owner = OwnedRun(folder,producer="fixture")
                with self.assertRaises(type(failure)):
                    with owner:
                        raise failure
                manifest = json.loads((owner.root / "run-ownership.json").read_text())
                self.assertEqual(manifest["state"],"cancelled" if isinstance(failure,KeyboardInterrupt) else "failed")
                self.assertEqual(json.loads((owner.root / "run-lease.json").read_text())["status"],"released")
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            owner = OwnedRun(folder,producer="fixture")
            original = RuntimeError("primary failure")
            with patch.object(owner,"_write_metadata",side_effect=OSError("receipt failure")):
                with self.assertRaises(RuntimeError) as caught:
                    with owner:
                        raise original
            self.assertIs(caught.exception,original)
            self.assertIn("receipt failure",original.__notes__[0])
            self.assertEqual(plan_run_reclamation(owner.root)["status"],"blocked")

    def test_budget_and_external_paths_fail_before_adoption(self):
        with tempfile.TemporaryDirectory() as folder:
            with OwnedRun(folder,producer="fixture",max_generated_bytes=4) as owner:
                with self.assertRaises(ValueError):owner.reserve_bytes(5)
                (owner.root / "generated.bin").write_bytes(b"1234")
                owner.register_file("generated.bin","diagnostic")
                with self.assertRaises(ValueError):owner.reserve_bytes(1)
                with self.assertRaises(ValueError):owner.register_file("../outside.bin","disposable")
                with self.assertRaises(ValueError):owner.register_file("./run-lease.json","disposable")
            self.assertEqual(plan_run_reclamation(owner.root)["retained_bytes"],4)

    def test_native_producer_timeout_joins_child_and_releases_failed_lease(self):
        from unittest.mock import Mock,patch
        import subprocess
        import numpy as np
        from reconstruction import native_qualification as native
        from reconstruction.native_geometry import GeometryArrays
        from primitives.capsule import CapsulePrimitive
        mesh = CapsulePrimitive().to_mesh_data(8)
        faces = np.asarray([triangle for face in mesh.faces for triangle in
                            ((face[0],face[i],face[i+1]) for i in range(1,len(face)-1))])
        data = GeometryArrays.capture(mesh.vertices,faces)
        child = Mock(returncode=-1)
        child.communicate.side_effect = [subprocess.TimeoutExpired("helper",15), (b"bounded log",None)]
        with tempfile.TemporaryDirectory() as folder, patch.object(native,"toolchain_identity",return_value="fixture-id"), \
                patch.object(native.subprocess,"Popen",return_value=child),patch.dict(native._CACHE,clear=True):
            receipt = native.qualify_geometry(data,python="fixture-python",ownership_root=folder)
            self.assertEqual(receipt["reason"],"helper_timeout")
            child.kill.assert_called_once()
            self.assertEqual(child.communicate.call_count,2)
            root = Path(receipt["owned_run_root"])
            self.assertEqual(json.loads((root / "run-ownership.json").read_text())["state"],"failed")
            self.assertEqual(plan_run_reclamation(root)["status"],"dry_run_ready")
            self.assertTrue((root / "input.npz").exists())
