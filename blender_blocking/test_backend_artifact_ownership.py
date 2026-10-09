"""Actual visual-hull OBJ adoption and fresh-process collision/failure fixtures."""
from contextlib import chdir
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_owned_lifecycle_processes import fixture_python
from unittest.mock import patch

from reconstruction import backend_artifact_ownership as owned
from reconstruction.backends.visual_hull.artifacts import write_visual_hull_mesh_artifact
from reconstruction.mesh_io import write_obj
from utils.run_ownership import plan_run_reclamation

ROOT = Path(__file__).resolve().parent
MESH = {"vertices": [(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)],
        "faces": [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]}
WORKER = """import json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1]).parent)]
import test_runner
from reconstruction.backends.visual_hull.artifacts import write_visual_hull_mesh_artifact
artifacts={}
path=write_visual_hull_mesh_artifact(root=Path(sys.argv[2]),candidate_id=sys.argv[3],backend_name='visual_hull_voxel',
    vertices=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)],faces=[(0,1,2)],artifacts=artifacts)
print(json.dumps({'path':str(path),'receipt':str(artifacts['mesh_export_ownership'])}))
"""


class BackendArtifactOwnershipTests(unittest.TestCase):
    def export(self, root, artifacts):
        return write_visual_hull_mesh_artifact(root=root,candidate_id="candidate",backend_name="visual_hull_voxel",
                                              vertices=MESH["vertices"],faces=MESH["faces"],artifacts=artifacts)

    def check_owner(self, receipt_path, state):
        root=Path(receipt_path).parent
        manifest=json.loads((root/'run-ownership.json').read_text())
        self.assertEqual(manifest['state'],state)
        self.assertEqual(json.loads((root/'run-lease.json').read_text())['status'],'released')
        plan=plan_run_reclamation(root)
        self.assertEqual(plan['status'],'dry_run_ready')
        self.assertEqual(plan['unknown_files'],[])
        self.assertEqual(plan['eligible'],[])
        return root,manifest

    def test_connected_producer_preserves_exact_obj_encoding_and_default_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);artifacts={}
            expected=write_obj(root/'expected.obj',MESH,header=('candidate candidate','visual_hull_voxel'))
            path=self.export(root,artifacts)
            self.assertEqual(path,root/'m/vh.obj')
            self.assertEqual(artifacts['mesh_obj'],path)
            self.assertEqual(path.read_bytes(),expected.read_bytes())
            owner,manifest=self.check_owner(artifacts['mesh_export_ownership'],'succeeded')
            receipt=json.loads(artifacts['mesh_export_ownership'].read_text())
            self.assertEqual(receipt['status'],'succeeded')
            self.assertEqual(receipt['subprocesses_started'],0)
            self.assertEqual(receipt['published_path'],str(path))
            self.assertEqual((owner/'mesh.obj').read_bytes(),path.read_bytes())
            self.assertEqual(manifest['published_output']['path'],str(path))

    def test_existing_final_and_unrelated_volume_stay_external_and_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'m').mkdir();previous=root/'m/vh.obj'
            previous.write_bytes(b'old retained mesh');shared=root/'shared-volume.npz';shared.write_bytes(b'old volume')
            artifacts={};path=self.export(root,artifacts)
            self.assertEqual(path,root/'m/vh_2.obj')
            self.assertEqual(previous.read_bytes(),b'old retained mesh')
            self.assertEqual(shared.read_bytes(),b'old volume')
            _,manifest=self.check_owner(artifacts['mesh_export_ownership'],'succeeded')
            self.assertEqual({entry['path'] for entry in manifest['artifacts']},{'mesh.obj','export-result.json'})

    def test_relative_result_paths_and_collision_bound_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as folder, chdir(folder):
            artifacts = {}
            actual = self.export(Path("candidate"), artifacts)
            self.assertEqual(actual, Path("candidate/m/vh.obj"))
            self.assertEqual(artifacts["mesh_obj"], actual)
            self.check_owner(artifacts["mesh_export_ownership"], "succeeded")
            parent = Path("full/m")
            parent.mkdir(parents=True)
            old_paths = [parent / "vh.obj"] + [parent / f"vh_{i}.obj" for i in range(2, 129)]
            for old_path in old_paths:
                old_path.write_bytes(b"prior final")
            artifacts = {}
            with self.assertRaisesRegex(FileExistsError, "collision retry bound"):
                self.export(Path("full"), artifacts)
            self.assertTrue(all(p.read_bytes() == b"prior final" for p in old_paths))
            self.assertFalse((parent / "vh_129.obj").exists())
            self.check_owner(artifacts["mesh_export_ownership"], "failed")

    def test_partial_write_and_publication_errors_keep_primary_history(self):
        for mode in ('partial','publication','cancel'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);artifacts={}
                original=KeyboardInterrupt('owned export cancellation') if mode=='cancel' else RuntimeError('primary '+mode)
                def writer(path,*args,**kwargs):
                    path.write_bytes(b'fresh partial')
                    raise original
                context=patch('reconstruction.mesh_io.write_obj',side_effect=writer) if mode!='publication' else \
                        patch.object(owned,'publish_file_no_clobber',side_effect=original)
                with context,self.assertRaises(type(original)) as caught:
                    self.export(root,artifacts)
                self.assertIs(caught.exception,original)
                self.assertFalse((root/'m/vh.obj').exists())
                owner,_=self.check_owner(artifacts['mesh_export_ownership'],'cancelled' if mode=='cancel' else 'failed')
                self.assertTrue((owner/'mesh.obj').is_file())
                self.assertIn(str(original),json.loads(artifacts['mesh_export_ownership'].read_text())['error'])

    def test_receipt_failure_after_publication_keeps_durable_final_binding(self):
        real=owned._receipt
        def receipt(owner,value):
            if value['status']=='succeeded':raise OSError('receipt publication fixture')
            return real(owner,value)
        with tempfile.TemporaryDirectory() as folder,patch.object(owned,'_receipt',side_effect=receipt):
            root=Path(folder);artifacts={}
            with self.assertRaisesRegex(OSError,'receipt publication fixture'):
                self.export(root,artifacts)
            self.assertTrue((root/'m/vh.obj').exists())
            _,manifest=self.check_owner(artifacts['mesh_export_ownership'],'failed')
            self.assertEqual(manifest['published_output']['path'],str(root/'m/vh.obj'))

    def test_secondary_diagnostics_do_not_replace_partial_write_error(self):
        primary=RuntimeError('primary partial fixture');real=owned._receipt
        def writer(path,*args,**kwargs):path.write_bytes(b'partial');raise primary
        def receipt(owner,value):
            if value['status']=='failed':raise OSError('secondary receipt fixture')
            return real(owner,value)
        with tempfile.TemporaryDirectory() as folder,patch('reconstruction.mesh_io.write_obj',side_effect=writer),\
                patch.object(owned,'_receipt',side_effect=receipt):
            artifacts={}
            with self.assertRaises(RuntimeError) as caught:self.export(Path(folder),artifacts)
            self.assertIs(caught.exception,primary)
            self.assertIn('secondary receipt fixture',primary.__notes__[0])
            self.check_owner(artifacts['mesh_export_ownership'],'failed')

    def test_real_concurrent_producers_publish_distinct_complete_meshes(self):
        children=[]
        with tempfile.TemporaryDirectory() as folder:
            try:
                for name in ('first','second'):
                    children.append(subprocess.Popen([str(fixture_python()),'-c',WORKER,str(ROOT),folder,name],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
                outputs=[]
                for child in children:
                    out,err=child.communicate(timeout=10.)
                    self.assertEqual(child.returncode,0,err.decode(errors='replace'))
                    outputs.append(json.loads(out))
                self.assertEqual({Path(row['path']).name for row in outputs},{'vh.obj','vh_2.obj'})
                self.assertNotEqual(outputs[0]['receipt'],outputs[1]['receipt'])
                for row in outputs:
                    self.assertTrue(Path(row['path']).read_text().startswith('# candidate '))
                    self.check_owner(row['receipt'],'succeeded')
            finally:
                for child in children:
                    if child.poll() is None:child.kill()
                    child.communicate(timeout=5.)


if __name__=='__main__':unittest.main()
