"""Owned Poisson transport/lifecycle fixtures; no Open3D or Blender execution."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from config_models.backends import VisualHullConfig
from reconstruction.backends.visual_hull import poisson_bridge as bridge
from reconstruction.backends.visual_hull.postprocess import _postprocess_mesh
from volume import MeshExtractionResult
from test_owned_lifecycle_processes import fixture_python

CAP = 8 * 1024 ** 2


def mesh():
    return MeshExtractionResult(status='ok', method='fixture_source',
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]], np.float64),
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], np.int64))


def config(parent):
    return {'external_open3d_python':str(fixture_python()), 'postprocess_artifact_root':str(parent),
        'poisson_owned_supervision':True, 'poisson_committed_limit_bytes':1024**3,
        'poisson_rss_limit_bytes':512*1024**2, 'poisson_transport_limit_bytes':CAP, 'poisson_timeout_s':5.}


def result_files(command, log_path, *, bad_identity=False, bad_faces=False):
    transport = bridge._transport_module()
    get = lambda name: command[command.index(name)+1]
    root=Path(get('--root'));cap=int(get('--owned-output-limit-bytes'))
    source,_=transport._read_owned_npz(root/'input.npz',CAP)
    arrays={'vertices':source['vertices']+.125, 'faces':source['faces'], 'normals':np.zeros_like(source['vertices'])}
    if bad_faces:
        arrays['faces']=arrays['faces'].copy();arrays['faces'][0,0]=len(arrays['vertices'])
        np.savez(root/'output.npz',**arrays);output_sha=transport._file_sha(root/'output.npz',cap)
    else:
        output_sha=transport._write_owned_npz(root/'output.npz',arrays,cap,output=True)
    metadata={'source_sha256':get('--expected-input-sha256'), 'config_sha256':get('--expected-config-sha256'),
        'worker_sha256':get('--expected-worker-sha256'), 'output_sha256':output_sha,
        'owned_transport_protocol':'bounded_poisson_npz_v1', 'metrics':{'fixture':True}, 'topology':{}}
    if bad_identity:metadata['source_sha256']='0'*64
    transport._write_owned_json(root/'result.json',metadata,1048576)
    Path(log_path).write_bytes(b'pure numpy fixture; no native Poisson\n')
    return {'status':'succeeded','returncode':0,'lifecycle_complete':True,'fixture':True}


class OwnedPoissonBridgeTests(unittest.TestCase):
    def test_config_default_and_explicit_limits_without_volume_inference(self):
        default=VisualHullConfig();default.validate()
        self.assertFalse(default.to_dict()['poisson_owned_supervision'])
        self.assertIsNone(bridge._owned_limits(default.to_dict()))
        model=VisualHullConfig(poisson_owned_supervision=True,poisson_committed_limit_bytes=1024**3,
            poisson_rss_limit_bytes=512*1024**2,poisson_transport_limit_bytes=CAP)
        model.validate();self.assertEqual(bridge._owned_limits(model.to_dict())[:3],(1024**3,512*1024**2,CAP))
        for name,value in (('poisson_owned_supervision',1),('poisson_committed_limit_bytes',None),
            ('poisson_rss_limit_bytes',True),('poisson_transport_limit_bytes',256*1024**2+1),('poisson_timeout_s',float('nan'))):
            changed=model.to_dict();changed[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):VisualHullConfig(**changed).validate()
            with self.subTest(mapping=name),self.assertRaises(ValueError):bridge._owned_limits(changed)
        with self.assertRaises(ValueError):bridge._owned_limits({'poisson_owned_supervision':True,'memory_budget_mb':8192})

    def test_real_npz_compatibility_and_preallocation_header_byte_guards(self):
        transport=bridge._transport_module();source=mesh()
        with tempfile.TemporaryDirectory() as parent:
            path=Path(parent)/'arrays.npz';arrays={'vertices':source.vertices,'faces':source.faces}
            digest=transport._write_owned_npz(path,arrays,CAP);actual,bound=transport._read_owned_npz(path,CAP)
            self.assertEqual(digest,bound);np.testing.assert_array_equal(actual['vertices'],source.vertices)
            with np.load(path,allow_pickle=False) as ordinary:np.testing.assert_array_equal(ordinary['faces'],source.faces)
            tiny=Path(parent)/'tiny.npz'
            with self.assertRaisesRegex(ValueError,'before write'):transport._write_owned_npz(tiny,arrays,1)
            self.assertFalse(tiny.exists())
            evil=Path(parent)/'header.npz';header=io.BytesIO()
            np.lib.format.write_array_header_1_0(header,{'descr':'<f8','fortran_order':False,'shape':(10**12,3)})
            with zipfile.ZipFile(evil,'w') as z:
                z.writestr('vertices.npy',header.getvalue());normal=io.BytesIO();np.save(normal,source.faces,allow_pickle=False);z.writestr('faces.npy',normal.getvalue())
            with patch.object(np,'load',side_effect=AssertionError('reject before allocation')):
                with self.assertRaisesRegex(ValueError,'header/dtype/data extent'):transport._read_owned_npz(evil,CAP)
            compressed=Path(parent)/'compressed.npz';np.savez_compressed(compressed,vertices=np.zeros((10000,3)),faces=source.faces)
            with patch.object(np,'load',side_effect=AssertionError('bound uncompressed members first')):
                with self.assertRaisesRegex(ValueError,'uncompressed'):transport._read_owned_npz(compressed,32768)

    def test_result_identity_or_invalid_faces_preserve_source_and_history(self):
        for bad_identity,bad_faces in ((False,False),(True,False),(False,True)):
            with self.subTest(identity=bad_identity,faces=bad_faces),tempfile.TemporaryDirectory() as parent:
                source=mesh();before=source.vertices.copy();old=Path(parent)/'cpu-historical';old.mkdir();(old/'prior').write_bytes(b'keep')
                def supervise(command,**kwargs):return result_files(command,kwargs['log_path'],bad_identity=bad_identity,bad_faces=bad_faces)
                with patch('utils.owned_process_supervisor.run_bounded_process',side_effect=supervise):
                    if bad_identity or bad_faces:
                        with self.assertRaises(ValueError) as raised:bridge.run_external_poisson(source,'screened_poisson',config(parent))
                        owner=Path(raised.exception.poisson_ownership_receipt).parent
                    else:
                        result=bridge.run_external_poisson(source,'screened_poisson',config(parent))
                        np.testing.assert_array_equal(result.vertices,before+.125);self.assertEqual(result.method,'screened_poisson_external_open3d')
                        owner=Path(result.metrics['bridge_artifacts'])
                np.testing.assert_array_equal(source.vertices,before);self.assertEqual((old/'prior').read_bytes(),b'keep')
                self.assertEqual(json.loads((owner/'run-ownership.json').read_bytes())['state'],'failed' if bad_identity or bad_faces else 'succeeded')
                self.assertEqual(json.loads((owner/'run-lease.json').read_bytes())['status'],'released')

    def test_unconfirmed_or_truthy_lifecycle_never_releases_or_admits(self):
        for complete in (False,1,None):
            with self.subTest(complete=complete),tempfile.TemporaryDirectory() as parent:
                def supervise(command,**kwargs):
                    Path(kwargs['log_path']).write_bytes(b'unconfirmed mock receipt')
                    return {'status':'succeeded','returncode':0,'lifecycle_complete':complete}
                with patch('utils.owned_process_supervisor.run_bounded_process',side_effect=supervise):
                    with self.assertRaisesRegex(RuntimeError,'tree joins') as raised:bridge.run_external_poisson(mesh(),'screened_poisson',config(parent))
                owner=Path(raised.exception.poisson_ownership_receipt).parent
                self.assertEqual(json.loads((owner/'run-lease.json').read_bytes())['status'],'active')
                self.assertEqual(json.loads((owner/'run-ownership.json').read_bytes())['state'],'failed')

    def test_cancellation_preserves_primary_and_closes_only_attached_exact_true(self):
        for complete in (True,False):
            with self.subTest(complete=complete),tempfile.TemporaryDirectory() as parent:
                original=KeyboardInterrupt('primary fixture cancellation')
                original.owned_process_receipt={'status':'cancelled','returncode':124,'lifecycle_complete':complete}
                with patch('utils.owned_process_supervisor.run_bounded_process',side_effect=original):
                    with self.assertRaises(KeyboardInterrupt) as raised:bridge.run_external_poisson(mesh(),'screened_poisson',config(parent))
                self.assertIs(raised.exception,original);owner=Path(original.poisson_ownership_receipt).parent
                self.assertEqual(json.loads((owner/'run-lease.json').read_bytes())['status'],'released' if complete else 'active')
                self.assertEqual(json.loads((owner/'bridge-result.json').read_bytes())['status'],'cancelled')

    def test_admission_requires_an_actual_integer_zero_exit_status(self):
        for code in (False, 0., '0', None):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as parent:
                def supervise(command, **kwargs):
                    Path(kwargs['log_path']).write_bytes(b'invalid exit-status fixture')
                    return {'status':'succeeded', 'returncode':code, 'lifecycle_complete':True}
                with patch('utils.owned_process_supervisor.run_bounded_process', side_effect=supervise):
                    with self.assertRaisesRegex(RuntimeError, 'helper failed') as raised:
                        bridge.run_external_poisson(mesh(), 'screened_poisson', config(parent))
                owner = Path(raised.exception.poisson_ownership_receipt).parent
                self.assertFalse((owner/'output.npz').exists())
                self.assertEqual(json.loads((owner/'run-ownership.json').read_bytes())['state'], 'failed')
                self.assertEqual(json.loads((owner/'run-lease.json').read_bytes())['status'], 'released')

    def test_secondary_receipt_cancellation_cannot_mask_the_primary(self):
        from contextlib import ExitStack
        from utils.run_ownership import OwnedRun
        for stage in ('supervision', 'summary', 'register', 'close'):
            for secondary_type in (KeyboardInterrupt, SystemExit):
                for complete in ((True,) if stage == 'close' else (False, True)):
                    with self.subTest(stage=stage, secondary=secondary_type.__name__, complete=complete), tempfile.TemporaryDirectory() as parent:
                        primary = KeyboardInterrupt('primary fixture cancellation')
                        secondary = secondary_type('secondary receipt cancellation')
                        primary.owned_process_receipt = {'status':'cancelled', 'returncode':124, 'lifecycle_complete':complete}
                        transport = bridge._transport_module()
                        write, register = transport._write_owned_json, OwnedRun.register_file
                        def write_json(path, value, limit):
                            if Path(path).name == {'supervision':'supervision.json', 'summary':'bridge-result.json'}.get(stage):
                                raise secondary
                            return write(path, value, limit)
                        def register_file(owner, relative, category):
                            if stage == 'register' and relative == 'bridge-result.json':
                                raise secondary
                            return register(owner, relative, category)
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(bridge, '_transport_module', return_value=transport))
                            stack.enter_context(patch.object(transport, '_write_owned_json', side_effect=write_json))
                            stack.enter_context(patch.object(OwnedRun, 'register_file', register_file))
                            if stage == 'close':
                                stack.enter_context(patch.object(OwnedRun, 'close', side_effect=secondary))
                            stack.enter_context(patch('utils.owned_process_supervisor.run_bounded_process', side_effect=primary))
                            with self.assertRaises(KeyboardInterrupt) as raised:
                                bridge.run_external_poisson(mesh(), 'screened_poisson', config(parent))
                        self.assertIs(raised.exception, primary)
                        self.assertTrue(any('secondary receipt cancellation' in note for note in primary.__notes__))
                        owner = Path(primary.poisson_ownership_receipt).parent
                        self.assertEqual(json.loads((owner/'run-lease.json').read_bytes())['status'],
                                         'released' if complete and stage != 'close' else 'active')

    def test_required_optional_failure_keep_mesh_and_current_receipt(self):
        for required in (False,True):
            with self.subTest(required=required),tempfile.TemporaryDirectory() as parent:
                def supervise(command,**kwargs):
                    Path(kwargs['log_path']).write_bytes(b'fixture exit17')
                    return {'status':'failed','returncode':17,'lifecycle_complete':True}
                with patch('utils.owned_process_supervisor.run_bounded_process',side_effect=supervise):
                    actual,status=_postprocess_mesh(mesh(),'screened_poisson',config={**config(parent),'postprocess_required':required})
                self.assertEqual(actual.method,'fixture_source');self.assertEqual(status['status'],'failed' if required else 'skipped')
                self.assertTrue(Path(status['poisson_ownership_receipt']).is_file());self.assertTrue(Path(status['poisson_process_receipt']).is_file())

    def test_default_bridge_retains_legacy_command_threads_and_result_contract(self):
        with tempfile.TemporaryDirectory() as parent:
            class Child:
                def wait(self, timeout):
                    self.timeout = timeout
                    return 0
            def popen(command, **kwargs):
                self.assertNotIn('--owned-transport-limit-bytes', command)
                self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '4')
                self.assertEqual(kwargs['env']['OPENBLAS_NUM_THREADS'], '4')
                folder = Path(command[command.index('--root') + 1])
                np.savez(folder/'output.npz', vertices=mesh().vertices, faces=mesh().faces,
                         normals=np.zeros_like(mesh().vertices))
                (folder/'result.json').write_text(json.dumps({'metrics':{'legacy':True},'topology':{}}))
                return Child()
            with patch.object(bridge.subprocess, 'Popen', side_effect=popen):
                result = bridge.run_external_poisson(mesh(), 'screened_poisson', {
                    'external_open3d_python':str(fixture_python()), 'postprocess_artifact_root':parent})
            self.assertTrue(Path(result.metrics['bridge_artifacts']).name.startswith('cpu-'))
            self.assertEqual(result.method, 'screened_poisson_external_open3d')
            self.assertEqual(result.requested_method, 'fixture_source')
            self.assertNotIn('poisson_ownership_receipt', result.metrics)

    @unittest.skipUnless(os.name=='nt','complete tree fixture requires Windows Job Objects')
    def test_real_tiny_numpy_worker_has_kernel_tree_completion(self):
        with tempfile.TemporaryDirectory() as parent:
            worker=Path(parent)/'fixture_worker.py';original=bridge.WORKER
            helpers=['_transport_limit','_file_sha','_read_owned_npz','_write_owned_npz','_read_owned_json','_write_owned_json','_validate_arrays']
            worker.write_text('import runpy,argparse,numpy as np\nfrom pathlib import Path\n'
                +'ns=runpy.run_path('+repr(str(original))+')\n'
                +'globals().update({k:ns[k] for k in '+repr(helpers)+'})\n'
                +'''if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--method');p.add_argument('--owned-transport-limit-bytes',type=int);p.add_argument('--owned-output-limit-bytes',type=int);p.add_argument('--expected-input-sha256');p.add_argument('--expected-config-sha256');p.add_argument('--expected-worker-sha256');a=p.parse_args()
 d,h=_read_owned_npz(a.root/'input.npz',a.owned_transport_limit_bytes);c,ch=_read_owned_json(a.root/'config.json',1048576);wh=_file_sha(__file__,8388608)
 if (h,ch,wh)!=(a.expected_input_sha256,a.expected_config_sha256,a.expected_worker_sha256):raise ValueError('fixture frozen identity differs')
 oh=_write_owned_npz(a.root/'output.npz',{'vertices':d['vertices']+.125,'faces':d['faces'],'normals':np.zeros_like(d['vertices'])},a.owned_output_limit_bytes,output=True)
 _write_owned_json(a.root/'result.json',{'source_sha256':h,'config_sha256':ch,'worker_sha256':wh,'output_sha256':oh,'owned_transport_protocol':'bounded_poisson_npz_v1','metrics':{'fixture':True},'topology':{}},1048576)
 print('pure numpy fixture; no Open3D',flush=True)
''')
            with patch.object(bridge,'WORKER',worker):result=bridge.run_external_poisson(mesh(),'screened_poisson',config(Path(parent)/'runs'))
            np.testing.assert_array_equal(result.vertices,mesh().vertices+.125)
            receipt=json.loads(Path(result.metrics['poisson_process_receipt']).read_bytes())
            self.assertIs(receipt['lifecycle_complete'],True);self.assertTrue(receipt['job_assigned_before_resume']);self.assertTrue(receipt['job_active_zero'])
            self.assertTrue(all(r['kernel_joined'] for r in receipt['observed_processes']))
            self.assertEqual(json.loads(Path(result.metrics['poisson_ownership_receipt']).read_bytes())['state'],'succeeded')


if __name__=='__main__':unittest.main()
