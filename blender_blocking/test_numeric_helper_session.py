"""Owned warm IPC lifecycle with a local numeric fixture, not the DVX operator."""
from pathlib import Path
import sys,tempfile,unittest,subprocess,os
from unittest.mock import patch
from reconstruction.differentiable.helper_session import NumericHelperSession

SERVICE='''import argparse,json,os,pickle,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');p.add_argument('--directory');p.add_argument('--idle-timeout',type=float);args=p.parse_args()
r=Path(args.directory);temp=r/'ready.tmp';temp.write_text(json.dumps({'available':True,'pid':os.getpid(),'fixture':True,'prefix':sys.prefix,'pythonpath':os.environ.get('PYTHONPATH'),'pythonhome':os.environ.get('PYTHONHOME')}));os.replace(temp,r/'ready.json')
while not (r/'stop').exists():
 if time.time()-(r/'heartbeat').stat().st_mtime>args.idle_timeout:break
 for source in sorted(r.glob('*.input.pkl')):
  job=source.name[:-len('.input.pkl')]
  with source.open('rb') as f:value=pickle.load(f)
  if value.get('crash'):os._exit(9)
  time.sleep(value.get('delay',0.))
  response={'job':job,'status':'failed','error':'fixture failure'} if value.get('fail') else {'job':job,'status':'success','value':{'target':value['target'],'sum':sum(value['numbers'])}}
  source.unlink(missing_ok=True);out=r/(job+'.output.pkl');temp=out.with_suffix('.tmp')
  with temp.open('wb') as f:pickle.dump(response,f)
  os.replace(temp,out)
 time.sleep(.005)
'''


class NumericHelperSessionTests(unittest.TestCase):
    def test_parent_python_environment_does_not_leak_into_owned_child(self):
        with tempfile.TemporaryDirectory() as root:
            script,session=self.create(root)
            with session, patch.dict(os.environ, {'PYTHONPATH': '/incompatible/host/wheels', 'PYTHONHOME': '/incompatible/interpreter'}):
                probe=session.call(timeout_s=2.)
                self.assertIsNone(probe['pythonpath'])
                self.assertIsNone(probe['pythonhome'])

    def test_virtual_environment_interpreter_keeps_its_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            environment=Path(root)/'venv'
            subprocess.run([sys.executable,'-m','venv','--without-pip',str(environment)],check=True)
            executable=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            script=Path(root)/'numeric_fixture.py';script.write_text(SERVICE)
            with NumericHelperSession(executable,script,ownership_root=Path(root)/'runs') as session:
                probe=session.call(timeout_s=3.)
                self.assertEqual(Path(probe['prefix']),environment)
                self.assertIn(environment/'pyvenv.cfg',session.source_paths)

    def create(self,root):
        script=Path(root)/'numeric_fixture.py';script.write_text(SERVICE)
        return script,NumericHelperSession(sys.executable,script,idle_timeout_s=5.,ownership_root=Path(root)/"runs")

    def test_alternating_targets_share_the_owner_without_stale_results(self):
        with tempfile.TemporaryDirectory() as root:
            script,session=self.create(root)
            with session:
                probe=session.call(timeout_s=2.)
                first=session.call({'target':'first','numbers':[1.,2.]},timeout_s=2.)
                second=session.call({'target':'second','numbers':[5.,6.]},timeout_s=2.)
                self.assertEqual(first['sum'],3.);self.assertEqual(second['sum'],11.)
                self.assertEqual(second['target'],'second')
                self.assertEqual(first['helper_session']['pid'],second['helper_session']['pid'])
                self.assertEqual(session.starts,1)
                process=session.process;directory=session.root
            self.assertIsNotNone(process.poll())
            self.assertTrue(directory.exists())
            import json
            from utils.run_ownership import plan_run_reclamation
            manifest=json.loads((directory/'run-ownership.json').read_text())
            self.assertEqual(manifest['state'],'succeeded')
            self.assertEqual(plan_run_reclamation(directory)['status'],'dry_run_ready')
            self.assertTrue((directory/'worker.log').exists())

    def test_failed_job_does_not_poison_the_next_target(self):
        with tempfile.TemporaryDirectory() as root:
            script,session=self.create(root)
            with session:
                with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                    session.call({'target':'failed','numbers':[],'fail':True},timeout_s=2.)
                result=session.call({'target':'next','numbers':[7.]},timeout_s=2.)
                self.assertEqual(result['sum'],7.)
                self.assertEqual(session.starts,1)
            import json
            manifest=json.loads((session.root/'run-ownership.json').read_text())
            self.assertEqual(manifest['state'],'failed')
            events=[json.loads(line) for line in (session.root/'jobs.jsonl').read_text().splitlines()]
            self.assertEqual([event['status'] for event in events],['failed','succeeded'])
            self.assertTrue(list(session.root.glob('*.output.pkl')))

    def test_source_mutation_restarts_before_another_job(self):
        with tempfile.TemporaryDirectory() as root:
            script,session=self.create(root)
            with session:
                first=session.call({'target':'one','numbers':[1.]},timeout_s=2.)
                first_root=session.root
                script.write_text(SERVICE+'\n# changed source identity\n')
                second=session.call({'target':'two','numbers':[2.]},timeout_s=2.)
                self.assertEqual(session.starts,2)
                self.assertTrue(second['helper_session']['restarted'])
                self.assertNotEqual(first['helper_session']['pid'],second['helper_session']['pid'])
                self.assertNotEqual(first_root,session.root)
                import json
                self.assertEqual(json.loads((first_root/'run-lease.json').read_text())['status'],'released')

    def test_crash_and_timeout_clear_old_jobs_before_restart(self):
        with tempfile.TemporaryDirectory() as root:
            script,session=self.create(root)
            with session:
                with self.assertRaises(TimeoutError):
                    session.call({'target':'crash','numbers':[],'crash':True},timeout_s=2.)
                result=session.call({'target':'after-crash','numbers':[3.]},timeout_s=2.)
                self.assertEqual(result['sum'],3.)
                with self.assertRaises(TimeoutError):
                    session.call({'target':'slow','numbers':[4.],'delay':2.},timeout_s=.05)
                result=session.call({'target':'after-timeout','numbers':[5.]},timeout_s=2.)
                self.assertEqual(result['target'],'after-timeout')
                self.assertEqual(result['sum'],5.)
                self.assertEqual(session.starts,3)


if __name__=='__main__':unittest.main()
