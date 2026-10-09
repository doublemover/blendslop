"""Fresh variant attempts and bounded stdlib children; no native Blender jobs."""
from __future__ import annotations
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import blender_blocking.refinement_lab.runner as runner_module
from blender_blocking.refinement_lab.contracts import ExperimentCase, ExperimentPlan, ExperimentVariant
from blender_blocking.refinement_lab.runner import (
    InProcessBlenderRunner, RunOptions, SubprocessRunner, _VariantAttempt, _run_variant_child,
)
from blender_blocking.utils.run_ownership import plan_run_reclamation

CHILD = """import json,os,sys,time
from pathlib import Path
result=Path(sys.argv[1]);mode=sys.argv[2]
os.write(1,b'fixture-start\\n');os.write(2,b'fixture-diagnostic\\n')
os.write(1,('mode:'+mode+'\\n').encode())
if mode=='sleep':time.sleep(30)
if mode=='noisy':
    for _ in range(128):
        os.write(1,b'A'*8192);os.write(2,b'B'*8192)
if mode=='invalid':result.write_text('not json',encoding='utf-8')
else:result.write_text(json.dumps({'passed':mode!='fail','validation_mode':'render-iou','average_iou':.91,'min_view_iou':.87}),encoding='utf-8')
os.write(1,b'\\nstdout-end\\n');os.write(2,b'\\nstderr-end\\n')
sys.exit(3 if mode=='fail' else 0)
"""


class VariantRunOwnershipTests(unittest.TestCase):
    def fixture(self, root, *, subprocess_mode=True, timeout=None, resume=False):
        refs={}
        for view in ('front','side','top'):
            path=root/(view+'.png');path.write_bytes(view.encode());refs[view]=path
        case=ExperimentCase('case','default-vase','builtin_sample',reference_paths=refs)
        variant=ExperimentVariant('baseline','baseline','profile_loft')
        plan=ExperimentPlan(plan_id='p',suite='default-vase',track='profile-loft-refinement',
            search='grid',objective='quality_win',output_root=root/'run',run_id='run',
            cases=(case,),variants=(variant,))
        options=RunOptions(html_report=False,write_overlays=False,write_bounds_debug=False,
            write_autopsy=False,append_global_index=False,write_lineage=False,
            write_adaptive_proposals=False,blender_executable=sys.executable,
            subprocess_blender=subprocess_mode,variant_timeout_s=timeout,resume_candidates=resume)
        cls=SubprocessRunner if subprocess_mode else InProcessBlenderRunner
        return cls(plan=plan,options=options),case,variant,refs

    def child_command(self, mode):
        def command(variant, **kwargs):
            return (sys.executable,'-c',CHILD,str(kwargs['result_json']),mode)
        return command

    def owner(self, result):
        root=result.artifacts['ownership_receipt'].parent
        manifest=json.loads((root/'run-ownership.json').read_text(encoding='utf-8'))
        lease=json.loads((root/'run-lease.json').read_text(encoding='utf-8'))
        return root,manifest,lease

    def assert_joined_audit(self, result, state):
        root,manifest,lease=self.owner(result)
        self.assertEqual(manifest['state'],state)
        self.assertEqual(lease['status'],'released')
        receipt=json.loads((root/'attempt.json').read_text(encoding='utf-8'))
        self.assertTrue(receipt['direct_child_joined'])
        self.assertTrue(receipt['log_drainers_joined'])
        audit=plan_run_reclamation(root)
        self.assertEqual(audit['status'],'dry_run_ready')
        return root

    def test_real_success_and_failure_preserve_history_paths_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp));folder=runner._case_variant_dir(case,variant)
            seen=[];aliases=None
            for mode in ('success','fail'):
                with patch.object(runner_module,'_variant_command',side_effect=self.child_command(mode)):
                    result=runner._run_one(case,variant,refs)
                root=self.assert_joined_audit(result,'succeeded' if mode=='success' else 'failed')
                self.assertIn(b'fixture-start',(root/'stdout.txt').read_bytes())
                self.assertIn(b'fixture-diagnostic',(root/'stderr.txt').read_bytes())
                self.assertEqual(result.artifacts['result'],result.result_json)
                self.assertEqual(result.avg_iou,.91)
                self.assertEqual(result.status,'pass' if mode=='success' else 'fail')
                self.assertEqual(result.exit_code,0 if mode=='success' else 3)
                seen.append((result.result_json,result.result_json.read_bytes()))
                names=('command.txt','config.json','stdout.txt','stderr.txt')
                if aliases is None:aliases={name:(folder/name).read_bytes() for name in names}
                else:self.assertEqual({name:(folder/name).read_bytes() for name in names},aliases)
            self.assertEqual(seen[0][0],folder/'result.json')
            self.assertEqual(seen[1][0],folder/'result_2.json')
            self.assertEqual(seen[0][0].read_bytes(),seen[0][1])
            # Every actual command still targets external canonical r/a directories.
            with patch.object(runner_module,'_run_variant_child',return_value=(0,'')):
                captured={}
                original=runner_module._variant_command
                def capture(*args,**kwargs):captured.update(kwargs);return original(*args,**kwargs)
                with patch.object(runner_module,'_variant_command',side_effect=capture):
                    runner._run_one(case,variant,refs)
            self.assertEqual(captured['render_dir'],(folder/'r').resolve())
            self.assertEqual(captured['artifact_root'],(folder/'a').resolve())
            self.assertIn('.variant-runs',captured['result_json'].parts)

    def test_noisy_real_child_drains_both_pipes_with_bounded_tails_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp))
            with patch.object(runner_module,'_variant_command',side_effect=self.child_command('noisy')):
                result=runner._run_one(case,variant,refs)
            root=self.assert_joined_audit(result,'succeeded')
            logs=json.loads((root/'logs.json').read_text(encoding='utf-8'))
            for name in ('stdout','stderr'):
                record=logs['streams'][name]
                self.assertGreater(record['byte_count'],1024*1024)
                self.assertEqual(record['retained_bytes'],32768)
                self.assertEqual((root/(name+'.txt')).stat().st_size,32768)
                self.assertTrue(record['truncated'])
                self.assertIn((name+'-end').encode(),(root/(name+'.txt')).read_bytes())

    def test_real_timeout_joins_before_failure_lease_release_and_retains_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp),timeout=1.5)
            children=[];real_popen=subprocess.Popen
            def spawn(*args,**kwargs):
                child=real_popen(*args,**kwargs);children.append(child);return child
            with patch.object(runner_module,'_variant_command',side_effect=self.child_command('sleep')),patch.object(runner_module.subprocess,'Popen',side_effect=spawn):
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    runner._run_one(case,variant,refs)
            self.assertIsNotNone(children[0].poll())
            root=Path(caught.exception.variant_ownership_receipt).parent
            self.assertEqual(json.loads((root/'run-lease.json').read_text())['status'],'released')
            self.assertEqual(json.loads((root/'run-ownership.json').read_text())['state'],'failed')
            self.assertIn(b'fixture-start',(root/'stdout.txt').read_bytes())
            self.assertEqual(plan_run_reclamation(root)['status'],'dry_run_ready')

    def test_cancellation_terminates_only_owned_real_child_and_keeps_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=_VariantAttempt(Path(tmp));real_popen=subprocess.Popen;children=[]
            def spawn(*args,**kwargs):
                child=real_popen(*args,**kwargs);original=child.wait;calls=[0]
                def wait(*args,**kwargs):
                    calls[0]+=1
                    if calls[0]==1:raise KeyboardInterrupt('fixture cancellation')
                    return original(*args,**kwargs)
                child.wait=wait;children.append(child);return child
            with patch.object(runner_module.subprocess,'Popen',side_effect=spawn):
                with self.assertRaisesRegex(KeyboardInterrupt,'fixture cancellation'):
                    with attempt:
                        attempt.record(('fixture',),{})
                        _run_variant_child((sys.executable,'-c','import time;time.sleep(30)'),cwd=tmp,timeout_s=None,attempt=attempt)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual(json.loads((attempt.root/'run-lease.json').read_text())['status'],'released')
            self.assertEqual(json.loads((attempt.root/'run-ownership.json').read_text())['state'],'cancelled')

    def test_unconfirmed_join_retains_active_lease_and_original_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=_VariantAttempt(Path(tmp));child=Mock(pid=321)
            child.stdout=io.BytesIO(b'out');child.stderr=io.BytesIO(b'err')
            child.poll.return_value=None
            original=subprocess.TimeoutExpired('fixture',.1)
            child.wait.side_effect=[original,subprocess.TimeoutExpired('fixture',5),subprocess.TimeoutExpired('fixture',5)]
            with patch.object(runner_module.subprocess,'Popen',return_value=child):
                with self.assertRaises(subprocess.TimeoutExpired) as caught:
                    with attempt:
                        attempt.record(('fixture',),{})
                        _run_variant_child(('fixture',),cwd=tmp,timeout_s=.1,attempt=attempt)
            self.assertIs(caught.exception,original)
            child.terminate.assert_called_once();child.kill.assert_called_once()
            self.assertEqual(json.loads((attempt.root/'run-lease.json').read_text())['status'],'active')
            self.assertEqual(plan_run_reclamation(attempt.root)['status'],'blocked')
            self.assertTrue(any('unconfirmed' in note for note in caught.exception.__notes__))

    def test_launch_publication_and_invalid_json_failures_retain_stages(self):
        for mode in ('launch','publication','invalid'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                runner,case,variant,refs=self.fixture(Path(tmp));folder=runner._case_variant_dir(case,variant)
                folder.mkdir(parents=True);old=folder/'result.json';old.write_bytes(b'previous result')
                with patch.object(runner_module,'_variant_command',side_effect=self.child_command('invalid' if mode=='invalid' else 'success')):
                    if mode=='launch':
                        context=patch.object(runner_module.subprocess,'Popen',side_effect=OSError('launch blocked'))
                    elif mode=='publication':
                        context=patch('blender_blocking.utils.artifact_publication.publish_file_no_clobber',side_effect=PermissionError('publication blocked'))
                    else:context=patch.object(runner_module,'_VARIANT_LOG_TAIL_BYTES',32768)
                    with context,self.assertRaises(Exception):runner._run_one(case,variant,refs)
                self.assertEqual(old.read_bytes(),b'previous result')
                roots=list((folder/'.variant-runs').glob('owned-*'));self.assertEqual(len(roots),1)
                self.assertTrue((roots[0]/'command.txt').exists())
                if mode!='launch':self.assertTrue((roots[0]/'result.json').exists())
                self.assertEqual(json.loads((roots[0]/'run-lease.json').read_text())['status'],'released')
                self.assertEqual(json.loads((roots[0]/'run-ownership.json').read_text())['state'],'failed')

    def test_inprocess_attempts_and_outer_errors_use_suffixes_and_preserve_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp),subprocess_mode=False);seen=[]
            def invoke(*args,**kwargs):
                seen.append(kwargs);Path(kwargs['result_json']).write_text('{"passed":true,"average_iou":.9}'.replace('.9','0.9'))
                return True
            with patch('blender_blocking.test_e2e_validation.test_with_custom_images',side_effect=invoke):
                first=runner._run_one(case,variant,refs)
                old_command=(first.result_json.parent/'command.txt').read_bytes()
                second=runner._run_one(case,variant,refs)
            self.assertEqual(second.result_json.name,'result_2.json')
            self.assertEqual((first.result_json.parent/'command.txt').read_bytes(),old_command)
            folder=runner._case_variant_dir(case,variant)
            self.assertEqual(seen[0]['render_output_dir'],folder/'r')
            self.assertEqual(seen[0]['artifact_root'],folder/'a')
            self.assert_joined_audit(first,'succeeded');self.assert_joined_audit(second,'succeeded')
            error=runner._candidate_execution_error_result(case,variant,refs,RuntimeError('outer'),started_utc='now',elapsed_s=1.)
            ref_error=runner._reference_generation_error_result(case,variant,RuntimeError('references'))
            self.assertEqual(error.result_json.name,'result_3.json');self.assertEqual(ref_error.result_json.name,'result_4.json')
            self.assert_joined_audit(error,'failed');self.assert_joined_audit(ref_error,'failed')

    def test_actual_resume_cache_hit_creates_no_new_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp),resume=True)
            with patch.object(runner_module,'_variant_command',side_effect=self.child_command('success')):
                first=runner._run_one(case,variant,refs)
            runner._write_candidate_state(case,variant,refs,first)
            parent=runner._case_variant_dir(case,variant)/'.variant-runs';before=set(parent.iterdir())
            with patch.object(runner,'_prepare_case_references',return_value=refs),patch.object(runner,'_write_manifest',return_value={}),patch.object(runner,'_run_one',side_effect=AssertionError('cache miss')):
                ok,results=runner.run()
            self.assertTrue(ok);self.assertEqual(results[0].metrics['cache']['source'],'local_resume')
            self.assertEqual(set(parent.iterdir()),before)
            self.assertEqual(results[0].result_json,first.result_json)

    def test_config_failure_retains_command_config_and_primary_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp),subprocess_mode=False)
            with patch.object(runner_module,'_apply_variant_to_config',side_effect=ValueError('invalid requested config')):
                with self.assertRaisesRegex(ValueError,'invalid requested config') as caught:
                    runner._run_one(case,variant,refs)
            root=Path(caught.exception.variant_ownership_receipt).parent
            self.assertIn('--result-json',(root/'command.txt').read_text(encoding='utf-8'))
            self.assertIn('variant',json.loads((root/'config.json').read_text(encoding='utf-8')))
            self.assertEqual(json.loads((root/'run-lease.json').read_text())['status'],'released')
            self.assertEqual(json.loads((root/'run-ownership.json').read_text())['state'],'failed')

    def test_secondary_receipt_error_does_not_replace_primary_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=_VariantAttempt(Path(tmp));child=Mock(pid=123)
            child.stdout=io.BytesIO(b'completed log');child.stderr=io.BytesIO(b'diagnostic')
            child.poll.return_value=None
            original=KeyboardInterrupt('primary cancellation')
            child.wait.side_effect=[original,0]
            register=attempt.owner.register_file
            def fail_receipt(relative,category):
                if str(relative)=='logs.json':raise OSError('secondary receipt failure')
                return register(relative,category)
            with patch.object(runner_module.subprocess,'Popen',return_value=child),patch.object(attempt.owner,'register_file',side_effect=fail_receipt):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with attempt:
                        attempt.record(('fixture',),{})
                        _run_variant_child(('fixture',),cwd=tmp,timeout_s=None,attempt=attempt)
            self.assertIs(caught.exception,original)
            self.assertTrue(any('secondary receipt failure' in note for note in original.__notes__))
            self.assertEqual((attempt.root/'stdout.txt').read_bytes(),b'completed log')
            self.assertEqual(json.loads((attempt.root/'run-lease.json').read_text())['status'],'active')

    def test_atomic_collision_retries_result_name_without_rerunning_child(self):
        from blender_blocking.utils.artifact_publication import publish_file_no_clobber
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp));collisions=[]
            def publish(stage,destination):
                if destination.name=='result.json' and not collisions:
                    destination.write_bytes(b'competing completed result');collisions.append(destination)
                    raise FileExistsError('competing publication')
                return publish_file_no_clobber(stage,destination)
            with patch.object(runner_module,'_variant_command',side_effect=self.child_command('success')) as command,patch('blender_blocking.utils.artifact_publication.publish_file_no_clobber',side_effect=publish):
                result=runner._run_one(case,variant,refs)
            command.assert_called_once()
            self.assertEqual(collisions[0].read_bytes(),b'competing completed result')
            self.assertEqual(result.result_json.name,'result_2.json')
            self.assert_joined_audit(result,'succeeded')

    def test_legacy_inprocess_signature_retry_keeps_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp),subprocess_mode=False);calls=[]
            def legacy(*args,**kwargs):
                calls.append(kwargs)
                if 'debug_output_dir' in kwargs:raise TypeError('legacy fixture signature')
                Path(kwargs['result_json']).write_text('{"passed":true}',encoding='utf-8')
                return True
            with patch('blender_blocking.test_e2e_validation.test_with_custom_images',side_effect=legacy):
                result=runner._run_one(case,variant,refs)
            self.assertEqual(len(calls),2)
            self.assertEqual(calls[0]['result_json'],calls[1]['result_json'])
            self.assertEqual(len(list((runner._case_variant_dir(case,variant)/'.variant-runs').glob('owned-*'))),1)
            self.assert_joined_audit(result,'succeeded')

    def test_optional_diagnostic_alias_failure_retains_authority_and_verdict(self):
        from blender_blocking.utils.artifact_publication import publish_file_no_clobber
        with tempfile.TemporaryDirectory() as tmp:
            runner,case,variant,refs=self.fixture(Path(tmp))
            def publish(stage,destination):
                if destination.name=='command.txt':raise PermissionError('optional alias locked')
                return publish_file_no_clobber(stage,destination)
            with patch.object(runner_module,'_variant_command',side_effect=self.child_command('success')),patch('blender_blocking.utils.artifact_publication.publish_file_no_clobber',side_effect=publish):
                result=runner._run_one(case,variant,refs)
            root=self.assert_joined_audit(result,'succeeded')
            self.assertEqual(result.status,'pass');self.assertEqual(result.avg_iou,.91)
            self.assertTrue((root/'command.txt').is_file())
            self.assertFalse((result.result_json.parent/'command.txt').exists())
            journal=json.loads((root/'attempt.json').read_text(encoding='utf-8'))
            self.assertTrue(any(p.get('status')=='publication_failed' and p.get('alias')=='command.txt' for p in journal['publications']))
            manifest=json.loads((root/'run-ownership.json').read_text(encoding='utf-8'))
            self.assertTrue(any('optional alias locked' in error for error in manifest['auxiliary_errors']))

    def test_timeout_option_rejects_invalid_values_without_changing_default(self):
        self.assertIsNone(RunOptions().variant_timeout_s)
        self.assertEqual(RunOptions(variant_timeout_s=.1).variant_timeout_s,.1)
        for value in (True,False,0,-1,float('nan'),float('inf'),'1'):
            with self.subTest(value=value),self.assertRaises(ValueError):RunOptions(variant_timeout_s=value)


if __name__=='__main__':unittest.main()
