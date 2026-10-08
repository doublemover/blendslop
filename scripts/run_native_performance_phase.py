"""Frozen, bounded performance comparisons. No installs or external dataset access."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASELINE="34de2d358ee8dd23a2ce5d8981a7ee4be802bea1"
VARIANTS=("legacy_unprofiled","instrumentation_only","projection_bypass","native_resident")
CASES=(("box",1234,"paired"),("vase",1234,"paired"),("torus",1234,"paired"),
       ("cylinder",77,"heldout"),("bottle",77,"heldout"),("chair",77,"heldout"))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def source_hashes():
    names=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard",
                                  "blender_blocking","scripts"],cwd=ROOT,text=True).splitlines()
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names if n.endswith('.py') and (ROOT/n).is_file()}


def own_memory():
    if os.name!='nt':return {"available":False}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
           'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    value=Counters();value.cb=ctypes.sizeof(value)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):return {'available':False}
    return {'working_set_bytes':int(value.WorkingSetSize),'peak_working_set_bytes':int(value.PeakWorkingSetSize),
            'scope':'process_lifetime_peak','source':'GetProcessMemoryInfo'}


def candidate_worker_arguments(args, destination):
    """Preserve the declared experiment at the actual reconstruction boundary."""
    return argparse.Namespace(worker='candidate', output=destination, case=args.case,
        mode='ensemble', arm='final', seed=args.seed,
        experiment=args.variant if args.config_overlay else None, config_overlay=args.config_overlay)


def worker(args):
    import bpy
    source=args.baseline if args.variant=='legacy_unprofiled' and not args.config_overlay else ROOT
    sys.path[:0]=[str(source),str(source/'blender_blocking')]
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    # Both historical import names occur in this project. Disable allocations in
    # the preserved source at runtime; source bytes stay unchanged and are pinned.
    for name in ('blender_blocking.evaluation.cost_model','evaluation.cost_model'):
        module=importlib.import_module(name)
        module.CostRecorder.__init__.__kwdefaults__['track_memory']=False
    import blender_blocking.config as config_module
    original=config_module.BlockingConfig
    def configured(*values,**keywords):
        cfg=original(*values,**keywords)
        if args.variant in VARIANTS and args.variant!='legacy_unprofiled':
            cfg.ensemble.native_resident=args.variant=='native_resident'
            cfg.ensemble.projection_diagnostics=args.variant=='instrumentation_only'
            cfg.ensemble.diagnostic_allocations=False
        return cfg
    config_module.BlockingConfig=configured
    spec=importlib.util.spec_from_file_location('performance_reference_harness',source/'scripts/run_improvement_pass.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.launched_at is not None:
        dump(args.output/'startup.json', {'launch_to_worker_ready_s':time.perf_counter()-args.launched_at,
             'boundary':'parent process launch through Blender startup and harness imports; reconstruction excluded'})
    rows=[]
    for repeat in range(args.repeats+1):
        # Warm means persistent interpreter/imports, with a fresh Blender scene
        # and fresh per-run geometry caches. It is not a cached-result benchmark.
        if repeat:bpy.ops.wm.read_factory_settings(use_empty=True)
        destination=args.output/('cold' if repeat==0 else f'warm-{repeat}')
        shutil.copytree(args.reference,destination/'references'/args.case)
        started=time.perf_counter()
        module.worker(candidate_worker_arguments(args, destination))
        receipt=destination/(args.variant if args.config_overlay else 'final')/args.case/'ensemble/comparison.json'
        row=json.loads(receipt.read_text())
        row.update(variant=args.variant,boundary='cold' if repeat==0 else 'warm',repeat=repeat,
                   process_memory=own_memory(),candidate_source_revision=BASELINE if args.variant=='legacy_unprofiled' and not args.config_overlay else args.revision,
                   runtime_override={'python_allocation_tracing':False},sample_outer_s=time.perf_counter()-started)
        rows.append(row);dump(args.output/'performance.json',{'rows':rows,'complete':repeat==args.repeats})


def campaign(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    baseline=output/'baseline-source'
    archive=subprocess.check_output(['git','archive','--format=zip',BASELINE],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:source.extractall(baseline)
    frozen=source_hashes()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'schema':'native_performance_campaign_v1','source_sha256':frozen,'baseline_revision':BASELINE,
              'candidate_revision':revision,'variants':VARIANTS,'warm_repeats':args.repeats,
              'cold_boundary':'first reconstruction in a fresh Blender process; process startup separately timed',
              'warm_boundary':'persistent interpreter/imports; fresh scene and candidate caches',
              'performance_claims':'unprofiled runs only; per-case median and range, no p95 with three repetitions',
              'budgets':{'threads':4,'performance_child_s':160,'campaign_s':args.budget},'jobs':[]}
    dump(output/'validation.json',manifest)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES'):env[key]=str(output/key.lower())
    started=time.perf_counter()
    def run(name,command,cap):
        if source_hashes()!=frozen:raise RuntimeError('source changed after freeze')
        if time.perf_counter()-started>=args.budget:raise RuntimeError('coordinated campaign budget exhausted')
        print('START '+name,flush=True);begin=time.perf_counter();log=output/(name+'.log')
        with log.open('w',encoding='utf-8') as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:status=process.wait(timeout=min(cap,args.budget-(begin-started)))
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                else:process.kill()
                process.wait();status='timeout_or_interrupted'
        manifest['jobs'].append({'name':name,'status':status,'wall_s':time.perf_counter()-begin,'command':command,'log':str(log)})
        manifest['source_unchanged']=source_hashes()==frozen;dump(output/'validation.json',manifest)
        print('FINISH '+name+': '+str(status),flush=True)
        if status!=0:raise RuntimeError('campaign job failed: '+name)
    blender=[args.blender,'--background','--factory-startup','--disable-autoexec','--threads','4','--python-exit-code','2','--python']
    previous=ROOT/'temp/improvement-phase-20261006/final-candidate'
    for index,(case,seed,split) in enumerate(CASES):
        # Rotate variant order across fixed cases to reduce order bias.
        for variant in VARIANTS[index%len(VARIANTS):]+VARIANTS[:index%len(VARIANTS)]:
            name=f'perf-{case}-{variant}'
            run(name,blender+[str(Path(__file__).resolve()),'--','--worker','--variant',variant,
                '--output',str(output/'performance'/case/variant),'--baseline',str(baseline),
                '--reference',str(previous/split/'references'/case),'--case',case,'--seed',str(seed),
                '--revision',revision,'--repeats',str(args.repeats)],160)
    for split,cases,seed in (('paired','box,vase,torus',1234),('heldout','cylinder,bottle,chair',77)):
        destination=output/split;shutil.copytree(previous/split/'references',destination/'references')
        shutil.copy2(previous/split/'final.json',destination/'previous-final.json')
        run('matrix-'+split,[sys.executable,str(ROOT/'scripts/run_improvement_pass.py'),'--blender',args.blender,
            '--output',str(destination),'--arm','final','--cases',cases,'--seed',str(seed),'--timeout','100'],1000)
    run('native-full',blender+[str(ROOT/'blender_blocking/test_runner.py'),'--'],240)
    run('saved-parts',blender+[str(ROOT/'scripts/verify_solid_usability.py'),'--','--output',str(output/'solids')],100)
    for scale,name in (('1','metres'),('.01','centimetres')):
        run('exports-'+name,blender+[str(ROOT/'scripts/verify_blender_exports.py'),'--',
            '--output',str(output/('exports-'+name)),'--scale-length',scale],100)
    run('candidate-exports',blender+[str(ROOT/'scripts/verify_candidate_exports.py'),'--',
        '--phase',str(output),'--output',str(output/'candidate-exports')],180)
    run('metrics-common',blender+[str(ROOT/'scripts/evaluate_protocol_campaign.py'),'--',
        '--phase',str(output),'--output',str(output/'metric-bundles')],180)
    run('native-prototypes',blender+[str(ROOT/'scripts/validate_native_prototypes.py'),'--',
        '--phase',str(output),'--output',str(output/'native-prototypes')],240)
    manifest['complete']=True;manifest['source_unchanged']=source_hashes()==frozen
    dump(output/'validation.json',manifest)


def load_campaign_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'blendslop_quality_campaign_v1':
        raise ValueError('unsupported campaign manifest')
    if manifest.get('source_tree_sha256'):
        actual = hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()
        if actual != manifest['source_tree_sha256']:
            raise ValueError('campaign source no longer matches the declared candidate freeze')
    cases, variants = manifest['cases'], manifest['variants']
    for rows in (cases, variants):
        names = [row['id'] for row in rows]
        if len(set(names)) != len(names) or any(not n.replace('_', '').replace('-', '').isalnum() for n in names):
            raise ValueError('campaign IDs must be unique simple names')
    required = {'front.png', 'side.png', 'top.png', 'ground_truth.obj', 'orbit45.png', 'manifest.json'}
    for case in cases:
        reference = ROOT / case['reference']
        if not all((reference / name).is_file() for name in required):
            raise ValueError('incomplete prepared reference: '+case['id'])
        prepared = json.loads((reference/'manifest.json').read_text(encoding='utf-8-sig'))
        if set(prepared['calibration']) != {'front', 'side', 'top'}:
            raise ValueError('three calibrated cameras required')
        from PIL import Image
        for name in ('front.png', 'side.png', 'top.png', 'orbit45.png'):
            with Image.open(reference/name) as image:
                if image.size != (512, 512):
                    raise ValueError('campaign requires prepared 512-square views: '+case['id']+':'+name)
        provenance = reference/'input-provenance.json'
        if provenance.is_file():
            required_case = required | {'input-provenance.json'}
        else:
            required_case = required
        hashes = {name: hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in sorted(required_case)}
        if case.get('reference_sha256') and case['reference_sha256'] != hashes:
            raise ValueError('prepared reference no longer matches the declared freeze: '+case['id'])
        case['reference_sha256'] = hashes
        for key, expected in (('source_path', case.get('source_sha256')), ('review_path', case.get('review_sha256'))):
            if case.get(key):
                actual = hashlib.sha256((ROOT/case[key]).read_bytes()).hexdigest()
                if expected != actual:
                    raise ValueError('source/review identity mismatch: '+case['id']+':'+key)
    native_hashes = {}
    def collect(values):
        if not isinstance(values, dict):
            return
        if values.get('dataset_dir') and values.get('scan') is not None:
            dataset = Path(values['dataset_dir'])
            dataset = dataset if dataset.is_absolute() else ROOT/dataset
            scan = int(values['scan'])
            for path in (dataset/'ObsMask'/f'ObsMask{scan}_10.mat', dataset/'ObsMask'/f'Plane{scan}.mat',
                         dataset/'Points'/'stl'/f'stl{scan:03}_total.ply'):
                if path.is_file():
                    native_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for key,value in values.items():
            if isinstance(value, dict):
                collect(value)
            elif key in {'reference','reference_fps','transform','prediction_transform','reference_metadata',
                         'assets','scale_matrix','culling','culling_receipt'} and isinstance(value,str):
                path = Path(value);path = path if path.is_absolute() else ROOT/path
                if path.is_file():
                    native_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    collect(manifest.get('protocols',{}))
    for case in cases:
        for name, digest in case['reference_sha256'].items():
            native_hashes[str(ROOT/case['reference']/name)] = digest
        for key in ('source_path', 'review_path'):
            if case.get(key):
                path = ROOT/case[key]
                native_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get('native_input_sha256') and manifest['native_input_sha256'] != native_hashes:
        raise ValueError('native/provenance inputs no longer match the declared freeze')
    manifest['native_input_sha256'] = native_hashes
    # Exercise typed validation without reconstructing or importing Blender APIs.
    sys.path[:0] = [str(ROOT), str(ROOT/'blender_blocking'), str(ROOT/'scripts')]
    from blender_blocking.config import BlockingConfig
    from run_improvement_pass import apply_config_overlay
    for variant in variants:
        apply_config_overlay(BlockingConfig(), variant['config_overlay'])
        if not set(variant['modes']) <= set(manifest['modes']):
            raise ValueError('unknown campaign mode')
    return manifest


def campaign_from_manifest(args):
    frozen_manifest = load_campaign_manifest(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = source_hashes()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    input_bytes = Path(args.manifest).read_bytes()
    record = {'schema': 'coordinated_quality_campaign_v1', 'candidate_revision': revision,
              'implementation_revision': frozen_manifest.get('code_revision', revision),
              'source_sha256': frozen, 'input_manifest_sha256': hashlib.sha256(input_bytes).hexdigest(),
              'manifest': frozen_manifest, 'jobs': [], 'rows': [], 'complete': False}
    dump(output/'validation.json', record)
    dump(output/'campaign-manifest.json', frozen_manifest)
    settings = frozen_manifest['budgets']
    started = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for key in ('BLENDER_USER_CONFIG', 'BLENDER_USER_SCRIPTS', 'BLENDER_USER_DATAFILES'):
        env[key] = str(output/key.lower())
    def run(name, command, cap):
        if source_hashes() != frozen or Path(args.manifest).read_bytes() != input_bytes:
            raise RuntimeError('campaign source/configuration changed after freeze')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in frozen_manifest.get('native_input_sha256',{}).items()):
            raise RuntimeError('native input changed after campaign freeze')
        remaining = settings['campaign_s']-(time.perf_counter()-started)
        if remaining <= 0:
            raise RuntimeError('campaign budget exhausted')
        print('START '+name, flush=True)
        status, wall_s = run_owned(command, output/(name+'.log'), min(cap, remaining), env)
        record['jobs'].append({'name': name, 'command': command, 'status': status, 'child_wall_s':wall_s})
        dump(output/'validation.json', record)
        print('FINISH '+name+': '+str(status), flush=True)
        return status
    blender = [args.blender, '--background', '--factory-startup', '--disable-autoexec', '--threads', '4', '--python-exit-code', '2', '--python']
    for variant in frozen_manifest['variants']:
        dump(output/'variant-configs'/(variant['id']+'.json'), variant['config_overlay'])
    for index, case in enumerate(frozen_manifest['cases']):
        destination = output/case['split']
        reference = destination/'references'/case['id']
        reference.mkdir(parents=True, exist_ok=False)
        for name in case['reference_sha256']:
            source = ROOT/case['reference']/name
            if hashlib.sha256(source.read_bytes()).hexdigest() != case['reference_sha256'][name]:
                raise ValueError('reference changed after freeze')
            shutil.copy2(source, reference/name)
        variants = frozen_manifest['variants']
        variants = variants[index%len(variants):]+variants[:index%len(variants)]
        for variant in variants:
            overlay = output/'variant-configs'/(variant['id']+'.json')
            for mode in variant['modes']:
                name = case['id']+'-'+variant['id']+'-'+mode
                if mode == 'ensemble':
                    perf = output/'performance'/case['id']/variant['id']
                    code = run(name, blender+[str(Path(__file__).resolve()), '--', '--worker',
                        '--variant', variant['id'], '--config-overlay', str(overlay), '--output', str(perf),
                        '--reference', str(reference), '--case', case['id'], '--seed', str(case['seed']),
                        '--revision', revision, '--repeats', str(settings['warm_repeats'])], settings['performance_child_s'])
                    receipt = perf/'cold'/variant['id']/case['id']/mode/'comparison.json'
                else:
                    code = run(name, blender+[str(ROOT/'scripts/run_improvement_pass.py'), '--', '--worker', 'candidate',
                        '--arm', 'final', '--experiment', variant['id'], '--config-overlay', str(overlay),
                        '--output', str(destination), '--case', case['id'], '--mode', mode, '--seed', str(case['seed'])],
                        settings['candidate_child_s'])
                    receipt = destination/variant['id']/case['id']/mode/'comparison.json'
                row = json.loads(receipt.read_text()) if code == 0 and receipt.exists() else {
                    'case': case['id'], 'requested_mode': mode, 'campaign_variant': variant['id'],
                    'seed': case['seed'], 'process_status': code, 'failure': 'candidate process failed'}
                row['split'] = case['split']
                record['rows'].append(row)
                dump(output/'campaign-rows.json', {'rows': record['rows']})
                dump(output/'validation.json', record)
    # Validation consumes the same completed/failure-preserving matrix, not a second reconstruction pass.
    jobs = [('native-full', ROOT/'blender_blocking/test_runner.py', [], 300),
            ('saved-parts', ROOT/'scripts/verify_solid_usability.py', ['--output', str(output/'solids')], 100),
            ('exports-metres', ROOT/'scripts/verify_blender_exports.py', ['--output', str(output/'exports-metres'), '--scale-length', '1'], 100),
            ('exports-centimetres', ROOT/'scripts/verify_blender_exports.py', ['--output', str(output/'exports-centimetres'), '--scale-length', '.01'], 100),
            ('candidate-exports', ROOT/'scripts/verify_candidate_exports.py', ['--phase', str(output), '--output', str(output/'candidate-exports')], 1800),
            ('metrics', ROOT/'scripts/evaluate_protocol_campaign.py', ['--phase', str(output), '--output', str(output/'metric-bundles'), '--manifest', str(output/'campaign-manifest.json')], 2400)]
    for name, script, arguments, cap in jobs:
        status = run(name, blender+[str(script), '--']+arguments, cap)
        if status != 0:
            record.setdefault('validation_failures', []).append(name)
    record['complete'] = True
    record['source_unchanged'] = source_hashes() == frozen
    dump(output/'validation.json', record)


def run_owned(command, log, timeout, env):
    launched = time.perf_counter()
    command = list(command)
    if '--worker' in command and '--variant' in command:
        command += ['--launched-at',str(launched)]
    with log.open('w', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            return child.wait(timeout=timeout), time.perf_counter()-launched
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            else:
                child.kill()
            child.wait()
            return 'timeout_or_interrupted', time.perf_counter()-launched


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blender');p.add_argument('--baseline',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--variant');p.add_argument('--config-overlay', type=Path);p.add_argument('--case');p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--launched-at', type=float)
    p.add_argument('--manifest', type=Path);p.add_argument('--validate-manifest-only', action='store_true')
    p.add_argument('--revision');p.add_argument('--repeats',type=int,default=3);p.add_argument('--budget',type=int,default=2400)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    if args.repeats not in (2,3):p.error('bounded campaign requires two or three warm repetitions')
    if args.validate_manifest_only:
        manifest = load_campaign_manifest(args.manifest)
        print(json.dumps({'cases': len(manifest['cases']), 'variants': [v['id'] for v in manifest['variants']],
                          'quality_rows': sum(len(v['modes']) for v in manifest['variants'])*len(manifest['cases'])}))
    elif args.worker:
        worker(args)
    elif args.manifest:
        campaign_from_manifest(args)
    else:
        campaign(args)


if __name__=='__main__':main()
