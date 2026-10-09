"""Explicit optional CPU helper bridge: fixed files, own capped child, no installs."""
from __future__ import annotations
from pathlib import Path
import json,os,subprocess,sys,time,uuid
import numpy as np


WORKER = Path(__file__).resolve().parents[4]/'scripts'/'open3d_cpu_worker.py'


def _owned_limits(config):
    import math
    owned = config.get('poisson_owned_supervision', False)
    if not isinstance(owned, bool):
        raise ValueError('poisson_owned_supervision must be a boolean')
    if not owned:
        return None
    limits = []
    for name, maximum in (('poisson_committed_limit_bytes', 8*1024**3),
                          ('poisson_rss_limit_bytes', 8*1024**3),
                          ('poisson_transport_limit_bytes', 256*1024**2)):
        value = config.get(name)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(name + ' must be an explicit bounded positive integer')
        limits.append(value)
    timeout = config.get('poisson_timeout_s', 90.)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 90):
        raise ValueError('owned Poisson helper timeout must be finite >0..90 seconds')
    return (*limits, float(timeout))


def _transport_module():
    # The worker owns the small fixed NPZ protocol; importing it never runs Open3D.
    import importlib.util
    spec = importlib.util.spec_from_file_location('owned_poisson_transport', WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_owned_poisson(mesh_result, method, config, limits):
    from volume import MeshExtractionResult
    from utils.run_ownership import OwnedRun
    from utils.owned_process_supervisor import run_bounded_process
    transport = _transport_module()
    committed, rss, cap, timeout = limits
    python = Path(os.path.abspath(config['external_open3d_python']))
    if not python.is_file():
        raise ValueError('configured Open3D helper Python is missing')
    arrays = {'vertices': np.asarray(mesh_result.vertices), 'faces': np.asarray(mesh_result.faces)}
    input_bound = transport._validate_arrays(arrays) + 16384
    # The supervisor's log limit is sampled. Reserve its declared4MiB plus
    # receipt/config space; never describe this as an exact unsampled disk quota.
    overhead = 4194304 + 1048576 + 262144
    if input_bound + overhead + 24576 >= cap:
        raise ValueError('owned Poisson transport budget cannot hold input/output/log receipts')
    worker_sha = transport._file_sha(WORKER, 8388608)
    parent = Path(os.path.abspath(config['postprocess_artifact_root']))
    owner = OwnedRun(parent, producer='owned_external_poisson', max_generated_bytes=cap,
                     shared_inputs={'interpreter': str(python), 'worker': str(WORKER),
                                    'worker_sha256': worker_sha})
    supervision, error, attempted = None, None, False
    started = time.perf_counter()
    summary = {'protocol': 'owned_external_poisson_v1', 'status': 'preparing',
               'transport_limit_bytes': cap, 'scope': 'fresh helper transport only; all files retained',
               'log_bound_scope': 'supervisor4MiB sampled limit; inter-poll overshoot possible',
               'lifecycle_complete': False}
    known = ('input.npz', 'config.json', 'command.json', 'worker.log', 'output.npz',
             'result.json', 'supervision.json', 'bridge-result.json')
    try:
        owner.reserve_bytes(input_bound)
        input_sha = transport._write_owned_npz(owner.root/'input.npz', arrays, cap - overhead)
        owner.register_file('input.npz', 'disposable')
        owner.reserve_bytes(1048576)
        config_sha = transport._write_owned_json(owner.root/'config.json', dict(config), 1048576)
        owner.register_file('config.json', 'diagnostic')
        output_cap = cap - sum(row['bytes'] for row in owner.records.values()) - overhead
        if output_cap <= 24576:
            raise ValueError('owned Poisson output allowance exhausted before launch')
        command = [str(python), str(WORKER), '--root', str(owner.root), '--method', method,
            '--owned-transport-limit-bytes', str(cap), '--owned-output-limit-bytes', str(output_cap),
            '--expected-input-sha256', input_sha,
            '--expected-config-sha256', config_sha, '--expected-worker-sha256', worker_sha]
        transport._write_owned_json(owner.root/'command.json', {
            'argv': command, 'timeout_s': timeout, 'join_timeout_s': 5.,
            'committed_limit_bytes': committed, 'rss_limit_bytes': rss,
            'input_sha256': input_sha, 'config_sha256': config_sha, 'worker_sha256': worker_sha}, 65536)
        owner.register_file('command.json', 'diagnostic')
        env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        attempted = True
        supervision = run_bounded_process(command, log_path=owner.root/'worker.log', timeout_s=timeout,
            join_timeout_s=5., max_memory_bytes=committed, max_rss_bytes=rss,
            require_complete_tree=True, env=env)
        if (supervision.get('lifecycle_complete') is not True or type(supervision.get('returncode')) is not int
                or supervision.get('returncode') != 0
                or supervision.get('status') != 'succeeded'):
            raise RuntimeError('owned Open3D helper failed or complete tree joins unavailable: '
                               + str(supervision.get('limit_reason') or supervision.get('error') or supervision.get('returncode')))
        metadata, _ = transport._read_owned_json(owner.root/'result.json', 1048576)
        if (metadata.get('source_sha256') != input_sha or metadata.get('config_sha256') != config_sha
                or metadata.get('worker_sha256') != worker_sha
                or metadata.get('owned_transport_protocol') != 'bounded_poisson_npz_v1'
                or not isinstance(metadata.get('metrics'), dict) or not isinstance(metadata.get('topology'), dict)):
            raise ValueError('owned Poisson result input/config/worker identity differs')
        output, output_sha = transport._read_owned_npz(owner.root/'output.npz', output_cap, output=True)
        if (metadata.get('output_sha256') != output_sha
                or transport._file_sha(owner.root/'input.npz', cap) != input_sha
                or transport._file_sha(owner.root/'config.json', 1048576) != config_sha
                or transport._file_sha(WORKER, 8388608) != worker_sha):
            raise ValueError('owned Poisson result/source bytes changed or output identity differs')
        summary.update(status='succeeded', output_sha256=output_sha)
        return MeshExtractionResult(status='ok', method='screened_poisson_external_open3d',
            requested_method=mesh_result.method, vertices=output['vertices'], faces=output['faces'],
            normals=output['normals'] if len(output['normals']) else None, topology=metadata['topology'],
            metrics={**metadata['metrics'], 'external_helper': metadata,
                'external_wall_s': time.perf_counter()-started, 'bridge_artifacts': str(owner.root),
                'poisson_ownership_receipt': str(owner.root/'run-ownership.json'),
                'poisson_process_receipt': str(owner.root/'supervision.json')},
            message='explicit configured Open3D CPU helper completed')
    except BaseException as exc:
        error = exc
        if supervision is None:
            supervision = getattr(exc, 'owned_process_receipt', None)
        owner.mark_failed(repr(exc))
        try:
            exc.poisson_ownership_receipt = owner.root/'run-ownership.json'
            if supervision is not None:
                exc.poisson_process_receipt = owner.root/'supervision.json'
        except BaseException:
            pass
        raise
    finally:
        auxiliary = []
        complete = not attempted or (isinstance(supervision, dict) and supervision.get('lifecycle_complete') is True)
        summary.update(status=('cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit))
                               else 'failed' if error else summary['status']),
                       lifecycle_complete=complete, launch_attempted=attempted,
                       error=None if error is None else repr(error), elapsed_s=time.perf_counter()-started)
        if supervision is not None:
            try:
                transport._write_owned_json(owner.root/'supervision.json', supervision, 131072)
            except BaseException as secondary:
                auxiliary.append(secondary)
        try:
            transport._write_owned_json(owner.root/'bridge-result.json', summary, 65536)
        except BaseException as secondary:
            auxiliary.append(secondary)
        for name in known:
            try:
                if (owner.root/name).is_file():
                    owner.register_file(name, 'disposable' if name == 'input.npz' else 'diagnostic')
            except BaseException as secondary:
                auxiliary.append(secondary)
        owner.auxiliary_errors.extend(repr(exc) for exc in auxiliary)
        if complete:
            try:
                owner.close(error=error or (auxiliary[0] if auxiliary else None))
            except BaseException as secondary:
                auxiliary.append(secondary)
        if error is not None:
            for secondary in auxiliary:
                error.add_note('Owned Poisson receipt also failed: ' + repr(secondary))
        elif auxiliary:
            secondary = auxiliary[0]
            secondary.poisson_ownership_receipt = owner.root/'run-ownership.json'
            if supervision is not None:
                secondary.poisson_process_receipt = owner.root/'supervision.json'
            raise secondary


def run_external_poisson(mesh_result,method,config):
    limits = _owned_limits(config)
    if limits is not None:
        return _run_owned_poisson(mesh_result, method, config, limits)
    from volume import MeshExtractionResult
    python=Path(config['external_open3d_python']).resolve()
    if not python.is_file(): raise ValueError('configured Open3D helper Python is missing')
    root=Path(config['postprocess_artifact_root']).resolve()/('cpu-'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True,exist_ok=False)
    np.savez(root/'input.npz',vertices=mesh_result.vertices,faces=mesh_result.faces)
    (root/'config.json').write_text(json.dumps(dict(config),default=str))
    script=Path(__file__).resolve().parents[4]/'scripts'/'open3d_cpu_worker.py'
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    started=time.perf_counter()
    with (root/'worker.log').open('w') as log:
        child=subprocess.Popen([str(python),str(script),'--root',str(root),'--method',method],stdout=log,stderr=subprocess.STDOUT,
            env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try: code=child.wait(timeout=min(90.,float(config.get('poisson_timeout_s',90.))))
        except BaseException as original:
            try:
                child.kill()
                child.wait(timeout=5.)
            except BaseException as cleanup_error:
                original.add_note('Open3D primary cleanup remains unconfirmed: ' + repr(cleanup_error))
            if isinstance(original, subprocess.TimeoutExpired):
                raise RuntimeError('Open3D child exceeded explicit timeout; log retained') from original
            raise
    if code!=0: raise RuntimeError(f'Open3D helper failed ({code}); log: {root / "worker.log"}')
    arrays=np.load(root/'output.npz',allow_pickle=False)
    metadata=json.loads((root/'result.json').read_text())
    return MeshExtractionResult(status='ok',method='screened_poisson_external_open3d',requested_method=mesh_result.method,
        vertices=arrays['vertices'],faces=arrays['faces'],normals=arrays['normals'],topology=metadata['topology'],
        metrics={**metadata['metrics'],'external_helper':metadata,'external_wall_s':time.perf_counter()-started,'bridge_artifacts':str(root)},
        message='explicit configured Open3D CPU helper completed')
