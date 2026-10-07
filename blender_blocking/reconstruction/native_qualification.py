"""Per-process, content/toolchain-scoped native Boolean admission receipts."""
from pathlib import Path
import hashlib, json, os, subprocess, tempfile, time
import numpy as np

_CACHE = {}
_IDENTITY_CACHE = {}
ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'scripts/qualify_native_solid.py'


def _file_identity_signature(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)


def _hash_identity_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def toolchain_identity(python, *, force=False):
    """Reuse one process-owned identity while every relevant file is unchanged.

    Geometry receipts remain separately keyed by exact coordinate/connectivity
    hashes. Changed path targets, inode, content metadata, producer/predicate/helper
    source or Blender build invalidate this identity; force performs a full rehash.
    """
    import bpy
    executable = Path(python).resolve(strict=True)
    site = executable.parents[1]/'Lib/site-packages'
    native_files = list(site.glob('open3d-*.dist-info/METADATA'))+list(site.glob('open3d/cpu/pybind*.pyd'))
    if len(native_files) != 2:
        raise ValueError('existing Open3D toolchain identity cannot be established')
    native_files += list(site.glob('numpy-*.dist-info/METADATA'))
    environment = executable.parents[1]/'pyvenv.cfg'
    if environment.is_file():
        native_files.append(environment)
    predicate = ROOT/'blender_blocking/evaluation/triangle_contacts.py'
    producer = Path(__file__).resolve()
    paths = [*native_files, executable, HELPER, predicate, producer]
    build = (bpy.app.version_string, bpy.app.build_hash.decode())
    key = (str(executable), build)
    signature = tuple(_file_identity_signature(path) for path in paths)
    cached = _IDENTITY_CACHE.get(key)
    if not force and cached is not None and cached[0] == signature:
        return cached[1]
    details = {'open3d_files': {str(path):_hash_identity_file(path) for path in native_files},
        'blender': build[0], 'build': build[1], 'python_path':str(executable),
        'python_sha256':_hash_identity_file(executable),
        'helper_sha256':_hash_identity_file(HELPER),
        'triangle_predicate_sha256':_hash_identity_file(predicate),
        'producer_sha256':_hash_identity_file(producer)}
    # A file changed during hashing is unavailable, never a reusable certificate.
    if signature != tuple(_file_identity_signature(path) for path in paths):
        raise ValueError('toolchain identity files changed during verification')
    identity = hashlib.sha256(json.dumps(details,sort_keys=True).encode()).hexdigest()
    _IDENTITY_CACHE[key] = (signature, identity)
    return identity


def qualify_geometry(data, *, python, timeout_s=15.):
    from .grouped_solids import solid_guard
    started = time.perf_counter()
    guard = solid_guard(data)

    def unavailable(reason, error=None, identity=None):
        return {'geometry_content_hash': data.content_hash, 'toolchain_identity': identity,
            'manifold_validated': False, 'single_solid_qualified': False, 'status': 'unavailable',
            'reason': reason, 'error': error, 'triangle_count': len(data.faces), 'solid_guard': guard,
            'qualification_elapsed_s': time.perf_counter()-started, 'helper_timeout_s': timeout_s,
            'timeout_scope': 'helper process wait; identity and transfer cost reported separately'}

    if not guard['valid_solid']:
        return unavailable('topology_or_volume_guard_failed')
    if len(data.faces) > 60000:
        return unavailable('triangle_limit_exceeded')
    try:
        identity = toolchain_identity(python)
    except Exception as exc:
        return unavailable('toolchain_identity_unavailable', str(exc))
    key = (data.content_hash, identity)
    if key in _CACHE:
        return {**_CACHE[key], 'cache_hit': True, 'qualification_elapsed_s': time.perf_counter()-started,
                'qualification_cost_boundary': 'identity verification and cache lookup; no helper execution'}
    root = ROOT / 'temp/native-qualification'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='owned-', dir=root) as directory:
        source, destination = Path(directory)/'input.npz', Path(directory)/'receipt.json'
        np.savez(source, vertices=data.vertices, faces=data.faces, content_hash=data.content_hash)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        child = subprocess.Popen([str(python), '-B', str(HELPER), '--input', str(source),
            '--output', str(destination), '--identity', identity], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            log, _ = child.communicate(timeout=max(.001, timeout_s))
        except subprocess.TimeoutExpired:
            child.kill(); child.communicate()
            return unavailable('helper_timeout', identity=identity)
        except BaseException:
            child.kill(); child.communicate()
            raise
        if child.returncode:
            return unavailable('helper_failed', log.decode(errors='replace')[-2000:], identity)
        receipt = json.loads(destination.read_text(encoding='utf-8'))
    if receipt['geometry_content_hash'] != data.content_hash or receipt['toolchain_identity'] != identity:
        raise ValueError('native qualification receipt identity mismatch')
    verification = receipt.get('exact_pair_verification', {})
    incomplete = not verification.get('complete', True) and not verification.get('non_disjoint_pairs', 0)
    receipt.update(status='qualified' if receipt['manifold_validated'] else 'unavailable' if incomplete else 'rejected',
                   reason=None if receipt['manifold_validated'] else 'narrow_phase_incomplete' if incomplete else 'boundary_qualification_failed',
                   single_solid_qualified=bool(receipt['manifold_validated'] and guard['connected_components'] == 1),
                   solid_guard=guard, cache_hit=False, qualification_elapsed_s=time.perf_counter()-started,
                   qualification_cost_boundary='identity verification, solid guard, transfer and actual helper execution')
    _CACHE[key] = receipt
    return receipt
