"""Per-process, content/toolchain-scoped native Boolean admission receipts."""
from pathlib import Path
import hashlib, json, os, subprocess, time
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


def qualify_geometry(data, *, python, timeout_s=15., ownership_root=None):
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
    from utils.run_ownership import OwnedRun
    from utils.primary_process_cleanup import finish_primary_process
    from contextlib import contextmanager
    owner = OwnedRun(ownership_root or ROOT / 'temp/native-qualification',
                     producer="native_boundary_qualification",
                     shared_inputs={"qualification_interpreter": str(python),
                                    "toolchain_identity": identity,
                                    "input_geometry_hash": data.content_hash})
    child = None
    cleanup_receipt = None
    pipes_drained = False

    @contextmanager
    def owned_helper_run():
        nonlocal cleanup_receipt
        error = None
        try:
            yield
        except BaseException as original:
            error = original
            raise
        finally:
            auxiliary = []
            if child is not None and cleanup_receipt is None:
                try:
                    _, _, cleanup_receipt = finish_primary_process(child, drain_pipes=not pipes_drained)
                except BaseException as join_error:
                    cleanup_receipt = getattr(join_error, "primary_process_receipt", None)
                    auxiliary.append(join_error)
            joined = child is None or bool(cleanup_receipt and cleanup_receipt["transport_closed"])
            if not joined or (cleanup_receipt and cleanup_receipt["errors"]):
                detail = "; ".join(cleanup_receipt["errors"]) if cleanup_receipt else "cleanup receipt unavailable"
                auxiliary.insert(0, RuntimeError("Native helper cleanup remains unconfirmed: " + detail))
            try:
                if cleanup_receipt is not None:
                    (owner.root / "helper-cleanup.json").write_text(
                        json.dumps(cleanup_receipt, indent=2), encoding="utf-8")
                    owner.register_file("helper-cleanup.json", "diagnostic")
                failure = error or (auxiliary[0] if auxiliary else None)
                if failure is not None:
                    owner.mark_failed(repr(failure))
                if joined:
                    owner.close(error=failure)
                else:
                    # Keep the original fresh lease active; no historical adoption.
                    owner._write_metadata("active")
            except Exception as publication_error:
                auxiliary.append(publication_error)
            if error is not None:
                for secondary in auxiliary:
                    if hasattr(error, "add_note"):
                        error.add_note("Native helper lifecycle also failed: " + repr(secondary))
            elif auxiliary:
                raise auxiliary[0]

    with owned_helper_run():
        owner.reserve_bytes(data.nbytes + 4096 + 8192 + 65536 + 8192)
        source, destination = owner.root / 'input.npz', owner.root / 'receipt.json'
        np.savez(source, vertices=data.vertices, faces=data.faces, content_hash=data.content_hash)
        owner.register_file("input.npz", "disposable")
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        child = subprocess.Popen([str(python), '-B', str(HELPER), '--input', str(source),
            '--output', str(destination), '--identity', identity], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        timed_out = False
        try:
            log, _ = child.communicate(timeout=max(.001, timeout_s))
            pipes_drained = True
        except subprocess.TimeoutExpired:
            log, _, cleanup_receipt = finish_primary_process(child)
            log = log or b""
            timed_out = True
        (owner.root / "helper-log-tail.txt").write_bytes(log[-8192:])
        owner.register_file("helper-log-tail.txt", "diagnostic")
        if timed_out or child.returncode:
            reason = 'helper_timeout' if timed_out else 'helper_failed'
            owner.mark_failed(reason)
            receipt = {**unavailable(reason, log.decode(errors='replace')[-2000:] or None, identity),
                       "owned_run_root": str(owner.root)}
            destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            owner.register_file("receipt.json", "diagnostic")
            return receipt
        if destination.stat().st_size > 65536:
            raise ValueError('native qualification receipt exceeds bounded size')
        receipt = json.loads(destination.read_text(encoding='utf-8'))
        # Validate before releasing the owner lease; malformed/stale helper output
        # is a failed producer run, never a successfully completed ownership receipt.
        if receipt['geometry_content_hash'] != data.content_hash or receipt['toolchain_identity'] != identity:
            owner.register_file("receipt.json", "diagnostic")
            raise ValueError('native qualification receipt identity mismatch')
        verification = receipt.get('exact_pair_verification', {})
        incomplete = not verification.get('complete', True) and not verification.get('non_disjoint_pairs', 0)
        receipt.update(status='qualified' if receipt['manifold_validated'] else 'unavailable' if incomplete else 'rejected',
                       reason=None if receipt['manifold_validated'] else 'narrow_phase_incomplete' if incomplete else 'boundary_qualification_failed',
                       single_solid_qualified=bool(receipt['manifold_validated'] and guard['connected_components'] == 1),
                       solid_guard=guard, cache_hit=False, qualification_elapsed_s=time.perf_counter()-started,
                       owned_run_root=str(owner.root),
                       qualification_cost_boundary='identity verification, solid guard, transfer and actual helper execution')
        if not receipt['manifold_validated']:
            owner.mark_failed(receipt['reason'])
        destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        owner.register_file("receipt.json", "diagnostic")
    _CACHE[key] = receipt
    return receipt
