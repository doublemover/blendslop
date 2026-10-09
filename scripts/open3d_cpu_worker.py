"""Poisson worker for an explicitly configured, already installed helper Python."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]


MAX_TRANSPORT_BYTES = 256 * 1024 ** 2


def _transport_limit(value):
    if type(value) is not int or not 0 < value <= MAX_TRANSPORT_BYTES:
        raise ValueError('owned Poisson transport limit must be a positive integer <=256MiB')
    return value


def _bounded_digest(stream, limit):
    digest = hashlib.sha256()
    stream.seek(0)
    used = 0
    while True:
        block = stream.read(min(1048576, limit - used + 1))
        used += len(block)
        if used > limit:
            raise ValueError('owned Poisson file exceeds its byte bound')
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _file_sha(path, limit):
    with Path(path).open('rb') as stream:
        return _bounded_digest(stream, limit)


def _validate_arrays(arrays, *, output=False):
    required = {'vertices', 'faces', 'normals'} if output else {'vertices', 'faces'}
    if set(arrays) != required:
        raise ValueError('owned Poisson array inventory differs')
    vertices, faces = arrays['vertices'], arrays['faces']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or vertices.dtype.kind != 'f' or vertices.dtype.itemsize not in (4, 8)
            or not np.isfinite(vertices).all()):
        raise ValueError('owned Poisson vertices must be finite floating Nx3')
    if (faces.ndim != 2 or not 3 <= faces.shape[1] <= 64
            or (output and (faces.shape[1] != 3 or not len(faces)))
            or faces.dtype.kind not in 'iu' or faces.dtype.itemsize not in (4, 8)
            or (len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError('owned Poisson faces have invalid integer shape or indices')
    if output:
        normals = arrays['normals']
        if (normals.shape not in (vertices.shape, (0, 3)) or normals.dtype.kind != 'f'
                or normals.dtype.itemsize not in (4, 8) or not np.isfinite(normals).all()):
            raise ValueError('owned Poisson normals must be finite floating Nx3 or empty')
    return sum(array.nbytes for array in arrays.values())


def _read_owned_npz(path, limit, *, output=False):
    """Bound uncompressed member headers before numpy can allocate any array."""
    import math, os, zipfile
    limit = _transport_limit(limit)
    names = {'vertices.npy', 'faces.npy', 'normals.npy'} if output else {'vertices.npy', 'faces.npy'}
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        named_before = Path(path).stat()
        digest = _bounded_digest(stream, limit)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if len(entries) != len(names) or {item.filename for item in entries} != names:
                raise ValueError('owned Poisson NPZ member inventory differs')
            if sum(item.file_size for item in entries) > limit:
                raise ValueError('owned Poisson uncompressed NPZ exceeds byte bound')
            for item in entries:
                with archive.open(item) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=4096)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=4096)
                    else:
                        raise ValueError('owned Poisson NPY header version unsupported')
                    if (len(shape) != 2 or any(type(n) is not int or n < 0 for n in shape)
                            or dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
                            or dtype.kind not in 'fiu' or dtype.itemsize not in (4, 8)
                            or math.prod(shape) * dtype.itemsize != item.file_size - member.tell()):
                        raise ValueError('owned Poisson NPY header/dtype/data extent invalid')
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            arrays = {name[:-4]: archive[name[:-4]] for name in names}
        _validate_arrays(arrays, output=output)
        if _bounded_digest(stream, limit) != digest:
            raise ValueError('owned Poisson NPZ changed during read')
        current, named = os.fstat(stream.fileno()), Path(path).stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        # Windows path stat and CRT fstat can expose different ctime meanings.
        # Compare each API against its own snapshot, and bind the open file to
        # the pathname by actual filesystem/device/inode/size identity.
        if (signature(before) != signature(current) or signature(named_before) != signature(named)
                or (current.st_dev, current.st_ino, current.st_size) != (named.st_dev, named.st_ino, named.st_size)):
            raise ValueError('owned Poisson NPZ identity changed during read')
    return arrays, digest


class _CappedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit
    def write(self, block):
        if self.stream.tell() + len(block) > self.limit:
            raise ValueError('owned Poisson serialization exceeds byte bound')
        return self.stream.write(block)
    def read(self, *args):
        # numpy recognizes file objects by their read attribute; this writer
        # is used only for output, and the underlying stream forbids reads.
        return self.stream.read(*args)
    def tell(self):
        return self.stream.tell()
    def seek(self, *args):
        return self.stream.seek(*args)
    def flush(self):
        return self.stream.flush()


def _write_owned_npz(path, arrays, limit, *, output=False):
    limit = _transport_limit(limit)
    if _validate_arrays(arrays, output=output) + 8192 * len(arrays) > limit:
        raise ValueError('owned Poisson numeric transport exceeds byte bound before write')
    with Path(path).open('xb') as stream:
        np.savez(_CappedWriter(stream, limit), **arrays)
    return _file_sha(path, limit)


def _read_owned_json(path, limit):
    limit = min(_transport_limit(limit), 1048576)
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('owned Poisson JSON exceeds byte bound')
    def reject_nonfinite(value):
        raise ValueError('owned Poisson JSON contains nonfinite numeric values')
    value = json.loads(data, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError('owned Poisson JSON must be an object')
    return value, hashlib.sha256(data).hexdigest()


def _write_owned_json(path, value, limit):
    limit = min(_transport_limit(limit), 1048576)
    with Path(path).open('xb') as stream:
        writer = _CappedWriter(stream, limit)
        for block in json.JSONEncoder(indent=2, allow_nan=False).iterencode(value):
            writer.write(block.encode('utf-8'))
    return _file_sha(path, limit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',default='screened_poisson')
    p.add_argument('--owned-transport-limit-bytes',type=int);p.add_argument('--owned-output-limit-bytes',type=int)
    p.add_argument('--expected-input-sha256');p.add_argument('--expected-config-sha256');p.add_argument('--expected-worker-sha256')
    args=p.parse_args()
    owned = args.owned_transport_limit_bytes is not None
    if owned:
        limit = _transport_limit(args.owned_transport_limit_bytes)
        output_limit = _transport_limit(args.owned_output_limit_bytes)
        if output_limit > limit:
            raise ValueError('owned Poisson output limit exceeds transport limit')
        data, input_sha = _read_owned_npz(args.root/'input.npz', limit)
        config, config_sha = _read_owned_json(args.root/'config.json', limit)
        worker_sha = _file_sha(Path(__file__), 8388608)
        if (input_sha != args.expected_input_sha256 or config_sha != args.expected_config_sha256
                or worker_sha != args.expected_worker_sha256):
            raise ValueError('owned Poisson frozen input/config/worker identity differs')
    else:
        data=np.load(args.root/'input.npz',allow_pickle=False)
        config=json.loads((args.root/'config.json').read_text())
    import open3d as o3d
    from volume import MeshExtractionResult
    from reconstruction.backends.visual_hull.poisson import _run_open3d_poisson
    source=MeshExtractionResult(status='ok',method='matched_source_mesh',vertices=data['vertices'],faces=data['faces'])
    started=time.perf_counter();result=_run_open3d_poisson(source,args.method,config)
    if not owned:
        np.savez(args.root/'output.npz',vertices=result.vertices,faces=result.faces,normals=result.normals)
    metadata={'open3d':o3d.__version__,'python':sys.version,'binary':sys.executable,
        'source_sha256':_file_sha(args.root/'input.npz', limit) if owned else hashlib.sha256((args.root/'input.npz').read_bytes()).hexdigest(),
        'config_sha256':_file_sha(args.root/'config.json', 1048576) if owned else hashlib.sha256((args.root/'config.json').read_bytes()).hexdigest(),
        'worker_sha256':_file_sha(Path(__file__), 8388608) if owned else hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'numpy':np.__version__,'kernel_wall_s':time.perf_counter()-started,
        'metrics':result.metrics,'topology':result.topology}
    if owned:
        if (metadata['source_sha256'] != input_sha or metadata['config_sha256'] != config_sha
                or metadata['worker_sha256'] != worker_sha):
            raise ValueError('owned Poisson source bytes changed during kernel execution')
        normals = result.normals if result.normals is not None else np.empty((0,3), np.float64)
        metadata['output_sha256'] = _write_owned_npz(args.root/'output.npz',
            {'vertices':result.vertices,'faces':result.faces,'normals':normals}, output_limit, output=True)
        metadata['owned_transport_protocol'] = 'bounded_poisson_npz_v1'
        _write_owned_json(args.root/'result.json', metadata, limit)
    else:
        (args.root/'result.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
