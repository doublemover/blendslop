"""Owned signed-field worker; trusted local pickle packets, no GUI imports."""
import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'blendslop_owned_implicit'
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(ROOT/'blender_blocking/reconstruction/implicit')]
sys.modules[PACKAGE] = namespace
solver = importlib.import_module(PACKAGE+'.numeric_solver')


def dependency_state():
    try:
        import torch
        available = (torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None
                     and torch.version.hip is None)
        from importlib.metadata import version, PackageNotFoundError
        optional = {}
        for name in ('dvx-python', 'scikit-image', 'shapely'):
            try:
                optional[name] = version(name)
            except PackageNotFoundError:
                optional[name] = None
        return {'available': available, 'torch': str(torch.__version__), 'device': 'cpu',
                'operators': ['minimum-distance ray proposal', 'optional extracted opaque original-pixel box projection'],
                'original_pixel_mesh_available': (available and optional['dvx-python'] == '0.1.1'
                    and optional['scikit-image'] == '0.26.0' and optional['shapely'] == '2.1.2'),
                'optional_dependencies': optional, 'native_qualification': False}
    except ImportError as exc:
        return {'available': False, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--directory', required=True)
    parser.add_argument('--idle-timeout', type=float, default=60.)
    args = parser.parse_args()
    directory = Path(args.directory)
    temporary = directory/'ready.tmp'
    temporary.write_text(json.dumps({**dependency_state(), 'pid': os.getpid()}))
    os.replace(temporary, directory/'ready.json')
    while not (directory/'stop').exists():
        heartbeat = directory/'heartbeat'
        if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime > args.idle_timeout:
            break
        jobs = sorted(directory.glob('*.input.pkl'))
        if not jobs:
            time.sleep(.01)
            continue
        for source in jobs:
            job = source.name[:-len('.input.pkl')]
            try:
                with source.open('rb') as stream:
                    value = solver.fit_implicit_field_job(pickle.load(stream))
                response = {'job': job, 'status': 'success', 'value': value}
            except Exception:
                response = {'job': job, 'status': 'failed', 'error': traceback.format_exc()}
            source.unlink(missing_ok=True)
            destination = directory/(job+'.output.pkl')
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(response, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, destination)


if __name__ == '__main__':
    main()
