#!/usr/bin/env python3
"""Qualify selected saved actual family boundaries without fitting or rendering."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'blender_blocking')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-run', type=Path, required=True)
    parser.add_argument('--families', nargs='+', required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=float, default=150.)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not 1 <= len(args.families) <= 7 or len(set(args.families)) != len(args.families):
        parser.error('select one to seven distinct saved family geometries')
    if not 0 < args.deadline_seconds <= 150:
        parser.error('outer deadline must be positive and at most150 seconds')
    return args


def main():
    import numpy as np
    import test_runner  # Expose the already installed Blender dependency paths.
    from reconstruction.native_geometry import GeometryArrays
    from reconstruction.native_qualification import qualify_geometry, toolchain_identity
    from utils.run_ownership import OwnedRun

    args = arguments()
    source = args.prepared_run.resolve(strict=True)
    parent = args.output.resolve()
    if not parent.is_relative_to(ROOT / 'temp/tasks'):
        raise ValueError('output must stay below the isolated project temp/tasks')
    prepared = json.loads(source.read_text(encoding='utf8'))
    frozen = {}
    for family in args.families:
        row = prepared['cases'][family]
        if row.get('silhouette', {}).get('status') != 'passed':
            raise ValueError('selected row requires its completed strict five-view silhouettes: ' + family)
        folder = Path(row['artifact_directory']).resolve(strict=True)
        archive = folder / 'evaluated-exact.npz'
        if archive.stat().st_size > 67108864 or digest(archive) != row['npz_sha256']:
            raise ValueError('saved geometry archive identity/bound changed: ' + family)
        frozen[family] = {'archive': str(archive), 'archive_sha256': row['npz_sha256'],
                          'geometry_hash': row['geometry_hash']}
    identity = toolchain_identity(args.python)
    owner = OwnedRun(parent, producer='saved_family_boundary_qualification',
                     shared_inputs={'prepared_run': str(source), 'prepared_sha256': digest(source),
                                    'python': str(args.python), 'toolchain_identity': identity})
    # Helper roots stay separately owned; this coordinator never claims their
    # transport files or child leases as its own disposable artifacts.
    children = parent / 'qualification-children'
    started = time.monotonic()
    with owner:
        workload = {'protocol': 'saved_actual_family_boundary_v1', 'families': frozen,
                    'prepared_sha256': digest(source), 'toolchain_identity': identity,
                    'helper_timeout_seconds': 15., 'deadline_seconds': args.deadline_seconds,
                    'fits': 0, 'renders': 0, 'surface_recomputations': 0}
        (owner.root / 'frozen-workload.json').write_text(json.dumps(workload, indent=2), encoding='utf8')
        owner.register_file('frozen-workload.json', 'diagnostic')
        receipt = {'protocol': workload['protocol'], 'cases': {}, 'aggregate_accepted': False,
                   'scope': 'actual boundary only; silhouette, surface and semantic edits remain independent'}
        for family, inputs in frozen.items():
            if args.deadline_seconds - (time.monotonic() - started) < 20.:
                receipt['cases'][family] = {'status': 'unavailable', 'reason': 'outer deadline before row'}
                continue
            archive = Path(inputs['archive'])
            if digest(archive) != inputs['archive_sha256']:
                raise ValueError('frozen input changed before native qualification: ' + family)
            with np.load(archive, allow_pickle=False) as saved:
                data = GeometryArrays.capture(saved['vertices'], saved['faces'])
            if data.content_hash != inputs['geometry_hash']:
                raise ValueError('decoded saved geometry identity changed: ' + family)
            print('boundary-qualify ' + family, flush=True)
            result = qualify_geometry(data, python=args.python, timeout_s=15., ownership_root=children)
            if (result.get('geometry_content_hash') != data.content_hash or
                    result.get('toolchain_identity') not in (None, identity)):
                raise ValueError('qualification receipt identity changed: ' + family)
            receipt['cases'][family] = result
            (owner.root / 'results.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
            owner.register_file('results.json', 'final_output')
        passed = all(row.get('single_solid_qualified') for row in receipt['cases'].values())
        receipt.update(status='boundaries_qualified' if passed else 'required_boundaries_unqualified',
                       elapsed_seconds=time.monotonic() - started)
        (owner.root / 'results.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
        owner.register_file('results.json', 'final_output')
        if not passed:
            owner.mark_failed('required boundary qualification unavailable/failed')
        print('BOUNDARY_RESULT=' + str(owner.root / 'results.json'), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
