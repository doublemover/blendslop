#!/usr/bin/env python3
"""Freeze source-conditioned contracts before reading retained candidate metrics.

Pure Python; no render, fitting, surface resampling, topology qualification or
acceptance gate mutation. All reference/camera bytes are verified against an
explicit source plan. Prior candidate exposure is disclosed rather than hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'blender_blocking'), str(ROOT)]

import numpy as np
from blender_blocking.evaluation.canonical_artifacts import CANONICAL_VIEWS, camera_frame_sha256
from blender_blocking.evaluation.family_surface_contracts import (
    SUPPORTED_FAMILIES, evaluate_family_surface_contract, freeze_family_surface_contract,
)
from blender_blocking.evaluation.reference_noise import oriented_surface_identity
from blender_blocking.reconstruction.native_geometry import GeometryArrays
from utils.run_ownership import OwnedRun


def read_bound(binding, *, limit=16777216):
    path = Path(binding['path']).absolute()
    if (path != path.resolve() or path.is_symlink() or getattr(path, 'is_junction', lambda:False)()):
        raise ValueError('redirected policy input is refused')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError('policy input bytes differ or exceed the bound: ' + str(path))
    return data


def arrays(binding):
    import io
    data = read_bound(binding)
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        if sum(item.file_size for item in zipped.infolist()) > 16777216:
            raise ValueError('source geometry archive expansion exceeds the bound')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return GeometryArrays.capture(archive['vertices'], archive['faces'])


def publish(owner, name, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
    owner.reserve_bytes(len(data))
    path = owner.root / name
    path.write_bytes(data)
    owner.register_file(name, 'diagnostic')
    return path


def freeze_sources(plan):
    if (plan.get('protocol') != 'family-surface-source-plan-v1' or
            not 1 <= len(plan['families']) <= len(SUPPORTED_FAMILIES) or
            not set(plan['families']).issubset(SUPPORTED_FAMILIES)):
        raise ValueError('explicit supported-family source plan required')
    declarations = {}
    for family, source in plan['families'].items():
        original, regenerated = arrays(source['reference_npz']), arrays(source['regenerated_npz'])
        if original.content_hash != source['reference_geometry_hash'] or regenerated.content_hash != source['regenerated_geometry_hash']:
            raise ValueError('source indexed geometry differs: ' + family)
        exact = oriented_surface_identity(original)
        if exact != oriented_surface_identity(regenerated) or exact != source['oriented_surface_sha256']:
            raise ValueError('source oriented triangle equivalence differs: ' + family)
        cameras = json.loads(read_bound(source['camera_records']))
        if set(cameras) != set(CANONICAL_VIEWS):
            raise ValueError('actual source cameras are incomplete')
        for view, record in cameras.items():
            binding = record.get('pass_artifacts', {}).get('neutral', {})
            if (record.get('geometry_hash') != regenerated.content_hash or
                    record.get('geometry_unchanged_after_passes') is not True or
                    binding.get('geometry_hash') != regenerated.content_hash or
                    binding.get('camera_sha256') != camera_frame_sha256(record)):
                raise ValueError('actual source camera/neutral producer binding is incomplete')
            bound = source['neutral_artifacts'][view]
            if bound['sha256'] != binding.get('sha256') or Path(bound['path']).name != binding.get('path'):
                raise ValueError('neutral artifact declaration differs from producer')
            read_bound(bound)
        declarations[family] = freeze_family_surface_contract(
            family, source['authored_parameters'], original, cameras,
            reconstruction_pixels=plan['reconstruction_pixels'],
            normal_allowance_degrees=plan['normal_allowance_degrees'])
        declarations[family]['source_provenance'] = {
            'original_and_regenerated_oriented_surface_sha256': exact,
            'indexed_reference_geometry_hash': original.content_hash,
            'indexed_regenerated_geometry_hash': regenerated.content_hash,
            'scope': 'actual recaptured source cameras bound to exactly equivalent oriented geometry; historical mask bindings unchanged'}
    return declarations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--source-plan-sha256', required=True)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--coverage-sha256', required=True)
    parser.add_argument('--additional-observations', type=Path)
    parser.add_argument('--additional-observations-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.additional_observations) != bool(args.additional_observations_sha256):
        parser.error('additional observations require an explicit byte hash')
    started = time.monotonic()
    source_binding = {'path':str(args.source_plan.absolute()), 'sha256':args.source_plan_sha256}
    plan = json.loads(read_bound(source_binding))
    declarations = freeze_sources(plan)
    owner = OwnedRun(args.output.absolute(), producer='family_surface_contract_report',
                     max_generated_bytes=16777216)
    error = None
    try:
        publish(owner, 'source-plan.json', plan)
        # Provenance is additive outside the signed policy body.
        frozen = {name:{'contract':{k:v for k,v in row.items() if k != 'source_provenance'},
                        'source_provenance':row['source_provenance']}
                  for name,row in declarations.items()}
        publish(owner, 'frozen-contracts.json', frozen)
        # The source-only limits are now persisted. No candidate metrics were
        # used by the preceding freeze; human historical exposure is disclosed.
        coverage_binding = {'path':str(args.coverage.absolute()), 'sha256':args.coverage_sha256}
        coverage = json.loads(read_bound(coverage_binding))
        observations = {}
        for family, declaration in frozen.items():
            candidate = coverage['actual_rows'][family]
            # Bind the summary to its real producer receipt, not just copied text.
            producer = json.loads(read_bound(candidate['source_receipt']))
            producer_row = producer['cases'][family]
            if (producer_row['geometry_hash'] != candidate['geometry_hash'] or
                    producer_row['surface_observation'] != candidate['surface_observation']):
                raise ValueError('raw metric summary differs from its retained producer')
            observations[family] = evaluate_family_surface_contract(
                declaration['contract'], candidate['surface_observation'],
                candidate_geometry_hash=candidate['geometry_hash'])
        additional = {}
        if args.additional_observations:
            binding = {'path':str(args.additional_observations.absolute()),
                       'sha256':args.additional_observations_sha256}
            selected = json.loads(read_bound(binding))
            for family, entry in selected.items():
                if family not in frozen or entry['raw_field'] != ['raw_surface', 'refined']:
                    raise ValueError('additional selected observation must use its declared refined raw receipt')
                producer = json.loads(read_bound(entry['source_receipt']))
                row = producer['cases'][family]
                if row['geometry_hash'] != entry['candidate_geometry_hash']:
                    raise ValueError('selected raw receipt geometry differs')
                additional[family] = evaluate_family_surface_contract(
                    frozen[family]['contract'],row['raw_surface']['refined'],
                    candidate_geometry_hash=entry['candidate_geometry_hash'])
            publish(owner, 'additional-observation-bindings.json', selected)
        report = {'protocol':'family-surface-policy-report-v1', 'status':'completed',
                  'source_plan':source_binding, 'coverage':coverage_binding,
                  'source_hashes':{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'blender_blocking/evaluation/family_surface_contracts.py':hashlib.sha256((ROOT/'blender_blocking/evaluation/family_surface_contracts.py').read_bytes()).hexdigest()},
                  'run_root':str(owner.root), 'observations':observations,
                  'additional_selected_observations':additional,
                  'aggregate_accepted':False, 'renders':0, 'fits':0,
                  'surface_comparisons':0, 'qualification_children':0,
                  'elapsed_seconds':time.monotonic()-started,
                  'remaining':'Artist tolerance and unsupported analytic certificates remain independent required gaps; existing acceptance gates unchanged'}
        result = publish(owner, 'results.json', report)
        print(json.dumps({'result':str(result), 'engineering':{k:v['engineering_surface_passed'] for k,v in observations.items()}, 'aggregate_accepted':False}))
    except BaseException as exc:
        error = exc
        try:
            publish(owner, 'failure.json', {'status':'failed', 'error':repr(exc), 'aggregate_accepted':False})
        except Exception as secondary:
            if hasattr(exc,'add_note'):
                exc.add_note('Policy failure receipt also failed: '+repr(secondary))
        raise
    finally:
        owner.close(error=error)


if __name__ == '__main__':
    main()
