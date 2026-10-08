"""Metric-aligned common-case scores; missing native assets stay unavailable."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()
from blender_blocking.evaluation.protocols import (
    evaluate_mesh_profile, save_sample_bundle, macro_cases, dtu_native_adapter, DTU_MODES,
)

COMMON_PROFILES = ('common_surface_v1', 'superflex_formula_common_v1', 'superfit_formula_common_v1')


def evaluate_row(row, *, group, phase, output, revision, seed, profiles=COMMON_PROFILES, native_cases=None):
    """Persist every requested cell, including process/mesh/evaluator failures."""
    variant = row.get('campaign_variant', 'final')
    case_id = group + ':' + row['case'] + ':' + variant + ':' + row['requested_mode']
    reference = phase / group / 'references' / row['case'] / 'ground_truth.obj'
    mesh = None
    failure = None
    try:
        if not row.get('result_path'):
            failure = 'reconstruction process failed'
        else:
            payload = json.loads(Path(row['result_path']).read_text())
            mesh = payload.get('mesh_path')
            if not mesh:
                failure = 'missing candidate mesh'
    except (OSError, ValueError) as exc:
        failure = f'reconstruction receipt unavailable: {type(exc).__name__}: {exc}'
    items = {}
    for profile in profiles:
        result = {'status': 'unavailable', 'profile': profile, 'reason': failure}
        if failure is None:
            try:
                native_config = (native_cases or {}).get(profile, {}).get(row['case'])
                if profile.endswith('_native_v1'):
                    if not native_config:
                        raise ValueError('native prepared inputs not supplied for this case/track')
                    result = evaluate_native_profile(reference, mesh, profile, native_config, seed,
                                                     output/group/row['case']/variant/row['requested_mode'])
                else:
                    result = evaluate_mesh_profile(reference, mesh, profile=profile, seed=seed)
            except Exception as exc:
                result = {'status': 'unavailable', 'profile': profile,
                          'reason': f'evaluation failed: {type(exc).__name__}: {exc}'}
        destination = output / group / row['case'] / variant / row['requested_mode'] / profile
        calibration = json.dumps(row.get('calibration', {}), sort_keys=True).encode()
        save_sample_bundle(destination, result, case_id=case_id, seed=seed, original_units='metres',
            source_hashes={'reference': row.get('ground_truth_sha256'), 'prediction': row.get('mesh_sha256')},
            mask_hashes=row.get('input_hashes', {}),
            camera_hashes={'calibration': hashlib.sha256(calibration).hexdigest()},
            evaluator_revision=revision)
        items[profile] = {'case_id': case_id, 'mode': variant+':'+row['requested_mode'], 'status': result['status'],
                         'metrics': result.get('metrics'), 'reason': result.get('reason'),
                         'bundle': str(destination)}
    return items


def resolved(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_native_profile(reference, prediction, profile, config, seed, output):
    import numpy as np
    from blender_blocking.evaluation.protocols.core import apply_transform
    from blender_blocking.evaluation.comparable_geometry import read_obj
    from blender_blocking.reconstruction.mesh_io import write_obj
    native_reference = resolved(config.get('reference', reference))
    metadata = config.get('reference_metadata', {})
    if isinstance(metadata, str):
        metadata = json.loads(resolved(metadata).read_text(encoding='utf-8-sig'))
    metadata = dict(metadata)
    source_hashes = config.get('source_sha256', {})
    for key in ('reference', 'reference_fps', 'transform', 'prediction_transform', 'reference_metadata'):
        value = config.get(key)
        if isinstance(value, str) and source_hashes.get(key) != file_hash(resolved(value)):
            raise ValueError('native input hash missing/mismatched: '+key)
    fps = np.load(resolved(config['reference_fps']), allow_pickle=False) if config.get('reference_fps') else None
    if profile.startswith('superflex'):
        if fps is None or metadata.get('reference_sha256') != file_hash(resolved(config['reference_fps'])):
            raise ValueError('released FPS metadata must match the supplied bytes')
    if profile.startswith('superfit'):
        receipt = metadata.get('upstream_preparation_receipt', {})
        if receipt.get('target_sha256') != file_hash(native_reference):
            raise ValueError('upstream cleanup receipt does not match prepared target')
    matrix = np.load(resolved(config['transform']), allow_pickle=False) if config.get('transform') else None
    if config.get('prediction_transform'):
        vertices, faces = read_obj(prediction)
        vertices = apply_transform(vertices, np.load(resolved(config['prediction_transform']), allow_pickle=False))
        output.mkdir(parents=True, exist_ok=True)
        prediction = write_obj(output/(profile+'-native-prediction.obj'), {'vertices': vertices, 'faces': faces})
    return evaluate_mesh_profile(native_reference, prediction, profile=profile, seed=seed,
        reference_fps=fps, provided_transform=matrix, reference_metadata=metadata)


def evaluate_dtu_track(row, config, mode, output):
    """Use the existing standalone evaluator and preserve its hash/culling gate."""
    if not config:
        return {'status': 'unavailable', 'reason': 'native DTU assets/frame/receipts not supplied', 'mode': mode}
    if not row.get('result_path'):
        return {'status': 'unavailable', 'reason': 'candidate process failed', 'mode': mode}
    payload = json.loads(Path(row['result_path']).read_text())
    mesh = payload.get('mesh_path')
    if not mesh:
        return {'status': 'unavailable', 'reason': 'candidate mesh missing', 'mode': mode}
    python = resolved(config.get('evaluator_python', '.venv312/Scripts/python.exe'))
    command = [str(python), '-B', str(ROOT/'scripts/evaluate_dtu_subset.py'), '--prediction', mesh,
               '--case-id', row['case'], '--mode', mode, '--seed', str(row.get('seed', 1234)), '--output', str(output)]
    for key in ('assets', 'dataset_dir', 'scale_matrix', 'culling', 'culling_receipt'):
        if config.get(key):
            command += ['--'+key.replace('_', '-'), str(resolved(config[key]))]
    if config.get('prediction_native_mm'):
        command += ['--prediction-native-mm']
    if config.get('scan') is not None:
        command += ['--scan', str(config['scan'])]
    command += ['--max-points', str(config.get('max_points', 2_000_000))]
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(__import__('os').environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        log, _ = child.communicate(timeout=float(config.get('timeout_s', 100.)))
    except BaseException:
        child.kill(); child.communicate(); raise
    if not (output/'result.json').is_file():
        return {'status': 'unavailable', 'mode': mode, 'reason': log.decode(errors='replace')[-2000:]}
    result = json.loads((output/'result.json').read_text(encoding='utf-8'))
    return {'status': result['status'], 'mode': mode, 'reason': result.get('reason'),
            'metrics': {key: result[key] for key in ('accuracy_mm', 'completeness_mm', 'overall_mm') if key in result},
            'bundle': str(output/'result.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plan = json.loads(args.manifest.read_text(encoding='utf-8-sig')) if args.manifest else {}
    protocols = plan.get('protocols', {})
    profiles = tuple(protocols.get('common', COMMON_PROFILES))+tuple(protocols.get('native_profiles', {}))
    native_cases = protocols.get('native_profiles', {})
    summary = {'schema': 'metric_aligned_campaign_v2', 'dataset': 'declared synthetic/external cases',
               'claims': 'common geometry and explicitly supplied native metric tracks; no upstream method reproduction',
               'profiles': {}, 'native_unavailable': []}
    if (args.phase/'campaign-rows.json').exists():
        rows = json.loads((args.phase/'campaign-rows.json').read_text())['rows']
    else:
        rows = []
        for group in ('paired', 'heldout'):
            manifest = json.loads((args.phase/group/'final.json').read_text())
            rows += [{**row, 'split': group} for row in manifest['rows']]
    for row in rows:
        group = row['split']
        items = evaluate_row(row, group=group, phase=args.phase, output=args.output,
                            revision=revision, seed=row.get('seed', 1234), profiles=profiles, native_cases=native_cases)
        for profile, item in items.items():
            summary['profiles'].setdefault(group+':'+profile, []).append(item)
        for mode, cases in protocols.get('dtu', {}).items():
            config = cases.get(row['case'])
            if config and 'variants' in config:
                config = {**config, **config['variants'].get(row.get('campaign_variant', 'final'), {})}
            try:
                result = evaluate_dtu_track(row, config, mode, args.output/group/row['case']/
                    row.get('campaign_variant', 'final')/row['requested_mode']/('dtu-'+mode))
            except Exception as exc:
                result = {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}', 'mode': mode}
            item = {**result, 'case_id': group+':'+row['case']+':'+row.get('campaign_variant', 'final')+':'+row['requested_mode'],
                    'mode': row.get('campaign_variant', 'final')+':'+row['requested_mode']}
            summary['profiles'].setdefault(group+':dtu-'+mode, []).append(item)
    summary['aggregates'] = {}
    for key, cases in summary['profiles'].items():
        for mode in sorted({case['mode'] for case in cases}):
            summary['aggregates'][key + ':' + mode] = macro_cases([case for case in cases if case['mode'] == mode])
    for key, cases in summary['profiles'].items():
        if '_native_' in key or ':dtu-' in key:
            missing = [row['case_id'] for row in cases if row['status'] != 'available']
            if missing:
                summary['native_unavailable'].append({'track': key, 'affected_cells': missing,
                                                       'blocks_other_tracks': False})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved selected metric tracks; unavailable native inputs affect their own cells only', flush=True)


if __name__ == '__main__':
    main()
