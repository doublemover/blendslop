"""Serial preparation of the eight recovered GLBs using the existing Blender tool."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'temp/quality-campaign-inputs-20261006'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def stage(source_manifest):
    manifest = json.loads(source_manifest.read_text(encoding='utf-8-sig'))
    extracted = Path(manifest['extractedPath'])
    semantic = json.loads((extracted/'PartObjaverse-Tiny_semantic.json').read_text())
    subset = json.loads((extracted/'subset-manifest.json').read_text())
    cases = []
    for index, selected in enumerate(subset['objects'], 1):
        filename = 'meshes/'+selected['id']+'.glb'
        verified = next(row for row in manifest['files'] if row['path'] == filename)
        source = extracted/filename
        data = source.read_bytes()
        if len(data) != verified['bytes'] or hashlib.sha256(data).hexdigest() != verified['sha256']:
            raise ValueError('staged input hash mismatch: '+filename)
        destination = INPUT/'sources'/filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError('refusing to replace changed source')
        if not destination.exists():
            shutil.copy2(source, destination)
        categories = {category: items[selected['id']] for category, items in semantic.items() if selected['id'] in items}
        case = {'id': f"ext{index:02}_"+selected['id'][:7], 'source_id': selected['id'],
                'source': str(destination.relative_to(ROOT)), 'sha256': verified['sha256'],
                'semantic_labels': categories, 'dataset': 'PartObjaverse-Tiny pinned capability subset',
                'selection_note': subset['selection'], 'upstream': subset['source']}
        cases.append(case)
    dump(INPUT/'staged-cases.json', {'cases': cases, 'original_archive_sha256': manifest['archiveSha256'],
        'original_archive_bytes': manifest['archiveBytes'], 'verified_manifest': str(source_manifest)})
    for filename in ('README.md', 'SUBSET_README.md', 'subset-manifest.json', 'PartObjaverse-Tiny_semantic.json'):
        destination = INPUT/'sources'/filename
        if not destination.exists():
            shutil.copy2(extracted/filename, destination)
    return cases


def prepare(cases, blender, *, render=False):
    for case in cases:
        output = INPUT/('references' if render else 'inspection')/case['id']
        if output.exists():
            raise FileExistsError('preserve existing preparation: '+str(output))
        command = [blender, '--background', '--factory-startup', '--disable-autoexec', '--threads', '4',
            '--python-exit-code', '2', '--python', str(ROOT/'scripts/prepare_external_mesh_inputs.py'), '--',
            '--source', str(ROOT/case['source']), '--output', str(output), '--case-id', case['id'],
            '--dataset', case['dataset'], '--selection-note', case['selection_note']]
        if render:
            command += ['--render', '--review-receipt', str(INPUT/'reviews'/(case['id']+'.json'))]
        else:
            command += ['--inspect-render']
        log = INPUT/'logs'/(("reference-" if render else "inspection-")+case['id']+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        print('PREPARE '+case['id'], flush=True)
        with log.open('w', encoding='utf-8') as stream:
            env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                code = child.wait(timeout=100.)
            except BaseException:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                else:
                    child.kill()
                child.wait(); raise
        if code:
            raise RuntimeError('input preparation failed: '+str(log))
        print('PREPARED '+case['id'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--blender', default='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
    p.add_argument('--action', choices=('stage', 'inspect', 'references'), required=True)
    args = p.parse_args()
    if args.action == 'stage':
        cases = stage(args.source_manifest)
        print('Staged '+str(len(cases))+' original verified GLBs')
    else:
        cases = json.loads((INPUT/'staged-cases.json').read_text())['cases']
        prepare(cases, args.blender, render=args.action == 'references')


if __name__ == '__main__':
    main()
