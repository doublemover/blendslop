"""Record an installed runtime without URLs, local paths or inferred qualification."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys


def capture(requirements, *, distributions=None, blender_version=None):
    versions = {}
    installed = metadata.distributions() if distributions is None else distributions
    for dist in installed:
        name = dist.metadata.get('Name', '')
        version = dist.version
        if (not isinstance(name, str) or not isinstance(version, str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]*', version)):
            raise ValueError('Ordinary distribution name and exact version required')
        key = re.sub(r'[-_.]+', '-', name).lower()
        if key in versions and versions[key] != version:
            raise ValueError('Conflicting installed distribution versions: ' + key)
        versions[key] = version
    return {
        'schema': 'blendslop.dependency-receipt/v1',
        'python': platform.python_version(),
        'implementation': platform.python_implementation(),
        'platform': sys.platform,
        'machine': platform.machine(),
        'declared_blender_version': blender_version,
        'requirements_sha256': hashlib.sha256(Path(requirements).read_bytes()).hexdigest(),
        'packages': dict(sorted(versions.items())),
        'qualification': ('Installed metadata only. Native tests, geometry acceptance '
                          'and dependency compatibility are separate gates.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, default=Path('blender_blocking/requirements.txt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--constraints', type=Path, required=True)
    parser.add_argument('--blender-version')
    args = parser.parse_args(argv)
    if args.output.resolve() == args.constraints.resolve():
        raise ValueError('Receipt and constraints must have distinct paths')
    for path in (args.output, args.constraints):
        if path.exists():
            raise FileExistsError('Fresh dependency receipt paths required')
    result = capture(args.requirements, blender_version=args.blender_version)
    for path in (args.output, args.constraints):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    args.constraints.write_text(
        '# Exact observed runtime; use only with the matching Python/platform receipt.\n'
        + ''.join(f'{name}=={version}\n' for name, version in result['packages'].items()),
        encoding='utf-8',
    )
    print(json.dumps({'packages': len(result['packages']), 'python': result['python'],
                      'platform': result['platform'], 'native_qualified': False}))


if __name__ == '__main__':
    main()
