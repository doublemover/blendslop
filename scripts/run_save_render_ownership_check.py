#!/usr/bin/env python3
"""Two current-scene still checks; run only under an explicit bounded supervisor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'blender_blocking'))
import test_runner  # Existing qualified package bootstrap; never installs dependencies.
from integration.blender_ops.render_utils import save_render


def file_record(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def scene_settings(scene):
    return {'engine': scene.render.engine, 'format': scene.render.image_settings.file_format,
            'extension': scene.render.use_file_extension, 'frame': scene.frame_current,
            'resolution': [scene.render.resolution_x, scene.render.resolution_y,
                           scene.render.resolution_percentage],
            'camera': scene.camera.name, 'camera_matrix': [list(row) for row in scene.camera.matrix_world],
            'shading': [scene.display.shading.light, scene.display.shading.color_type]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    root = args.output_parent.absolute()
    root.mkdir(parents=True, exist_ok=False)
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError('factory-startup camera is required')
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.film_transparent = False
    scene.frame_set(19)
    previous = scene.render.filepath
    scene.render.image_settings.file_format = 'PNG'
    png = root / 'prior.png'
    with png.open('xb') as stream:
        stream.write(b'preserve these existing final bytes\n')
    prior = file_record(png)
    before = scene_settings(scene)
    try:
        save_render(str(png))
    except FileExistsError:
        pass
    else:
        raise AssertionError('an existing exact PNG final must raise FileExistsError')
    assert file_record(png) == prior
    assert scene.render.filepath == previous
    assert scene_settings(scene) == before
    owners = sorted((root / '.render-runs').glob('owned-*'))
    assert len(owners) == 1
    failed_owner = owners[0]
    failed = json.loads((failed_owner / 'render-result.json').read_bytes())
    failed_stage = failed_owner / failed['stage']['path']
    from PIL import Image
    with Image.open(failed_stage) as image:
        assert image.format == 'PNG' and image.size == (128, 128)
        image.verify()
    assert failed['status'] == 'failed' and failed['published'] is False
    assert json.loads((failed_owner / 'run-lease.json').read_bytes())['status'] == 'released'
    assert json.loads((failed_owner / 'run-ownership.json').read_bytes())['state'] == 'failed'

    scene.render.image_settings.file_format = 'OPEN_EXR'
    requested = root / 'native-extension'
    before = scene_settings(scene)
    result = save_render(str(requested))
    assert result is None and scene.render.filepath == str(requested)
    assert scene_settings(scene) == before and not requested.exists()
    current = sorted((root / '.render-runs').glob('owned-*'))
    assert len(current) == 2
    success_owner = next(owner for owner in current if owner != failed_owner)
    success = json.loads((success_owner / 'render-result.json').read_bytes())
    final = Path(success['final_path'])
    assert final == requested.with_suffix('.exr') and final.is_file()
    success_stage = success_owner / success['stage']['path']
    with final.open('rb') as stream:
        assert stream.read(4) == b'\x76\x2f\x31\x01'
    assert file_record(final)['sha256'] == file_record(success_stage)['sha256']
    assert success['status'] == 'succeeded' and success['published'] is True
    assert json.loads((success_owner / 'run-lease.json').read_bytes())['status'] == 'released'
    assert json.loads((success_owner / 'run-ownership.json').read_bytes())['state'] == 'succeeded'
    assert file_record(png) == prior
    evidence = {'protocol': 'save-render-native-check-v1', 'status': 'succeeded',
                'scope': 'two 128-square factory-cube frames; PNG collision and native EXR filename',
                'blender': bpy.app.version_string, 'build_hash': bpy.app.build_hash.decode(),
                'native_frames': 2, 'geometry_changes': 0, 'cleanup': False,
                'prior_final': prior, 'png_stage': file_record(failed_stage),
                'png_owner': str(failed_owner), 'exr_stage': file_record(success_stage),
                'exr_final': file_record(final), 'exr_owner': str(success_owner),
                'format_proof': 'PNG readability/128-square; EXR magic and identical closed staged/final bytes',
                'limits': 'outer supervisor declares work/join/memory; stage budget is accounting only'}
    with (root / 'native-evidence.json').open('x', encoding='utf8') as stream:
        json.dump(evidence, stream, indent=2)
        stream.write('\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
