#!/usr/bin/env python3
"""Measure declared grayscale display transfer; no scene geometry or 3D render."""
from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    import bpy
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory under temp/tasks')
    arguments = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    root = Path(__file__).resolve().parents[1]
    output = arguments.output.resolve()
    if not output.is_relative_to(root/'temp/tasks'):
        raise ValueError('output must stay under repository temp/tasks')
    output.mkdir(parents=True, exist_ok=False)
    scene = bpy.context.scene
    settings = {key: getattr(scene.view_settings, key)
                for key in ('view_transform', 'look', 'exposure', 'gamma')}
    render = scene.render.image_settings
    original = {key: getattr(render, key) for key in ('file_format', 'color_mode', 'color_depth')}
    samples = 4097
    image = bpy.data.images.new('DeclaredCoverageTransfer', width=samples, height=1,
                                alpha=True, float_buffer=True)
    try:
        values = np.linspace(0., 1., samples)
        pixels = np.column_stack((values, values, values, np.ones(samples))).astype(np.float32)
        image.pixels.foreach_set(pixels.ravel())
        render.file_format, render.color_mode, render.color_depth = 'PNG', 'BW', '16'
        image.save_render(str(output/'transfer-lut.png'), scene=scene)
    finally:
        for key, value in original.items():
            setattr(render, key, value)
        bpy.data.images.remove(image)
    (output/'transfer-settings.json').write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
    receipt = {'samples': samples, 'linear_range': [0., 1.], 'encoding': '16-bit PNG grayscale',
               'scope': 'encoded grayscale for declared linear background fractions; black/white coverage only',
               'blender': bpy.app.version_string, 'python': sys.version, 'renders': 0,
               'sha256': hashlib.sha256((output/'transfer-lut.png').read_bytes()).hexdigest()}
    (output/'transfer-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
