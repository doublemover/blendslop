"""Retained single-case triangle inputs remain independently owned and immutable."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from utils.run_ownership import OwnedRun

spec=importlib.util.spec_from_file_location('adaptive_checkpoint',Path(__file__).resolve().parents[1]/'scripts/run_adaptive_family_detail.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AdaptiveTriangleCheckpointTests(unittest.TestCase):
    def test_original_single_case_receipt_verified_without_adoption(self):
        with tempfile.TemporaryDirectory() as folder:
            wire={'root_nodes':[{'primitive_type':'rounded_triangle'}]};raw={'protocol':'frozen_raw'}
            with OwnedRun(folder,producer='retained_triangle_fixture') as owner:
                files={'program.json':wire,'results.json':{'program':wire,'geometry_hash':'a'*64,'surface_observation':raw}}
                for name,value in files.items():
                    (owner.root/name).write_text(json.dumps(value),encoding='utf8');owner.register_file(name,'final_output')
                for name in ('evaluated-exact.npz','evaluated.obj'):
                    (owner.root/name).write_bytes(b'retained fixture');owner.register_file(name,'final_output')
            item={'family':'rounded_triangle_dot','baseline_directory':str(owner.root),
                  'baseline_receipt':str(owner.root/'results.json'),'baseline_receipt_layout':'single_case',
                  'baseline_geometry_hash':'a'*64,'baseline_raw_surface':raw,
                  'program_sha256':hashlib.sha256((owner.root/'program.json').read_bytes()).hexdigest()}
            before={p.name:p.read_bytes() for p in owner.root.iterdir()}
            self.assertEqual(module.verify_baseline_input(item,wire),files['results.json'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in owner.root.iterdir()})
            (owner.root/'evaluated.obj').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'byte identity'):
                module.verify_baseline_input(item,wire)


if __name__=='__main__':
    unittest.main()
