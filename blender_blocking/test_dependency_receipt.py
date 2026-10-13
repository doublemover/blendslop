"""Pure metadata/constraint contracts; no Blender installation or imports."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

PATH=Path(__file__).resolve().parents[1]/'scripts/capture_dependency_receipt.py'
SPEC=importlib.util.spec_from_file_location('capture_dependency_receipt',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class DependencyReceiptTests(unittest.TestCase):
    def fixture(self, rows):
        with tempfile.TemporaryDirectory() as root:
            requirements=Path(root)/'requirements.txt';requirements.write_text('example>=1\n')
            distributions=[SimpleNamespace(metadata={'Name':name},version=version) for name,version in rows]
            return MODULE.capture(requirements,distributions=distributions,blender_version='5.2.2')

    def test_exact_versions_are_canonical_and_do_not_claim_native_acceptance(self):
        row=self.fixture([('Example_Package','1.2.3'),('numpy','2.5.4')])
        self.assertEqual(row['packages'],{'example-package':'1.2.3','numpy':'2.5.4'})
        self.assertEqual(len(row['requirements_sha256']),64)
        self.assertIn('separate gates',row['qualification'])
        self.assertNotIn('path',row)

    def test_urls_and_ambiguous_versions_reject(self):
        for rows in [[('package','https://example.invalid/source')],[('bad/name','1')],[(None,'1')],[('package',None)],[('One_Package','1'),('one-package','2')]]:
            with self.assertRaises(ValueError):self.fixture(rows)


    def test_cli_preserves_existing_files_and_rejects_shared_path(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'receipt.json'
            args = ['--output', str(output), '--constraints', str(output)]
            with self.assertRaises(ValueError): MODULE.main(args)
            self.assertFalse(output.exists())
            output.write_text('original receipt')
            args[-1] = str(Path(root) / 'constraints.txt')
            with self.assertRaises(FileExistsError): MODULE.main(args)
            self.assertEqual(output.read_text(), 'original receipt')
            self.assertFalse(Path(args[-1]).exists())


if __name__=='__main__':unittest.main()
