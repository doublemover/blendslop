"""Process-owned identity reuse still invalidates on actual toolchain mutation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,unittest
from blender_blocking.reconstruction import native_qualification as qualification


class ToolchainIdentityCacheTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);python=root/'environment'/'Scripts'/'python.exe'
        site=root/'environment'/'Lib'/'site-packages'
        files=[python,site/'open3d-0.19.dist-info'/'METADATA',site/'open3d'/'cpu'/'pybind-fixture.pyd',
               site/'numpy-2.3.dist-info'/'METADATA',root/'scripts'/'qualify.py',
               root/'blender_blocking'/'evaluation'/'triangle_contacts.py']
        for index,p in enumerate(files):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fixture-'+str(index)).encode())
        return python,files

    def test_unchanged_identity_reads_large_file_bytes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            python,files=self.fixture(directory)
            bpy=SimpleNamespace(app=SimpleNamespace(version_string='fixture',build_hash=b'fixture-build'))
            qualification._IDENTITY_CACHE.clear()
            with patch.dict('sys.modules',{'bpy':bpy}),patch.object(qualification,'ROOT',Path(directory)),\
                 patch.object(qualification,'HELPER',files[4]),\
                 patch.object(qualification,'_hash_identity_file',wraps=qualification._hash_identity_file) as hash_file:
                first=qualification.toolchain_identity(python);calls=hash_file.call_count
                second=qualification.toolchain_identity(python)
                self.assertEqual(first,second);self.assertEqual(hash_file.call_count,calls)
                forced=qualification.toolchain_identity(python,force=True)
                self.assertEqual(first,forced);self.assertGreater(hash_file.call_count,calls)

    def test_helper_predicate_native_and_build_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            python,files=self.fixture(directory)
            bpy=SimpleNamespace(app=SimpleNamespace(version_string='fixture',build_hash=b'build-one'))
            qualification._IDENTITY_CACHE.clear()
            with patch.dict('sys.modules',{'bpy':bpy}),patch.object(qualification,'ROOT',Path(directory)),patch.object(qualification,'HELPER',files[4]):
                identity=qualification.toolchain_identity(python)
                for file in (files[4],files[5],files[2],python):
                    file.write_bytes(file.read_bytes()+b'-changed')
                    current=qualification.toolchain_identity(python)
                    self.assertNotEqual(identity,current);identity=current
                bpy.app.build_hash=b'build-two'
                self.assertNotEqual(identity,qualification.toolchain_identity(python))

    def test_geometry_content_identity_is_still_independent(self):
        import numpy as np
        from reconstruction.native_geometry import GeometryArrays
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
        first=GeometryArrays.capture(vertices,faces);second=GeometryArrays.capture(vertices+[.1,0.,0.],faces)
        self.assertNotEqual(first.content_hash,second.content_hash)
        self.assertEqual(first.connectivity_hash,second.connectivity_hash)


if __name__=='__main__':unittest.main()
