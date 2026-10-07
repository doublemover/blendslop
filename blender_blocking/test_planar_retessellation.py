"""Exact-plane simplification preserves original boundaries, holes and gaps."""
import unittest
import numpy as np
from reconstruction.native_geometry import GeometryArrays
from reconstruction.planar_retessellation import retessellate_coplanar
from reconstruction.grouped_solids import solid_guard,signed_volume,concatenate


def grid_cube(n=4,*,offset=(0.,0.,0.)):
    axis=np.linspace(-1.,1.,n+1);vertices=[];lookup={};faces=[]
    for normal in range(3):
        other=[i for i in range(3) if i!=normal]
        for end in (0,n):
            for i in range(n):
                for j in range(n):
                    quad=[]
                    for a,b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1)):
                        point=np.zeros(3);point[normal]=axis[end];point[other]=[axis[a],axis[b]]
                        point+=offset;key=tuple(point)
                        if key not in lookup:lookup[key]=len(vertices);vertices.append(point)
                        quad.append(lookup[key])
                    xyz=np.asarray([vertices[k] for k in quad])
                    sign=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])[normal]
                    if sign*(1 if end==n else -1)<0:quad=quad[::-1]
                    faces.extend(((quad[0],quad[1],quad[2]),(quad[0],quad[2],quad[3])))
    return GeometryArrays.capture(np.asarray(vertices),np.asarray(faces))


try:
    from shapely import constrained_delaunay_triangles
    SHAPELY_AVAILABLE=True
except ImportError:
    SHAPELY_AVAILABLE=False


@unittest.skipUnless(SHAPELY_AVAILABLE,"optional Shapely>=2.1 constrained triangulation unavailable")
class PlanarRetessellationTests(unittest.TestCase):
    def test_dense_cube_reduces_to_its_original_boundary(self):
        original=grid_cube(8)
        final,report=retessellate_coplanar(original,timeout_s=2.)
        self.assertEqual(report['status'],'retessellated',report)
        self.assertEqual(len(final.faces),12)
        self.assertEqual(len(final.vertices),8)
        self.assertTrue(solid_guard(final)['valid_solid'])
        self.assertAlmostEqual(signed_volume(final),signed_volume(original),places=12)
        self.assertFalse(report['geometric_tolerance_used'])
        self.assertFalse(report['single_solid_qualified'])
        for vertex in final.vertices:self.assertTrue(np.any(np.all(original.vertices==vertex,axis=1)))

    def test_real_gap_and_disconnected_components_remain(self):
        original=concatenate([grid_cube(4),grid_cube(4,offset=(2.01,0.,0.))])
        final,report=retessellate_coplanar(original,timeout_s=2.)
        self.assertEqual(report['status'],'retessellated',report)
        self.assertEqual(solid_guard(final)['connected_components'],2)
        self.assertEqual(len(final.faces),24)
        np.testing.assert_allclose(final.vertices.min(0),[-1.,-1.,-1.])
        np.testing.assert_allclose(final.vertices.max(0),[3.01,1.,1.])

    def test_perforated_plate_preserves_material_volume_and_inner_boundary(self):
        from primitives.polygon_extrusion import PolygonExtrusionPrimitive
        from test_geometry_consistency import triangles
        outer=np.array([[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]],float)
        hole=np.array([[-.5,-.5],[-.5,0],[-.5,.5],[0,.5],[.5,.5],[.5,0],[.5,-.5],[0,-.5]])
        mesh=PolygonExtrusionPrimitive(outer,(hole,),height=.2).to_mesh_data()
        original=GeometryArrays.capture(mesh.vertices,triangles(mesh))
        final,report=retessellate_coplanar(original,timeout_s=2.)
        self.assertEqual(report['status'],'retessellated',report)
        self.assertAlmostEqual(signed_volume(final),.6,places=12)
        self.assertTrue(solid_guard(final)['valid_solid'])
        self.assertLess(len(final.faces),len(original.faces))
        self.assertFalse(any(abs(p[0])<.5 and abs(p[1])<.5 for p in final.vertices))

    def test_deadline_or_nonclosed_input_retains_the_exact_original(self):
        original=grid_cube(4)
        final,report=retessellate_coplanar(original,timeout_s=0.)
        self.assertIs(final,original)
        self.assertEqual(report['status'],'unchanged')
        opened=GeometryArrays.capture(original.vertices,original.faces[:-1])
        final,report=retessellate_coplanar(opened)
        self.assertIs(final,opened)
        self.assertIn('input topology',report['reason'])

    def test_nearly_coplanar_curvature_is_not_flattened(self):
        original=grid_cube(4);vertices=original.vertices.copy()
        vertices[:,0]+=np.where(vertices[:,0]>0.,.01*vertices[:,1]**2,0.)
        bent=GeometryArrays.capture(vertices,original.faces)
        final,report=retessellate_coplanar(bent,timeout_s=2.)
        for p in final.vertices:self.assertTrue(np.any(np.all(bent.vertices==p,axis=1)))
        self.assertAlmostEqual(signed_volume(final),signed_volume(bent),places=12)


if __name__=='__main__':unittest.main()
