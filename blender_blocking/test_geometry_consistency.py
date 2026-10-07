"""Deterministic field/mesh/output contracts; these are not quality benchmarks."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from primitives.superfrustum import SuperFrustum
from primitives.analytic_primitives import EllipsoidPrimitive, SuperquadricPrimitive, AnisotropicGaussianPrimitive
from reconstruction.output_targets import output_mesh_targets
from reconstruction.grouped_solids import overlap_groups, signed_volume
from reconstruction.native_geometry import GeometryArrays
from reconstruction.projection_contract import project_vertices
from test_quality_geometry import target_for_masks


def triangles(mesh):
    return np.asarray([(f[0], f[i], f[i + 1]) for f in mesh.faces for i in range(1, len(f) - 1)], int)


class FakeObject(dict):
    def __init__(self, kind='MESH', parent=None, excluded=False):
        super().__init__(blendslop_export_exclude=excluded)
        self.type, self.parent, self.children_recursive = kind, parent, []


class GeometryConsistencyTests(unittest.TestCase):
    def test_frustum_caps_and_independent_cylinder_distance(self):
        part = SuperFrustum(radius_bottom=1., radius_top=1., height=2.)
        points = np.array([[0., 0., -1.], [0., 0., 1.], [0., 0., -3.],
                           [0., 0., 0.], [2., 0., 2.], [2., 0., 0.]])
        np.testing.assert_allclose(part.sdf_batch(points), [0., 0., 2., -1., np.sqrt(2.), 1.], atol=1e-12)
        np.testing.assert_allclose([part.sdf(p) for p in points], part.sdf_batch(points))

    def test_frustum_field_samples_mesh_and_transformed_zero_set(self):
        for radii in ((1., .3), (.02, 1.), (1., .000001), (1., 1.)):
            part = SuperFrustum(position=(2., -3., .5), orientation=(.7, 1.1),
                                radius_bottom=radii[0], radius_top=radii[1], height=2.4)
            for points in (part.sample_surface(93), part.to_mesh_data(16).vertices):
                np.testing.assert_allclose(part.sdf_batch(points), 0., atol=3e-15)
            restored = SuperFrustum.from_dict(part.to_dict())
            np.testing.assert_array_equal(restored.to_mesh_data(16).vertices, part.to_mesh_data(16).vertices)
            center = part.position[None, :]
            self.assertLess(part.sdf_batch(center)[0], 0.)

    def test_frustum_distance_commutes_with_scaling(self):
        part = SuperFrustum(position=(.3, -.2, 1.), orientation=(1., .4), height=3., radius_bottom=.7, radius_top=.2)
        points = np.array([[0., 0., 0.], [1., -1., 2.], [.1, .1, 1.]])
        scaled = SuperFrustum(position=part.position*100., orientation=part.orientation,
                             height=part.height*100., radius_bottom=70., radius_top=20.)
        np.testing.assert_allclose(scaled.sdf_batch(points*100.)/100., part.sdf_batch(points), atol=1e-14)

    def test_poles_have_outward_normals_and_balanced_directed_edges(self):
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_rotvec((.4, -.2, .1)).as_matrix()
        for part in (EllipsoidPrimitive(center=(2., -1., 4.), radii=(.7, .4, 1.), rotation=rotation),
                     SuperquadricPrimitive(radii=(.7, .4, 1.), epsilon1=.6, epsilon2=.8, rotation=rotation),
                     AnisotropicGaussianPrimitive(covariance=rotation @ np.diag([.49, .16, 1.]) @ rotation.T)):
            mesh = part.to_mesh_data(16); faces = triangles(mesh)
            directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
            self.assertTrue(np.all(counts == 2))
            np.testing.assert_array_equal(np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)), 0.)
            self.assertGreater(signed_volume(GeometryArrays.capture(mesh.vertices, faces)), 0.)
            center = part.center
            for face in faces[np.any((faces == 0) | (faces == len(mesh.vertices)-1), axis=1)]:
                a,b,c = mesh.vertices[face]
                self.assertGreater(np.dot(np.cross(b-a, c-a), (a+b+c)/3.-center), 0.)

    def test_contact_grouping_does_not_bridge_real_gaps(self):
        a = SimpleNamespace(vertices=np.array([[0.,0.,0.], [1.,1.,1.]]))
        for shift, groups in (([1.,0.,0.], 1), ([1.,1.,0.], 1), ([1.,1.,1.], 1),
                              ([.9,0.,0.], 1), ([1.+1e-10,0.,0.], 2), ([2.,0.,0.], 2)):
            b = SimpleNamespace(vertices=a.vertices+shift)
            self.assertEqual(len(overlap_groups([a,b])), groups)

    def test_output_targets_keep_intended_parts_and_exclude_artist_subtrees(self):
        root = FakeObject('EMPTY'); part = FakeObject(parent=root)
        artist = FakeObject('EMPTY', parent=root, excluded=True); child = FakeObject(parent=artist)
        root.children_recursive = [part, artist, child]
        self.assertEqual([id(o) for o in output_mesh_targets([root, part])], [id(part)])
        self.assertEqual([id(o) for o in output_mesh_targets([root], artist_preview=True)], [id(part), id(child)])

    def test_coarse_projection_preserves_pixel_centers_non_square_crop(self):
        target = target_for_masks({'front': np.zeros((80, 120), bool)})
        c = target.constraints[0]
        points = np.array([[-1., 0., -1.], [1., 0., 1.], [0., 0., 0.], [.2, 0., -.3]])
        original = project_vertices(target,c,points)
        coarse = project_vertices(target,c,points,output_shape=(16,24))
        np.testing.assert_allclose(coarse, (original+.5)*.2-.5)
        direct = replace(c,mask=np.zeros((16,24),bool))
        np.testing.assert_allclose(coarse, project_vertices(target,direct,points))

    def test_low_opacity_cannot_match_target_area_for_admission(self):
        from primitives.soft_silhouette import soft_mask_metrics
        target = np.zeros((8,8)); target[2:6,2:6] = 1.
        result = soft_mask_metrics(target*.1,target)
        self.assertEqual(result['pred_hard_threshold'], .5)
        self.assertEqual(result['pred_hard_area_ratio'], 0.)
        self.assertEqual(result['area_iou_loss'], 1.)
        self.assertEqual(result['diagnostic_adaptive_threshold'], .1)

class ProgramAndProfileConsistencyTests(unittest.TestCase):
    def test_program_local_edits_preserve_pca_frame_and_are_independent(self):
        from scipy.spatial.transform import Rotation
        from reconstruction.program_transforms import local_pose_edit, rotation_matrix
        frame = Rotation.from_rotvec([.2, -.4, .3]).as_matrix()
        params = {'rotation': frame.tolist(), 'x': 2., 'y': 3., 'z': -1.}
        self.assertEqual(local_pose_edit(params, rotation_increment=np.zeros(3)), params)
        translated = local_pose_edit(params, translation=[.1, 0., 0.])
        np.testing.assert_array_equal(rotation_matrix(translated), frame)
        np.testing.assert_allclose([translated[k] for k in ('x','y','z')], [2.,3.,-1.]+frame[:,0]*.1)
        rotated = local_pose_edit(params, rotation_increment=[0.,.05,0.])
        np.testing.assert_allclose(rotation_matrix(rotated), frame @ Rotation.from_rotvec([0.,.05,0.]).as_matrix())
        self.assertEqual([rotated[k] for k in ('x','y','z')], [2.,3.,-1.])

    def test_double_reflection_recovers_pose_and_reflected_support(self):
        from scipy.spatial.transform import Rotation
        from reconstruction.program_transforms import reflect_parameters, rotation_matrix
        frame = Rotation.from_rotvec([.6,-.2,.1]).as_matrix()
        params = {'rotation': frame.tolist(), 'x': 2., 'y': 3., 'z': -1.}
        mirrored = reflect_parameters(params, plane=.4)
        again = reflect_parameters(mirrored, plane=.4)
        np.testing.assert_allclose(rotation_matrix(again), frame)
        np.testing.assert_allclose([again[k] for k in ('x','y','z')], [2.,3.,-1.])
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        original = signs * [.7,.4,1.] @ frame.T + [2.,3.,-1.]
        reflected = original.copy(); reflected[:,0] = .8-reflected[:,0]
        actual = signs * [.7,.4,1.] @ rotation_matrix(mirrored).T + [mirrored[k] for k in ('x','y','z')]
        for point in reflected:
            self.assertLess(np.linalg.norm(actual-point,axis=1).min(), 1e-12)

    def test_program_control_cycle_visits_later_nodes_and_every_axis(self):
        from primitives.shape_program import ShapeNode, ShapeProgram
        from reconstruction.backends.shape_program.geometry_search import parameter_variants
        program = ShapeProgram('shape-program-v1','fixture',tuple(ShapeNode(str(i),'add','box',{}) for i in range(3)))
        visited = []
        for cursor in range(0,54,6):
            variants = parameter_variants(program,limit=6,start_control=cursor)
            visited += [(p.metadata['refinement_control']['node'], p.metadata['refinement_control']['kind'],
                         p.metadata['refinement_control']['axis'], p.metadata['refinement_control']['direction']) for p in variants]
        self.assertEqual(len(set(visited)), 54)
        self.assertIn(('2','size',0,-1), visited[:6])

    def test_compiler_capabilities_fail_closed_before_scene_mutation(self):
        from primitives.shape_program import ShapeNode, ShapeProgram, validate_compilable_program
        from primitives.shape_program_compiler import compile_shape_program
        for node in (ShapeNode('a','add','capsule'), ShapeNode('a','mirror','box'),
                     ShapeNode('a','add','box',children=('child',))):
            program = ShapeProgram('shape-program-v1','p',(node,))
            self.assertTrue(validate_compilable_program(program))
            with self.assertRaisesRegex(ValueError,'cannot compile'):
                compile_shape_program(program)
        good = ShapeProgram('shape-program-v1','p',(ShapeNode('a','add','box'),ShapeNode('b','subtract','sphere')))
        self.assertEqual(validate_compilable_program(good),())

    def test_profile_intervals_cover_full_taper_and_tilted_centerline(self):
        from placement.resfit_initialization import PrimitiveInitializationConfig, initialize_from_profile_bands
        points = np.array([[0.,0.,0.],[.1,.2,.4],[.3,.6,1.2],[.5,1.,2.]])
        entries = [{'center':p, 'radius':r} for p,r in zip(points,[1.,.8,.4,.2])]
        parts = initialize_from_profile_bands(entries,PrimitiveInitializationConfig(primitive_count=2))
        self.assertEqual(len(parts),2)
        np.testing.assert_allclose(parts[0].position-parts[0].get_axis_vector()*parts[0].height/2., points[0],atol=1e-15)
        np.testing.assert_allclose(parts[-1].position+parts[-1].get_axis_vector()*parts[-1].height/2., points[-1],atol=1e-15)
        np.testing.assert_allclose(parts[0].position+parts[0].get_axis_vector()*parts[0].height/2.,
                                   parts[1].position-parts[1].get_axis_vector()*parts[1].height/2.,atol=1e-15)
        self.assertNotEqual(parts[0].radius_bottom,parts[0].radius_top)
        self.assertEqual(initialize_from_profile_bands(entries[:1]),[])

    def test_profile_conversion_keeps_last_knot_and_all_rows(self):
        from placement.resfit.profiles import _profile_rows_to_slices
        from placement.resfit_initialization import PrimitiveInitializationConfig
        rows = [{'view':'front','z_world':float(i), 'width_world':2.,'half_width_world':1.} for i in range(17)]
        slices = _profile_rows_to_slices(rows,PrimitiveInitializationConfig(primitive_count=4))
        self.assertEqual(len(slices),17)
        self.assertEqual(slices[-1]['center'][2],16.)

    def test_unknown_row_values_do_not_become_exact_edges(self):
        from reconstruction.profile_evidence import measure_profile_row
        mask = np.zeros(12,bool); mask[3:9] = True
        valid = np.ones(12,bool); valid[8:] = False
        before = measure_profile_row(mask,valid)
        mask[8:] = ~mask[8:]
        self.assertEqual(measure_profile_row(mask,valid),before)
        self.assertEqual(before['left_edge_px'],3.)
        self.assertIsNone(before['right_edge_px'])
        self.assertIsNone(before['exact_width_px'])
        self.assertEqual(before['width_lower_px'],5.)
        unknown = measure_profile_row(mask,np.zeros(12,bool))
        self.assertFalse(unknown['observed'])

    def test_calibrated_profile_completion_is_unknown_invariant_and_labeled(self):
        from reconstruction.projection_contract import calibrated_profile
        mask = np.zeros((20,20),bool); mask[2:18,5:15] = True
        target = target_for_masks({'front':mask.copy(),'side':mask.copy()})
        valid = np.ones_like(mask); valid[7:12,12:] = False
        front, side = target.constraints
        target = replace(target,constraints=(replace(front,valid_mask=valid),side))
        request = SimpleNamespace(target=target,config={'num_samples':17})
        first = calibrated_profile(request)
        changed = front.mask.copy(); changed[~valid] = ~changed[~valid]
        other = replace(target,constraints=(replace(front,mask=changed,valid_mask=valid),side))
        second = calibrated_profile(SimpleNamespace(target=other,config=request.config))
        np.testing.assert_array_equal(first.rx,second.rx)
        np.testing.assert_array_equal(first.cx,second.cx)
        self.assertIn('censored_completion',first.meta)
        self.assertTrue(any(r['exact_radius_world'] is None for r in first.meta['row_evidence']['front']))

    def test_final_csg_receipt_binds_output_and_never_inherits_operand_status(self):
        from reconstruction.output_qualification import qualify_retained_output
        from unittest.mock import patch
        mesh = EllipsoidPrimitive().to_mesh_data(8)
        data = GeometryArrays.capture(mesh.vertices,triangles(mesh))
        unavailable = qualify_retained_output(data,{})
        self.assertEqual(unavailable['geometry_content_hash'],data.content_hash)
        self.assertFalse(unavailable['boundary_qualified'])
        with patch('reconstruction.native_qualification.qualify_geometry',return_value={'manifold_validated':False,'status':'rejected'}) as qualify:
            receipt = qualify_retained_output(data,{'native_qualification_python':'fixture-python'})
        self.assertFalse(receipt['boundary_qualified'])
        self.assertIs(qualify.call_args.args[0],data)

class PairedProfileTests(unittest.TestCase):
    def test_paired_circular_seed_uses_both_centers(self):
        from placement.resfit.profiles import _profile_rows_to_slices
        from placement.resfit_initialization import PrimitiveInitializationConfig
        rows = [dict(view=view, z_world=float(i), axes=(0,2) if view=='front' else (1,2),
                     width_world=2., half_width_world=1., center_x_world=i*(.1 if view=='front' else .2),
                     world_center=(i*.1,0.,float(i)) if view=='front' else (0.,i*.2,float(i)))
                for view in ('front','side') for i in range(5)]
        slices = _profile_rows_to_slices(rows, PrimitiveInitializationConfig(primitive_count=2))
        np.testing.assert_allclose(slices[-1]['center'], [.4,.8,4.])
        for row in rows:
            if row['view']=='side': row['half_width_world']=.5
        self.assertEqual(_profile_rows_to_slices(rows,PrimitiveInitializationConfig(primitive_count=2)),[])

    def test_censored_profile_rows_are_not_exact_seed_knots(self):
        from placement.resfit.profiles import _profile_rows_to_slices
        from placement.resfit_initialization import PrimitiveInitializationConfig
        rows = [dict(view='front',z_world=float(i),width_world=2.,half_width_world=1.,
                     profile_width_exact=(i!=2)) for i in range(5)]
        slices = _profile_rows_to_slices(rows,PrimitiveInitializationConfig(primitive_count=2))
        self.assertEqual([s['center'][2] for s in slices],[0.,1.,3.,4.])


if __name__ == '__main__': unittest.main()
