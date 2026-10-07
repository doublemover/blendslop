from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'blender_blocking')]
from blender_blocking.verify_setup import configure_dependency_paths
configure_dependency_paths()

class ReliabilityTests(unittest.TestCase):
    def test_all_failed_never_selected(self):
        from reconstruction.candidate_scoring import select_best
        from reconstruction.types import CandidateResult,CandidateMetrics
        broken=CandidateResult('broken','stub','failed',metric_result=CandidateMetrics(area_iou_mean=1.))
        for policy in ['best_score','fast_preview','quality_first','editability_first','pareto']:
            self.assertIsNone(select_best([broken],policy=policy)[0])
    def test_gaussian_saved_covariance_replays_exact_mesh(self):
        from primitives.analytic_primitives import AnisotropicGaussianPrimitive
        rotation = np.linalg.qr(np.array([[1., 2., 3.], [2., 5., 1.], [3., 1., 4.]]))[0]
        primitive = AnisotropicGaussianPrimitive(covariance=rotation @ np.diag([.1, .1, .4]) @ rotation.T)
        restored = AnisotropicGaussianPrimitive.from_dict(primitive.to_dict())
        np.testing.assert_array_equal(restored.covariance, primitive.covariance)
        np.testing.assert_array_equal(restored.to_mesh_data(20).vertices, primitive.to_mesh_data(20).vertices)
        repaired = AnisotropicGaussianPrimitive.from_dict({"center": [0, 0, 0], "covariance": np.diag([-1., .1, .2])})
        self.assertGreaterEqual(np.linalg.eigvalsh(repaired.covariance).min(), 1e-8 - 1e-12)

    def test_shared_kmeans_keeps_height_default(self):
        from placement.resfit_initialization import deterministic_kmeans
        points = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 1], [10, 0, 1]], float)
        centers, labels = deterministic_kmeans(points, 2, iterations=1)
        np.testing.assert_array_equal(centers, [[0, 0, .5], [10, 0, .5]])
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])
        spread, _ = deterministic_kmeans(points, 2, iterations=1, seed_strategy="farthest")
        self.assertTrue(np.isfinite(spread).all())
        with self.assertRaises(ValueError):
            deterministic_kmeans(points, 2, seed_strategy="unknown")

    def test_default_mask_preserves_negative_space_and_components(self):
        import cv2
        from validation.silhouette_iou import mask_from_image_array
        expected = np.zeros((96, 96), np.uint8)
        cv2.circle(expected, (32, 48), 24, 1, -1)
        cv2.circle(expected, (32, 48), 12, 0, -1)
        cv2.rectangle(expected, (72, 40), (88, 56), 1, -1)
        image = np.repeat((expected * 255)[:, :, None], 4, axis=2)
        np.testing.assert_array_equal(mask_from_image_array(image), expected.astype(bool))

    def test_union_objective_sees_both_components(self):
        import cv2
        from dataclasses import replace
        from reconstruction.types import Bounds2D,Bounds3D,CandidateRequest
        from reconstruction.targets import make_reconstruction_target
        from placement.resfit.silhouette_union import mesh_union_silhouette_hook
        from primitives.analytic_primitives import EllipsoidPrimitive
        mask=np.zeros((64,64),np.uint8)
        cv2.circle(mask,(20,32),8,1,-1);cv2.circle(mask,(44,32),8,1,-1)
        t=make_reconstruction_target(masks_by_view={'front':mask},bounds=Bounds3D(-1,1,-1,1,-1,1))
        c=replace(t.constraints[0],camera=replace(t.constraints[0].camera,bounds=Bounds2D(-1,-1,1,1)))
        t=replace(t,constraints=(c,))
        hook=mesh_union_silhouette_hook(t)
        parts=[EllipsoidPrimitive(center=(-.36,0.,-.0156),radii=(.25,.25,.25)),EllipsoidPrimitive(center=(.39,0.,-.0156),radii=(.25,.25,.25))]
        self.assertLess(hook(parts)['mesh_union'],hook(parts[:1])['mesh_union']-.3)
    def test_default_export_is_the_fitted_geometry(self):
        from reconstruction.differentiable.config import _normalize_differentiable_config
        from reconstruction.differentiable.candidate_adapter import _scale_primitives_for_mesh_export
        from primitives.analytic_primitives import EllipsoidPrimitive
        # This detects the former unconditional 1.6x enlargement after fitting.
        normalized, errors, _ = _normalize_differentiable_config({})
        self.assertFalse(errors)
        primitive = EllipsoidPrimitive(center=(0.2, -0.3, 0.4), radii=(0.2, 0.3, 0.4))
        exported = _scale_primitives_for_mesh_export([primitive], scale=normalized['mesh_proxy_scale'])[0]
        np.testing.assert_array_equal(primitive.to_mesh_data(12).vertices, exported.to_mesh_data(12).vertices)

    def test_duplicate_surfaces_and_reversed_winding(self):
        from evaluation.solid_validity import solid_validity_report
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        clean=solid_validity_report(vertices,faces)
        self.assertEqual(clean['inconsistent_winding_edges'],0)
        duplicated=solid_validity_report(vertices,np.vstack([faces,faces[:1]]))
        self.assertEqual(duplicated['duplicate_geometric_faces'],1)
        faces[0]=faces[0,::-1]
        self.assertGreater(solid_validity_report(vertices,faces)['inconsistent_winding_edges'],0)

if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
