"""Actual optional field-edge/opaque-pixel pullbacks; no Blender rendering claims."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

PACKAGE = 'blendslop_implicit_mesh_numeric'
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(Path(__file__).resolve().parent/'reconstruction'/'implicit')]
sys.modules.setdefault(PACKAGE, namespace)
model_module = importlib.import_module(PACKAGE+'.field_model')
mesh_module = importlib.import_module(PACKAGE+'.mesh_objective')
solver = importlib.import_module(PACKAGE+'.numeric_solver')
try:
    import torch
    import dvx.torch as dvx
    import shapely
    from skimage.measure import marching_cubes
    PINNED = torch.__version__ == '2.14.1+cpu' and str(dvx.__version__) == '0.1.1'
except ImportError:
    PINNED = False


def fixture(n=16, anisotropic=False):
    spacing = np.array([1.6, 1.12, .8] if anisotropic else [1.6]*3)/n
    lo = np.array([.23, -.61, .17])
    origin = lo+spacing*.5
    center = lo+spacing*n*.5+spacing*np.array([.13, -.17, .07])
    half = np.array([.24, .22, .13])
    axes = [origin[a]+np.arange(n)*spacing[a] for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    q = np.abs(points-center)-half
    field = (np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)).astype(np.float32)
    return model_module.NarrowBandField.create(field.transpose(2, 1, 0), spacing,
        origin_xyz=origin, band_width=float(spacing.min()*2), maximum_displacement=float(spacing.min()*.9))


def targets(model, n=64):
    lo = model.origin_xyz-model.voxel_size_xyz*.5
    hi = lo+np.array(model.seed_zyx.shape[::-1])*model.voxel_size_xyz
    # Expanded supports strictly contain source and field domain; original
    # pixel spacing may be anisotropic even for a square filtered array.
    return {view: {'square_resolution': n, 'padded_world_bounds':
            [lo[a]-.1, hi[a]+.1, lo[b]-.1, hi[b]+.1],
            'foreground': np.zeros((n, n)), 'valid': np.ones((n, n)), 'weights': np.ones((n, n))}
            for view, (a, b) in {'front': (0, 2), 'top': (0, 1)}.items()}


@unittest.skipUnless(PINNED, 'actual pinned Torch/DVX/Shapely/scikit-image interpreter required')
class ImplicitMeshNumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_16_and_32_anisotropic_edge_carriers_reproduce_authoritative_extraction(self):
        for n in (16, 32):
            model = fixture(n, anisotropic=True)
            parameters = torch.zeros(len(model.active_flat), dtype=torch.float64, requires_grad=True)
            field = model.torch_decode(parameters)
            vertices, faces, report = mesh_module.torch_extracted_zero_mesh(field, model)
            raw, expected_faces, _, _ = marching_cubes(field.detach().numpy().transpose(2, 1, 0),
                level=0., method='lewiner', gradient_direction='descent', allow_degenerate=False)
            expected = model.origin_xyz+raw*model.voxel_size_xyz
            np.testing.assert_array_equal(faces, expected_faces)
            np.testing.assert_allclose(vertices.detach().numpy(), expected, atol=2e-7)
            self.assertLess(report['maximum_world_reconstruction_error'], 2e-7)

    def test_field_edge_pullback_matches_independent_local_parameter_differences(self):
        model = fixture()
        values = np.random.default_rng(414).normal(size=len(model.active_flat))*.01
        parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        def objective(p):
            vertices, faces, _ = mesh_module.torch_extracted_zero_mesh(model.torch_decode(p), model)
            weight = torch.arange(vertices.numel(), dtype=torch.float64).reshape(vertices.shape)%7-3
            return (vertices*weight).sum()
        objective(parameters).backward()
        gradient = parameters.grad.numpy()
        for index in np.argsort(np.abs(gradient))[-8:]:
            plus, minus = values.copy(), values.copy()
            plus[index] += 2e-5; minus[index] -= 2e-5
            numerical = float((objective(torch.tensor(plus))-objective(torch.tensor(minus)))/4e-5)
            np.testing.assert_allclose(gradient[index], numerical, atol=3e-5, rtol=2e-3)

    def test_original_pixel_closed_form_coverage_matches_independent_polygon_cell_intersection(self):
        from shapely import Polygon, union_all, box, intersection, area
        model = fixture(32, anisotropic=True)
        field = model.torch_decode(torch.zeros(len(model.active_flat), dtype=torch.float64))
        rows = targets(model, n=64)
        predictions, report = mesh_module.original_pixel_mesh_predictions(field, model, rows)
        vertices, faces, _ = mesh_module.torch_extracted_zero_mesh(field, model)
        for view, axes in {'front': (0, 2), 'top': (0, 1)}.items():
            row = rows[view];u0, u1, v0, v1 = row['padded_world_bounds']
            xy = vertices.detach().numpy()[:, axes]
            xy = (xy-[u0, v0])/[u1-u0, v1-v0]*64-.5
            triangles = []
            for face in faces:
                a, b = xy[face[1]]-xy[face[0]], xy[face[2]]-xy[face[0]]
                if a[0]*b[1]-a[1]*b[0] != 0:
                    triangles.append(Polygon(xy[face]))
            union = union_all(triangles)
            yy, xx = np.indices((64, 64))
            expected = area(intersection(union, box(xx-.5, yy-.5, xx+.5, yy+.5)))
            np.testing.assert_allclose(predictions[view].numpy(), expected, atol=2e-12)
        self.assertIn('original pixel-cell', report['operator'])

    def test_original_pixel_objective_gradient_matches_local_double_differences(self):
        model = fixture()
        rows = targets(model, n=32)
        values = np.random.default_rng(415).normal(size=len(model.active_flat))*.01
        target = torch.tensor(np.random.default_rng(416).uniform(size=(32, 32)), dtype=torch.float64)
        parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        def objective(p):
            prediction, _ = mesh_module.original_pixel_mesh_predictions(model.torch_decode(p), model, {'top': rows['top']})
            return (prediction['top']-target).square().mean()
        objective(parameters).backward()
        gradient = parameters.grad.numpy()
        for index in np.argsort(np.abs(gradient))[-6:]:
            plus, minus = values.copy(), values.copy()
            plus[index] += 1e-5; minus[index] -= 1e-5
            numerical = float((objective(torch.tensor(plus))-objective(torch.tensor(minus)))/2e-5)
            np.testing.assert_allclose(gradient[index], numerical, atol=1e-7, rtol=1e-3)

    def job(self):
        model = fixture()
        return {'execution_approved': True, 'objective': 'original_pixel_extracted_mesh',
                'seed_field_zyx': model.seed_zyx, 'voxel_size_xyz': model.voxel_size_xyz,
                'origin_xyz': model.origin_xyz, 'band_width': model.band_width,
                'maximum_displacement': model.maximum_displacement, 'steps': 1,
                'timeout_s': 5., 'original_pixel_targets': targets(model, n=32)}

    def test_unknown_original_probability_is_excluded_without_thresholding_observed_data(self):
        first, second = self.job(), self.job()
        for job, value in ((first, np.nan), (second, 1e100)):
            job['steps'] = 0
            for row in job['original_pixel_targets'].values():
                row['valid'][:, :8] = 0.; row['weights'][:, :8] = 0.
                row['foreground'][:, :8] = value
                row['foreground'][:, 8:] = .31
        left, right = solver.fit_implicit_field_job(first), solver.fit_implicit_field_job(second)
        self.assertEqual(left['best_total'], right['best_total'])
        self.assertEqual(left['objective'], 'original_pixel_extracted_mesh')
        self.assertIn('DVX2D', left['runtime']['operator'])
        np.testing.assert_array_equal(left['field_zyx'], right['field_zyx'])

    def test_unsupported_later_mesh_operator_retains_the_last_actual_scored_field(self):
        job = self.job(); job['steps'] = 2
        original = mesh_module.original_pixel_mesh_predictions
        calls = []
        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) > 1:
                raise ValueError('fixture unsupported active extraction topology')
            return original(*args, **kwargs)
        with patch.object(mesh_module, 'original_pixel_mesh_predictions', side_effect=interrupted):
            result = solver.fit_implicit_field_job(job)
        self.assertEqual(result['objective_evaluations'], 1)
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertFalse(result['final_update_evaluated'])
        self.assertEqual(result['stop_reason'], 'extracted_operator_unavailable_after_scored_checkpoint')
        self.assertIn('unsupported active extraction', result['operator_failure'])
        self.assertEqual(result['best_evaluation'], 1)
        np.testing.assert_array_equal(result['field_zyx'], job['seed_field_zyx'])

    def test_empty_feature_weighting_preserves_probabilities_and_uses_only_owned_observed_support(self):
        job = self.job(); job['steps'] = 0; job['known_empty_feature_weight'] = 1.
        for row in job['original_pixel_targets'].values():
            feature = np.zeros_like(row['weights']); feature[12:17, 12:17] = .5
            row['empty_feature_weights'] = feature
            row['foreground'][:] = .23
        result = solver.fit_implicit_field_job(job)
        self.assertEqual(result['known_empty_feature_weight_sum'], 25.)
        self.assertEqual(result['objective_weights']['known_empty_feature'], 1.)
        term = result['term_history'][0]
        expected = term['evidence']+term['known_empty_feature']+.02*term['displacement']+.0001*term['eikonal']
        self.assertAlmostEqual(result['best_total'], expected, places=6)
        invalid = self.job(); invalid['original_pixel_targets']['top']['empty_feature_weights'] = np.full((32,32), 2.)
        with self.assertRaisesRegex(ValueError, 'subset'):
            solver.fit_implicit_field_job(invalid)
        invalid = self.job(); invalid['objective'] = 'minimum_distance_rays'; invalid['known_empty_feature_weight'] = 1.
        with self.assertRaisesRegex(ValueError, 'actual original-pixel'):
            solver.fit_implicit_field_job(invalid)

    def test_unsupported_carriers_and_pixel_allowances_fail_explicitly(self):
        model = fixture()
        field = model.torch_decode(torch.zeros(len(model.active_flat)))
        with self.assertRaisesRegex(ValueError, 'triangle allowance'):
            mesh_module.torch_extracted_zero_mesh(field, model, maximum_faces=1)
        rows = targets(model)
        rows['front']['square_resolution'] = 193
        with self.assertRaisesRegex(ValueError, 'bounded'):
            mesh_module.original_pixel_mesh_predictions(field, model, rows)


if __name__ == '__main__':
    unittest.main()
