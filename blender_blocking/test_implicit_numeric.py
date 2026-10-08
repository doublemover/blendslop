"""Optional pinned-CPU implicit-field numerical fixtures, independent of Blender."""
import importlib
import itertools
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# The owned numeric runtime deliberately does not import the unrelated GUI,
# OpenCV or reconstruction registry. Only this repository's numeric modules load.
PACKAGE = 'blendslop_implicit_numeric'
if PACKAGE not in sys.modules:
    namespace = types.ModuleType(PACKAGE)
    namespace.__path__ = [str(Path(__file__).resolve().parent/'reconstruction'/'implicit')]
    sys.modules[PACKAGE] = namespace
model_module = importlib.import_module(PACKAGE+'.field_model')
solver = importlib.import_module(PACKAGE+'.numeric_solver')
NarrowBandField = model_module.NarrowBandField

print('implicit numeric stage=optional Torch runtime probe', flush=True)
try:
    import torch
    PINNED_CPU = torch.__version__ == '2.14.1+cpu' and torch.version.cuda is None and torch.version.hip is None
except ImportError:
    torch = None
    PINNED_CPU = False


def fixture():
    axes = -.8+(np.arange(16)+.5)*.1
    points = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1)
    q = np.abs(points)-[.55, .55, .15]
    phi = (np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(q.max(axis=-1), 0)).astype(np.float32)
    return phi.transpose(2, 1, 0), np.full(3, .1), np.full(3, axes[0])


def model():
    phi, spacing, origin = fixture()
    return NarrowBandField.create(phi, spacing, origin_xyz=origin,
                                 band_width=.2, maximum_displacement=.18)


def payload():
    phi, spacing, origin = fixture()
    target = np.zeros((16, 16), float)
    target[3:13, 3:13] = 1.
    return {'execution_approved': True, 'seed_field_zyx': phi, 'voxel_size_xyz': spacing,
            'origin_xyz': origin, 'band_width': .2, 'maximum_displacement': .18,
            'steps': 2, 'timeout_s': 5., 'ray_targets': {'top': {'foreground': target}},
            'eikonal_weight': .0001}


class ImplicitExecutionGuardTests(unittest.TestCase):
    def test_numerical_execution_is_lazy_and_requires_explicit_approval(self):
        with self.assertRaises(PermissionError):
            solver.fit_implicit_field_job({})


@unittest.skipUnless(PINNED_CPU, 'actual pinned Torch2.14.1+cpu runtime is not present in this interpreter')
class ImplicitNumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_torch_field_matches_reference_and_active_pullback(self):
        m = model()
        rng = np.random.default_rng(91)
        values = rng.normal(size=len(m.active_flat))*.2
        parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        field = m.torch_decode(parameters)
        np.testing.assert_allclose(field.detach().numpy(), m.decode(values), atol=2e-16)
        weight = torch.tensor(rng.normal(size=m.seed_zyx.shape), dtype=torch.float64)
        loss = (field*weight).sum()
        loss.backward()
        expected = m.maximum_displacement*(1-np.tanh(values)**2)*weight.numpy().ravel()[m.active_flat]
        # Protected clamps are separate hard constraints; verify their active branch.
        unclipped = m.seed_zyx.ravel()[m.active_flat]+m.maximum_displacement*np.tanh(values)
        protected = m.protected_empty_zyx.ravel()[m.active_flat]
        expected = np.where(protected & (unclipped < m.minimum_empty_distance), 0, expected)
        np.testing.assert_allclose(parameters.grad.numpy(), expected, atol=1e-15)

    def test_protected_empty_coordinates_have_a_feasible_inward_gradient(self):
        phi, spacing, origin = fixture()
        known = np.zeros_like(phi, bool)
        known[:, 7:9, 7:9] = True
        m = NarrowBandField.create(phi, spacing, origin_xyz=origin, band_width=.2,
                                   maximum_displacement=.18, known_empty_zyx=known)
        parameters = torch.zeros(len(m.active_flat), dtype=torch.float64, requires_grad=True)
        eligible = np.flatnonzero(known.ravel()[m.active_flat] & (phi.ravel()[m.active_flat] < 0))
        index = int(eligible[0])
        flat = m.active_flat[index]
        value = m.torch_decode(parameters).reshape(-1)[flat]
        self.assertAlmostEqual(float(value.detach()), m.minimum_empty_distance, places=13)
        value.backward()
        self.assertGreater(float(parameters.grad[index]), 1e-5)
        values = np.zeros(len(m.active_flat))
        values[index] = 1e-5
        derivative = (m.decode(values).ravel()[flat]-m.decode(np.zeros_like(values)).ravel()[flat])/1e-5
        self.assertAlmostEqual(float(parameters.grad[index]), derivative, places=8)
        values[index] = -1
        self.assertGreaterEqual(m.decode(values).ravel()[flat], m.minimum_empty_distance)
        self.assertLessEqual(float(np.max(np.abs(m.decode(values)-phi))), .18+1e-8)

    def test_ray_and_regularizer_backward_matches_local_double_differences(self):
        m = model()
        rng = np.random.default_rng(93)
        values = rng.normal(size=len(m.active_flat))*.03
        parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        target = torch.tensor(rng.uniform(size=(16, 16)), dtype=torch.float64)
        def objective(p):
            field = m.torch_decode(p)
            ray = solver.field_ray_predictions(field, .03)['top']
            displacement, eikonal = solver.field_regularizers(field, m)
            return (ray-target).square().mean()+.02*displacement+.0001*eikonal
        objective(parameters).backward()
        gradient = parameters.grad.numpy()
        indices = np.argsort(np.abs(gradient))[-16:]
        for index in indices:
            plus, minus = values.copy(), values.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical = float((objective(torch.tensor(plus))-objective(torch.tensor(minus)))/2e-6)
            np.testing.assert_allclose(gradient[index], numerical, atol=2e-8, rtol=2e-5)

    def test_two_updates_keep_an_evaluated_bounded_checkpoint_and_fixed_transform(self):
        job = payload()
        result = solver.fit_implicit_field_job(job)
        self.assertEqual(result['optimizer_updates'], 2)
        self.assertEqual(result['objective_evaluations'], 3)
        self.assertTrue(result['final_update_evaluated'])
        self.assertLessEqual(result['best_total'], result['history'][0])
        self.assertEqual(result['best_total'], min(result['history']))
        self.assertEqual(result['model']['origin_xyz'], list(job['origin_xyz']))
        self.assertLessEqual(float(np.abs(result['field_zyx']-job['seed_field_zyx']).max()), .18+1e-6)
        self.assertFalse(result['native_qualification'])
        self.assertFalse(result['automatic_admission'])
        self.assertIn('DVX is not used', result['runtime']['operator'])
        m = model()
        replay = m.torch_decode(torch.tensor(result['parameters'])).detach().numpy()
        np.testing.assert_array_equal(replay, result['field_zyx'])

    def test_unknown_pixel_values_do_not_enter_observed_ray_objective(self):
        first, second = payload(), payload()
        valid = np.ones((16, 16))
        valid[:, :4] = 0
        for job, value in ((first, np.nan), (second, 1e100)):
            foreground = job['ray_targets']['top']['foreground'].copy()
            foreground[:, :4] = value
            job['ray_targets']['top'] = {'foreground': foreground, 'valid': valid}
            job['steps'] = 0
        left, right = solver.fit_implicit_field_job(first), solver.fit_implicit_field_job(second)
        self.assertEqual(left['best_total'], right['best_total'])
        np.testing.assert_array_equal(left['field_zyx'], right['field_zyx'])

    def test_partial_timeout_does_not_return_unscored_final_update(self):
        job = payload()
        job['timeout_s'] = .01
        clock = itertools.count(0, .005)
        with patch.object(solver, 'time', types.SimpleNamespace(perf_counter=lambda: next(clock))):
            result = solver.fit_implicit_field_job(job)
        self.assertEqual(result['stop_reason'], 'elapsed_time_budget')
        self.assertEqual(result['optimizer_updates'], 1)
        self.assertEqual(result['objective_evaluations'], 1)
        self.assertFalse(result['final_update_evaluated'])
        np.testing.assert_array_equal(result['field_zyx'], job['seed_field_zyx'])

    def test_owned_progress_artifact_contains_only_a_scored_field_snapshot(self):
        import tempfile, pickle, hashlib
        with tempfile.TemporaryDirectory() as root:
            progress = Path(root)/'progress.pkl'
            job = payload()
            job['progress_paths'] = [str(progress)]
            result = solver.fit_implicit_field_job(job)
            with progress.open('rb') as stream:
                saved = pickle.load(stream)['value']
            self.assertEqual(saved['best_evaluation'], result['best_evaluation'])
            self.assertEqual(saved['best_total'], result['best_total'])
            np.testing.assert_array_equal(saved['field_zyx'], result['field_zyx'])
            self.assertEqual(saved['retained_field_hash'], hashlib.sha256(
                saved['model']['constraint_hash'].encode()+saved['field_zyx'].tobytes()).hexdigest())
            self.assertFalse(saved['automatic_admission'])

    def test_bounded_checkpoint_pool_contains_only_evaluated_replayable_fields(self):
        import hashlib
        job=payload();job.update(steps=3,retain_scored_checkpoints=True)
        result=solver.fit_implicit_field_job(job)
        self.assertEqual(len(result['scored_checkpoints']),4)
        model=NarrowBandField.create(job['seed_field_zyx'],job['voxel_size_xyz'],origin_xyz=job['origin_xyz'],
            band_width=job['band_width'],maximum_displacement=job['maximum_displacement'])
        for index,row in enumerate(result['scored_checkpoints'],1):
            self.assertEqual(row['best_evaluation'],index)
            self.assertTrue(np.isfinite(row['best_total']))
            np.testing.assert_allclose(model.decode(row['parameters']),row['field_zyx'],atol=1e-7,rtol=0.)
            self.assertEqual(row['retained_field_hash'],hashlib.sha256(
                result['model']['constraint_hash'].encode()+row['field_zyx'].tobytes()).hexdigest())
        self.assertEqual(result['best_total'],min(row['best_total'] for row in result['scored_checkpoints']))
        with self.assertRaises(ValueError):
            solver.fit_implicit_field_job({**job,'retain_scored_checkpoints':'yes'})

    def test_partial_timeout_pool_excludes_unscored_last_update(self):
        job=payload();job.update(timeout_s=.01,retain_scored_checkpoints=True)
        clock=itertools.count(0,.005)
        with patch.object(solver,'time',types.SimpleNamespace(perf_counter=lambda:next(clock))):
            result=solver.fit_implicit_field_job(job)
        self.assertEqual(result['optimizer_updates'],1)
        self.assertEqual(len(result['scored_checkpoints']),1)
        np.testing.assert_array_equal(result['scored_checkpoints'][0]['field_zyx'],job['seed_field_zyx'])

    def test_no_observed_rays_or_unsupported_camera_fail_before_optimization(self):
        job = payload()
        job['ray_targets']['top']['valid'] = np.zeros((16, 16))
        with self.assertRaisesRegex(ValueError, 'no positive observed-ray'):
            solver.fit_implicit_field_job(job)
        job = payload()
        job['ray_targets'] = {'oblique': job['ray_targets']['top']}
        with self.assertRaisesRegex(ValueError, 'canonical'):
            solver.fit_implicit_field_job(job)


if __name__ == '__main__':
    unittest.main(verbosity=2)
