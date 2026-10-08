"""Scale/area/overlap contracts for explicitly revised ResFit objectives."""
import unittest
import numpy as np
from primitives.analytic_primitives import EllipsoidPrimitive
from placement.resfit_objective import ResFitObjectiveEvaluator, ResFitLossWeights
from placement.surface_quadrature import surface_area_weights


class NormalizedResFitTests(unittest.TestCase):
    def test_unit_conversion_preserves_all_geometric_terms(self):
        parts=[EllipsoidPrimitive(center=(-.3,0.,0.),radii=(.8,.3,.5)),EllipsoidPrimitive(center=(.5,0.,0.),radii=(.4,.4,.4))]
        points=np.concatenate([p.sample_surface(71) for p in parts])+[.03,0.,0.]
        weights=ResFitLossWeights(visual_hull_occupancy=.1)
        first=ResFitObjectiveEvaluator(points,weights,occupied_points=points[::3],objective_mode='normalized_area_v1',length_scale=2.)(parts)
        scale=100.
        scaled=[EllipsoidPrimitive(center=p.center*scale,radii=p.radii*scale,rotation=p.rotation) for p in parts]
        second=ResFitObjectiveEvaluator(points*scale,weights,occupied_points=points[::3]*scale,objective_mode='normalized_area_v1',length_scale=2.*scale)(scaled)
        for key in first.terms:self.assertAlmostEqual(first.terms[key],second.terms[key],places=12,msg=key)
        self.assertAlmostEqual(first.total,second.total,places=12)

    def test_irrelevant_part_cannot_dilute_existing_overlap(self):
        a=EllipsoidPrimitive();b=EllipsoidPrimitive(center=(.5,0.,0.));far=EllipsoidPrimitive(center=(20.,0.,0.))
        objective=ResFitObjectiveEvaluator(a.sample_surface(71),ResFitLossWeights(),objective_mode='normalized_area_v1',length_scale=2.)
        self.assertGreater(objective([a,b]).terms['overlap_penalty'],0.)
        self.assertAlmostEqual(objective([a,b]).terms['overlap_penalty'],objective([a,b,far]).terms['overlap_penalty'],places=12)

    def test_anisotropic_area_weights_are_not_equal_sample_counts(self):
        part=EllipsoidPrimitive(radii=(2.,.3,.7));samples=part.sample_surface(64)
        weights=surface_area_weights(part,samples)
        self.assertGreater(np.max(weights)-np.min(weights),.01)
        np.testing.assert_allclose(part.sdf_batch(samples),0.,atol=1e-14)
        scaled=EllipsoidPrimitive(radii=part.radii*7.)
        np.testing.assert_allclose(surface_area_weights(scaled,samples*7.),weights*49.,atol=1e-12)

    def test_legacy_mode_remains_available_and_labeled(self):
        part=EllipsoidPrimitive(radii=(2.,1.,1.));points=part.sample_surface(71)
        legacy=ResFitObjectiveEvaluator(points)
        revised=ResFitObjectiveEvaluator(points,objective_mode='normalized_area_v1')
        self.assertEqual(legacy.objective_mode,'legacy_world_squared')
        self.assertEqual(revised.objective_mode,'normalized_area_v1')
        with self.assertRaises(ValueError):ResFitObjectiveEvaluator(points,objective_mode='unknown')

class CoupledAndPriorityTests(unittest.TestCase):
    def test_residual_vector_exactly_matches_complete_objective(self):
        parts=[EllipsoidPrimitive(center=(-.3,0.,0.),radii=(.8,.3,.5)),EllipsoidPrimitive(center=(.5,0.,0.),radii=(.4,.4,.4))]
        points=np.concatenate([p.sample_surface(45) for p in parts])+[.03,0.,0.]
        weights=ResFitLossWeights(visual_hull_occupancy=.1,silhouette=.2,uncertainty_penalty=.1)
        for mode in ('legacy_world_squared','normalized_area_v1'):
            objective=ResFitObjectiveEvaluator(points,weights,occupied_points=points[::3],
                silhouette_hook=lambda p:{'constant_fixture':.3},uncertainty_penalty_hook=lambda p:.2,
                objective_mode=mode,length_scale=2.)
            scored=objective(parts);vector=objective.residual_vector(parts,evaluated=scored)
            self.assertAlmostEqual(vector@vector,scored.total,places=12)
            size=len(vector);parts[0].center[0]+=.02
            self.assertEqual(len(objective.residual_vector(parts)),size)

    def test_coupled_block_improves_fixture_and_honors_actual_evaluation_count(self):
        from placement.resfit_coupled import coupled_block_optimize
        from placement.resfit_optimizer import CoordinateDescentConfig
        truth=EllipsoidPrimitive(radii=(.8,.4,.6));seed=EllipsoidPrimitive(center=(.04,-.03,.02),radii=(.84,.42,.63))
        weights=ResFitLossWeights(primitive_count=0.,overlap_penalty=0.,constraint_penalty=0.)
        objective=ResFitObjectiveEvaluator(truth.sample_surface(64),weights,objective_mode='normalized_area_v1',length_scale=2.)
        result=coupled_block_optimize([seed],objective,CoordinateDescentConfig(iterations=2,max_objective_evaluations=50,max_elapsed_s=2.))
        self.assertLess(result.final_result.total,result.initial_result.total)
        self.assertLessEqual(result.objective_evaluations,50)
        self.assertTrue(result.parameter_visits)

    def test_coupled_budget_retains_seed_without_fabricated_visits(self):
        from placement.resfit_coupled import coupled_block_optimize
        from placement.resfit_optimizer import CoordinateDescentConfig
        part=EllipsoidPrimitive();objective=ResFitObjectiveEvaluator(part.sample_surface(32))
        result=coupled_block_optimize([part],objective,CoordinateDescentConfig(iterations=3,max_objective_evaluations=1))
        self.assertEqual(result.objective_evaluations,1)
        self.assertEqual(result.parameter_visits,())
        self.assertEqual(result.final_result.total,result.initial_result.total)

    def test_changed_part_controls_come_first_under_a_tiny_allowance(self):
        from placement.resfit_optimizer import coordinate_descent_optimize,CoordinateDescentConfig
        parts=[EllipsoidPrimitive(center=(-2.,0.,0.)),EllipsoidPrimitive(center=(2.,0.,0.))]
        objective=ResFitObjectiveEvaluator(np.concatenate([p.sample_surface(32) for p in parts]))
        result=coordinate_descent_optimize(parts,objective,CoordinateDescentConfig(iterations=1,max_objective_evaluations=3),priority_parts=(1,))
        self.assertEqual(result.parameter_visits[0][0],1)

    def test_extent_constraint_responds_to_size_and_has_no_input_constant(self):
        from placement.resfit.penalties import _build_constraint_penalty_hook
        from reconstruction.types import Bounds3D
        bounds=Bounds3D.from_min_max((-1.,)*3,(1.,)*3)
        hook=_build_constraint_penalty_hook(constraint_signal={'score':.1,'constraint_count':20},bounds=bounds,geometry_dependent=True)
        small=EllipsoidPrimitive(radii=(.5,)*3);large=EllipsoidPrimitive(radii=(2.,)*3)
        self.assertEqual(hook([small]),0.)
        self.assertGreater(hook([large]),0.)

class DistinctSeedScreenTests(unittest.TestCase):
    def test_tiny_budget_retains_a_scored_whole_seed(self):
        from placement.resfit.optimizer import fit_residual_primitives_multistart
        from placement.resfit.config import ResFitPipelineConfig
        from placement.resfit_initialization import PrimitiveInitializationConfig
        from placement.resfit_optimizer import CoordinateDescentConfig,OptimizationBudget
        part=EllipsoidPrimitive(radii=(.8,.4,.6));points=part.sample_surface(64)
        class NoJobs:
            def map(self,jobs,**kwargs):
                self.jobs=jobs
                return []
        pool=NoJobs();budget=OptimizationBudget(max_objective_evaluations=1)
        result=fit_residual_primitives_multistart(points,ResFitPipelineConfig(primitive_family='ellipsoid',
            initialization=PrimitiveInitializationConfig(primitive_count=2),
            optimizer=CoordinateDescentConfig(iterations=0,max_objective_evaluations=1)),
            whole_primitives=(part,),max_attempts=3,budget=budget,executor=pool)
        self.assertEqual(result.selected_attempt,'whole_oriented_seed')
        self.assertEqual(result.objective_evaluations,1)
        self.assertEqual(len(result.primitives),1)
        self.assertEqual(pool.jobs,[])

    def test_seed_screen_deduplicates_identical_profile_geometry(self):
        from placement.resfit.optimizer import fit_residual_primitives_multistart,fit_residual_primitives
        from placement.resfit.config import ResFitPipelineConfig
        from placement.resfit_initialization import PrimitiveInitializationConfig
        from placement.resfit_optimizer import CoordinateDescentConfig
        from types import SimpleNamespace
        part=EllipsoidPrimitive(radii=(.8,.4,.6));points=part.sample_surface(64)
        class InlineFixture:
            def map(self,jobs,**kwargs):
                return [SimpleNamespace(status='success',partial=False,value=fit_residual_primitives(**payload),
                    elapsed_s=.01,queue_s=0.,total_wall_s=.01) for kind,payload,allowance in jobs]
        config=ResFitPipelineConfig(primitive_family='ellipsoid',initialization=PrimitiveInitializationConfig(primitive_count=2),
            optimizer=CoordinateDescentConfig(iterations=0,max_objective_evaluations=8))
        result=fit_residual_primitives_multistart(points,config,whole_primitives=(part,),profile_primitives=(part,),
                                                max_attempts=3,executor=InlineFixture())
        self.assertTrue(any(row.get('reason')=='duplicate_geometry_seed' for row in result.attempts))
        self.assertLessEqual(result.objective_evaluations,8)


if __name__=='__main__':unittest.main()
