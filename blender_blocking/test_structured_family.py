"""Observed annulus and planar-negative-space proposal regressions."""
import unittest
import numpy as np

from reconstruction.frozen_family import observed_target, fitted_family_program
from test_frozen_family import cameras
from reconstruction.structured_family import structured_family_program


class StructuredFamilyTests(unittest.TestCase):
    def torus_observations(self, size=192):
        scale, major, minor = 3., .69, .21
        y, x = np.mgrid[:size, :size]
        u, v = (x+.5-size/2)*scale/size, (size/2-y-.5)*scale/size
        fields = {"top": np.abs(np.hypot(u, v)-major)-minor,
                  "front": np.hypot(np.maximum(np.abs(u)-major, 0), v)-minor,
                  "side": np.hypot(np.maximum(np.abs(u)-major, 0), v)-minor}
        coverages = {view: np.clip(.5-field/(scale/size), 0., 1.) for view, field in fields.items()}
        masks = {view: coverage >= .5 for view, coverage in coverages.items()}
        return masks, coverages, major, minor

    def test_torus_recovers_both_observed_ring_boundaries_and_depth(self):
        masks, coverages, major, minor = self.torus_observations()
        records = cameras()
        target = observed_target(masks, records, coverage_masks=coverages)
        program = fitted_family_program("torus", target, masks, records, coverage_masks=coverages,
                                        max_evaluations=160, max_elapsed_s=3.)
        parameters = program.root_nodes[0].parameters
        self.assertEqual(program.root_nodes[0].primitive_type, "torus")
        self.assertAlmostEqual(parameters["major_radius"], major, delta=.002)
        self.assertAlmostEqual(parameters["minor_radius"], minor, delta=.002)
        self.assertGreater(program.metadata["observed_hole_pixels"], 4)
        self.assertGreater(program.metadata["inner_contour_samples"], 8)
        self.assertLessEqual(program.metadata["support_evaluations"], 160)

    def test_direct_and_routed_proposals_reject_invalid_budgets_before_work(self):
        masks, coverages, _, _ = self.torus_observations(size=64)
        records = cameras()
        target = observed_target(masks, records, coverage_masks=coverages)
        invalid = [(True, 3.), (False, 3.), (1.0, 3.), (1.5, 3.), (0, 3.),
                   (513, 3.), (256, 0.), (256, -1.), (256, 10.1),
                   (256, float("nan")), (256, float("inf")), (256, True)]
        for entry in (structured_family_program, fitted_family_program):
            for family in ("torus", "concave_arch"):
                for evaluations, elapsed in invalid:
                    with self.subTest(entry=entry.__name__, family=family,
                                      evaluations=evaluations, elapsed=elapsed):
                        with self.assertRaisesRegex(ValueError, "bounded evaluation/time"):
                            entry(family, target, {}, {}, max_evaluations=evaluations,
                                  max_elapsed_s=elapsed)
        # Endpoints are legal allowances, although this deliberately missing input
        # still prevents a proposal. Budget validation must not reject them.
        with self.assertRaisesRegex(ValueError, "explicit complete top coverage"):
            structured_family_program("torus", target, {}, {}, max_evaluations=np.int64(512),
                                      max_elapsed_s=10.)

    def test_direct_torus_refuses_rotated_top_basis_even_with_canonical_target(self):
        masks, coverages, _, _ = self.torus_observations(size=64)
        records = cameras()
        target = observed_target(masks, records, coverage_masks=coverages)
        matrix = np.asarray(records["top"]["matrix_world"], float)
        matrix[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        changed = {**records, "top": {**records["top"], "matrix_world": matrix.tolist()}}
        with self.assertRaisesRegex(ValueError, "canonical top camera X/Y basis"):
            structured_family_program("torus", target, masks, changed, coverage_masks=coverages)
        changed["top"]["ortho_scale"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite proper orthographic camera"):
            structured_family_program("torus", target, masks, changed, coverage_masks=coverages)

    def test_filled_ring_or_missing_coverage_cannot_seed_torus(self):
        masks, coverages, _, _ = self.torus_observations()
        target = observed_target(masks, cameras(), coverage_masks=coverages)
        with self.assertRaisesRegex(ValueError, "explicit"):
            fitted_family_program("torus", target, masks, cameras())
        from scipy.ndimage import binary_fill_holes
        filled = {**coverages, "top": binary_fill_holes(masks["top"]).astype(float)}
        with self.assertRaisesRegex(ValueError, "observed hole"):
            fitted_family_program("torus", target, masks, cameras(), coverage_masks=filled)

    def test_arch_uses_eight_corner_exterior_and_measured_thickness(self):
        size, scale = 96, 3.
        y, x = np.mgrid[:size, :size]
        u, v = (x+.5-size/2)*scale/size, (size/2-y-.5)*scale/size
        masks = {"front": (np.abs(u) < 1.1) & (np.abs(v) < 1.) & ~((np.abs(u) < .6) & (v < .4)),
                 "side": (np.abs(u) < .22) & (np.abs(v) < 1.),
                 "top": (np.abs(u) < 1.1) & (np.abs(v) < .22)}
        coverages = {view: mask.astype(float) for view, mask in masks.items()}
        records = cameras()
        target = observed_target(masks, records, coverage_masks=coverages)
        program = fitted_family_program("concave_arch", target, masks, records, coverage_masks=coverages)
        node = program.root_nodes[0]
        self.assertEqual(node.primitive_type, "polygon_extrusion")
        self.assertEqual(len(node.parameters["outer"]), 8)
        self.assertEqual(node.parameters["height_world"], target.bounds.size[1])
        self.assertTrue(node.parameters["thickness_source"].startswith("other observed"))
        # Native-free cap triangulation preserves the exterior negative space.
        from primitives.polygon_extrusion import PolygonExtrusionPrimitive
        part = PolygonExtrusionPrimitive(node.parameters["outer"], node.parameters["holes"],
                                         height=node.parameters["height_world"])
        self.assertGreater(float(part.sdf_batch(np.array([[0., -.7, 0.]]))[0]), 0.)
        self.assertLess(float(part.sdf_batch(np.array([[.8, 0., 0.]]))[0]), 0.)

    def test_filled_arch_observation_stays_a_filled_proposal_until_features_reject(self):
        size = 64
        coverage = np.zeros((size, size)); coverage[12:52, 10:54] = 1.
        thin = np.zeros_like(coverage); thin[12:52, 27:37] = 1.
        top = np.zeros_like(coverage); top[27:37, 10:54] = 1.
        coverages = {"front": coverage, "side": thin, "top": top}
        masks = {view: c >= .5 for view, c in coverages.items()}
        target = observed_target(masks, cameras(), coverage_masks=coverages)
        program = fitted_family_program("concave_arch", target, masks, cameras(), coverage_masks=coverages)
        self.assertEqual(len(program.root_nodes[0].parameters["outer"]), 4)
        self.assertNotIn("reference_geometry", program.metadata)
        self.assertEqual(program.metadata["full_five_view_admission"].split(';')[0], "unrun")


if __name__ == "__main__":
    unittest.main()
