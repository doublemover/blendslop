"""Focused frozen family fits and evidence isolation; no Blender required."""
import unittest
import numpy as np

from reconstruction.frozen_family import (
    SUPPORTED_FAMILIES, camera_support_evidence, coverage_contour_points,
    fitted_family_program, initial_family_rows, observed_target,
    rounded_box_support, _fit_rounded_box,
)
from reconstruction.oriented_support import SupportEvidence


def cameras(scale=3.):
    frames = {"front": np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
              "side": np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
              "top": np.eye(3)}
    result = {}
    for view, frame in frames.items():
        matrix = np.eye(4)
        matrix[:3, :3] = frame
        result[view] = {"matrix_world": matrix.tolist(), "ortho_scale": scale}
    return result


class FrozenFamilyTests(unittest.TestCase):
    def test_contour_has_pixel_cell_edges_without_half_pixel_bias(self):
        coverage = np.zeros((12, 12))
        coverage[3:9, 2:10] = 1.
        points = coverage_contour_points(coverage)
        np.testing.assert_allclose(points.min(axis=0), [2., 3.])
        np.testing.assert_allclose(points.max(axis=0), [10., 9.])
        coverage[0, 4] = 1.
        with self.assertRaisesRegex(ValueError, "clipped"):
            coverage_contour_points(coverage)

    def test_calibrated_sphere_fit_uses_observed_coverage(self):
        size, scale, radius = 192, 3., .73
        y, x = np.mgrid[:size, :size]
        distance = np.hypot((x+.5-size/2)*scale/size, (y+.5-size/2)*scale/size)
        coverage = np.clip(.5+(radius-distance)/(scale/size), 0., 1.)
        coverages = {v: coverage.copy() for v in cameras()}
        masks = {v: a >= .5 for v, a in coverages.items()}
        target = observed_target(masks, cameras(), coverage_masks=coverages)
        program = fitted_family_program("sphere", target, masks, cameras(),
                                        coverage_masks=coverages, max_evaluations=128)
        params = program.root_nodes[0].parameters
        np.testing.assert_allclose([params[k]/2 for k in
                                    ("width_world", "depth_world", "height_world")], radius, atol=.003)
        self.assertEqual(program.metadata["coverage_views"], ["front", "side", "top"])
        self.assertLessEqual(program.metadata["support_evaluations"], 128)
        self.assertNotIn("reference_geometry", program.to_dict())
        self.assertNotIn("fixture_parameters", program.to_dict())

    def test_rounded_box_recovers_support_without_fixture_recipe(self):
        directions = np.random.default_rng(3).normal(size=(144, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        center, sizes, radius = np.array([.1, -.2, .3]), np.array([.9, .5, .7]), .13
        values = rounded_box_support(directions, center, sizes, radius)
        evidence = SupportEvidence(directions, values, np.zeros(len(values)),
                                   np.zeros(len(values), bool), np.ones(len(values)), 2.)
        fit = _fit_rounded_box(evidence, center+.03, sizes*1.08, 256, 3.)
        np.testing.assert_allclose(fit["center"], center, atol=1e-6)
        np.testing.assert_allclose(fit["dimensions"], sizes, atol=1e-6)
        self.assertAlmostEqual(fit["radius"], radius, places=5)
        self.assertLessEqual(fit["support_evaluations"], 256)

    def test_camera_translation_changes_world_support_once(self):
        coverage = np.zeros((16, 16)); coverage[4:12, 4:12] = 1.
        masks = {v: coverage >= .5 for v in cameras()}
        original = cameras()
        target = observed_target(masks, original, coverage_masks={v: coverage for v in masks})
        first = camera_support_evidence(masks, original, target.bounds,
                                        coverage_masks={v: coverage for v in masks})
        displacement = np.array([.4, -.3, .2])
        shifted = cameras()
        for record in shifted.values():
            matrix = np.asarray(record["matrix_world"])
            matrix[:3, 3] += displacement
            record["matrix_world"] = matrix.tolist()
        second = camera_support_evidence(masks, shifted, target.bounds,
                                         coverage_masks={v: coverage for v in masks})
        np.testing.assert_allclose(second.values-first.values,
                                   first.directions @ displacement, atol=1e-14)

    def test_unsupported_holes_and_camera_mismatch_are_explicit(self):
        mask = np.zeros((16, 16), bool); mask[3:13, 3:13] = True
        masks = {v: mask.copy() for v in cameras()}
        masks["top"][7:9, 7:9] = False
        target = observed_target(masks, cameras())
        with self.assertRaisesRegex(ValueError, "hole"):
            fitted_family_program("sphere", target, masks, cameras())
        with self.assertRaisesRegex(ValueError, "structured"):
            fitted_family_program("asymmetric_multipart_solid", target, masks, cameras())
        bad = cameras(); bad["front"]["matrix_world"][0][0] = -1.
        with self.assertRaisesRegex(ValueError, "proper"):
            observed_target(masks, bad)

    def test_clipped_oblique_support_is_censored_without_contour_completion(self):
        from reconstruction.oriented_support import support_residual
        coverage = np.zeros((16, 16)); coverage[4:12, 4:12] = 1.
        masks = {v: coverage >= .5 for v in cameras()}
        coverages = {v: coverage.copy() for v in masks}
        records = cameras()
        target = observed_target(masks, records, coverage_masks=coverages)
        masks["oblique"] = np.zeros((16, 16), bool)
        masks["oblique"][4:12, 6:] = True
        coverages["oblique"] = masks["oblique"].astype(float)
        records["oblique"] = records["top"]
        evidence = camera_support_evidence(masks, records, target.bounds,
                                           coverage_masks=coverages)
        self.assertGreater(int(evidence.censored.sum()), 0)
        prediction = evidence.values.copy()
        prediction[evidence.censored] += .4
        np.testing.assert_allclose(support_residual(evidence, prediction), 0.)
        prediction[evidence.censored] = evidence.values[evidence.censored]-.4
        self.assertTrue((support_residual(evidence, prediction)[evidence.censored] < 0).all())

    def test_world_bevel_adapter_targets_mesh_descendant_not_empty_root(self):
        from importlib.util import module_from_spec, spec_from_file_location
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import sys
        root_path = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root_path/"scripts"))
        spec = spec_from_file_location("frozen_family_runner", root_path/"scripts/run_frozen_family_reconstruction.py")
        runner = module_from_spec(spec); spec.loader.exec_module(runner)
        selections = []
        modifier = SimpleNamespace(type="BEVEL", width=.02, segments=3)
        child = SimpleNamespace(type="MESH", modifiers=[modifier],
                                select_set=lambda value: selections.append(value),
                                parent=None, children_recursive=(), get=lambda key, default=False: default)
        root = SimpleNamespace(type="EMPTY", children_recursive=(child,), get=lambda key, default=False: default)
        active = SimpleNamespace(active=None)
        applied = []
        fake = SimpleNamespace(context=SimpleNamespace(view_layer=SimpleNamespace(
            objects=active, update=lambda: None)),
            ops=SimpleNamespace(object=SimpleNamespace(select_all=lambda **kw: None,
                transform_apply=lambda **kw: applied.append(active.active))))
        with patch.dict(sys.modules, {"bpy": fake}):
            runner._world_bevel(root, .13)
        self.assertEqual(applied, [child])
        self.assertEqual(selections, [True])
        self.assertEqual(modifier.width, .13)
        self.assertEqual(modifier.segments, 8)

    def test_retained_recipe_replay_preserves_pose_precision_and_metadata(self):
        from reconstruction.frozen_family import retained_family_program
        from primitives.shape_program import ShapeNode, ShapeProgram
        parameters = {"x": .12345678912345678, "height_world": 1.2345678912345678,
                      "rotation": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
        program = ShapeProgram("1", "saved", (ShapeNode("n", "add", "box", parameters),),
                               metadata={"observed_only": True})
        wire = program.to_dict()
        self.assertEqual(retained_family_program(wire).to_dict(), wire)
        wire["root_nodes"][0]["children"] = ["unexpected"]
        with self.assertRaisesRegex(ValueError, "leaf"):
            retained_family_program(wire)

    def test_prepared_manifest_rejects_changed_obj_before_native_work(self):
        from importlib.util import module_from_spec, spec_from_file_location
        from pathlib import Path
        import hashlib, json, tempfile, sys
        root_path = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root_path/"scripts"))
        spec = spec_from_file_location("frozen_manifest_runner", root_path/"scripts/run_frozen_family_reconstruction.py")
        runner = module_from_spec(spec); spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)/"sphere"; folder.mkdir()
            (folder/"evaluated-exact.npz").write_bytes(b"fixture archive")
            (folder/"evaluated.obj").write_text("fixture OBJ")
            (folder/"program.json").write_text(json.dumps({"fixture": True}))
            row = {"geometry_hash": "a"*64,
                   "npz_sha256": hashlib.sha256((folder/"evaluated-exact.npz").read_bytes()).hexdigest(),
                   "obj_sha256": hashlib.sha256((folder/"evaluated.obj").read_bytes()).hexdigest()}
            prepared = {"run_root": name, "cases": {"sphere": row}}
            self.assertEqual(runner._prepared_inputs(prepared, ["sphere"])["sphere"]["obj_sha256"], row["obj_sha256"])
            (folder/"evaluated.obj").write_text("changed OBJ")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                runner._prepared_inputs(prepared, ["sphere"])

    def test_saved_triangle_order_requires_exact_vertices_and_orientation(self):
        from reconstruction.frozen_family import retained_triangle_indices
        from reconstruction.native_geometry import GeometryArrays
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        source = GeometryArrays.capture(vertices, faces)
        replay = GeometryArrays.capture(vertices, np.roll(faces[::-1], 1, axis=1))
        np.testing.assert_array_equal(retained_triangle_indices(source, replay), faces)
        shifted = vertices.copy(); shifted[0, 0] = 1e-12
        with self.assertRaisesRegex(ValueError, "vertices"):
            retained_triangle_indices(source, GeometryArrays.capture(shifted, faces))
        flipped = faces.copy(); flipped[0] = flipped[0, ::-1]
        with self.assertRaisesRegex(ValueError, "oriented surface"):
            retained_triangle_indices(source, GeometryArrays.capture(vertices, flipped))

    def test_full_inventory_distinguishes_reused_unrun_and_unsupported(self):
        rows = initial_family_rows(["sphere"], ["smooth_vase", "rounded_triangle_dot"])
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows["sphere"]["status"], "pending")
        self.assertEqual(rows["smooth_vase"]["status"], "reused")
        self.assertEqual(rows["cylinder"]["status"], "unrun")
        self.assertEqual(rows["torus"]["status"], "unrun")
        self.assertEqual(rows["concave_arch"]["status"], "unrun")
        self.assertEqual(rows["asymmetric_multipart_solid"]["status"], "unsupported")
        self.assertFalse(any(row["aggregate_accepted"] for row in rows.values()))
        self.assertEqual(len(SUPPORTED_FAMILIES), 9)
        with self.assertRaises(ValueError):
            initial_family_rows(["sphere"], ["sphere"])


if __name__ == "__main__":
    unittest.main()
