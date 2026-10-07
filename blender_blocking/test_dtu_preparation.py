"""Small analytic regressions for DTU preparation, without datasets or Blender."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from blender_blocking.evaluation.protocols.dtu import dtu_native_adapter
from blender_blocking.evaluation.protocols.dtu_preparation import (
    PreparationLimitExceeded, array_sha256, dtu_directional_sets,
    prepare_dtu_prediction, sample_dtu_mesh, silhouette_keep_mask, thin_dtu_points,
)


class DtuPreparationTests(unittest.TestCase):
    def assets(self):
        return {"reference": np.array([[0.,0.,1.],[1.,0.,1.],[0.,0.,-1.]]),
                "obs_mask": np.ones((4, 4, 4), dtype=bool),
                "observation_bb": np.array([[-1.,-1.,-1.],[2.,2.,2.]]),
                "observation_res": np.ones(3), "ground_plane": np.array([0.,0.,1.,0.]),
                "scale_mat": np.eye(4)}

    def test_mesh_sampling_matches_reference_midcell_equation(self):
        vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        out = sample_dtu_mesh(vertices, np.array([[0,1,2]]))
        c = np.mgrid[:6, :6].astype(float)
        c = ((c + .5) / 5).transpose(1,2,0)
        uv = c[c.sum(axis=-1) < 1]
        expected = np.concatenate((vertices, np.column_stack((uv, np.zeros(len(uv))))))
        np.testing.assert_array_equal(out, expected)
        with self.assertRaises(PreparationLimitExceeded):
            sample_dtu_mesh(vertices, np.array([[0,1,2]]), max_points=3)
        with self.assertRaises(ValueError):
            sample_dtu_mesh(vertices, np.array([[0.,1.,2.]]))

    def test_skinny_triangle_refuses_large_allocation_without_coarsening(self):
        vertices = np.array([[0.,0.,0.],[1_000_000.,0.,0.],[0.,1.,0.]])
        with self.assertRaises(PreparationLimitExceeded):
            sample_dtu_mesh(vertices, np.array([[0,1,2]]), max_points=4)

    def test_degenerate_triangles_keep_vertices_without_interior_samples(self):
        vertices = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        np.testing.assert_array_equal(sample_dtu_mesh(vertices, np.array([[0,1,2]])), vertices)

    def test_seeded_thinning_matches_reference_radius_greedy_rule(self):
        data = np.array([[0.,0.,0.],[.1,0.,0.],[.2,0.,0.],[.41,0.,0.],[.41,0.,0.]])
        shuffled = data.copy(); np.random.default_rng(77).shuffle(shuffled, axis=0)
        keep = np.ones(len(data), dtype=bool)
        for index in range(len(data)):
            if keep[index]:
                distance = np.linalg.norm(shuffled - shuffled[index], axis=1)
                keep[distance <= .2] = False
                keep[index] = True
        output, receipt = thin_dtu_points(data, seed=77)
        np.testing.assert_array_equal(output, shuffled[keep])
        self.assertEqual(receipt["seed"], 77)
        np.testing.assert_array_equal(data[0], [0.,0.,0.])

    def test_observation_mask_only_filters_accuracy_queries(self):
        assets = self.assets()
        assets["obs_mask"][1,1,2] = False
        data = np.array([[0.,0.,1.],[1.,0.,1.],[3.,0.,1.]])
        sets = dtu_directional_sets(data, assets)
        np.testing.assert_array_equal(sets["prediction_queries"], [[1.,0.,1.]])
        np.testing.assert_array_equal(sets["prediction_targets"], data)
        self.assertEqual(len(sets["reference_queries"]), 2)
        self.assertEqual(len(sets["reference_targets"]), 3)

    def test_bb_rounding_and_strict_patch_upper_boundary(self):
        assets = self.assets()
        data = np.array([[-.5,0.,1.], [122.,0.,1.]])
        sets = dtu_directional_sets(data, assets)
        self.assertEqual(len(sets["prediction_targets"]), 1)
        self.assertEqual(len(sets["prediction_queries"]), 1)

    def test_prepared_native_sets_are_bound_to_prediction_assets_and_implementation(self):
        assets = self.assets(); data = assets["reference"][:2].copy()
        prepared = prepare_dtu_prediction(mode="partgs_block", prediction=data, assets=assets, seed=77)
        self.assertEqual(prepared["status"], "available")
        def evaluate(prediction=data):
            return dtu_native_adapter(mode="partgs_block", assets=assets, prediction=prediction,
                directional_sets=prepared["directional_sets"], sampling_receipt=prepared["sampling_receipt"],
                transform_direction="prediction_to_native_mm")
        self.assertEqual(evaluate()["overall_mm"], 0)
        self.assertEqual(evaluate(data + .01)["status"], "unavailable")
        assets["ground_plane"][3] += 1
        self.assertEqual(evaluate()["status"], "unavailable")

    def test_culling_modes_require_verified_hash_matched_wrapper(self):
        assets = self.assets(); data = assets["reference"][:2].copy()
        self.assertEqual(prepare_dtu_prediction(mode="partgs_point", prediction=data,
                                              assets=assets)["status"], "unavailable")
        keep = np.array([True, False])
        culling = {"keep": keep, "receipt": {"verified_wrapper": True, "source_sha256": "fixture",
            "input_sha256": array_sha256(data), "keep_sha256": array_sha256(keep), "selection": "points",
            "prediction_sha256": array_sha256(data), "native_prediction_sha256": array_sha256(data),
            "camera_sha256": {"fixture": "camera"}, "mask_sha256": {"fixture": "mask"}}}
        prepared = prepare_dtu_prediction(mode="partgs_point", prediction=data, assets=assets, culling=culling)
        self.assertEqual(prepared["status"], "available")
        self.assertEqual(len(prepared["directional_sets"]["prediction_targets"]), 1)
        culling["keep"][1] = True
        self.assertEqual(prepare_dtu_prediction(mode="partgs_point", prediction=data,
                                              assets=assets, culling=culling)["status"], "unavailable")

    def test_explicit_silhouette_rule_rejects_behind_and_outside(self):
        data = np.array([[0.,0.,1.],[1.,0.,1.],[0.,0.,-1.],[3.,0.,1.]])
        camera = np.column_stack((np.eye(3), np.zeros(3)))
        mask = np.array([[True, False], [True, True]])
        np.testing.assert_array_equal(silhouette_keep_mask(data, [camera], [mask]),
                                      [True, False, False, False])

    def test_small_cli_fixture_persists_arrays_and_preserves_existing_directory(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_dtu_subset.py"
        spec = importlib.util.spec_from_file_location("dtu_subset_fixture", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "assets.npz", **self.assets())
            np.savez(root / "prediction.npz", points=self.assets()["reference"][:2])
            args = SimpleNamespace(assets=root / "assets.npz", dataset_dir=None, scan=None,
                prediction_native_mm=False, scale_matrix=None, prediction=root / "prediction.npz",
                culling=None, culling_receipt=None, mode="partgs_block", seed=77, max_points=100,
                output=root / "result", case_id="fixture")
            result = module.evaluate_subset(args)
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["overall_mm"], 0)
            self.assertTrue((args.output / "samples.npz").is_file())
            self.assertEqual(json.loads((args.output / "result.json").read_text())["case_id"], "fixture")
            with self.assertRaises(FileExistsError):
                module.evaluate_subset(args)
            obj = root / "prediction.obj"
            obj.write_text("v 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\n")
            vertices, faces = module.load_prediction(obj)
            self.assertEqual(vertices.shape, (3, 3))
            np.testing.assert_array_equal(faces, [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
