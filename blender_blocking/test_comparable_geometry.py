import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]
from blender_blocking.evaluation.comparable_geometry import normalize_mesh, sample_surface, read_obj
from blender_blocking.evaluation.geometry import surface_distance_report


class ComparableGeometryTests(unittest.TestCase):
    def test_uniform_scale_translation_invariance_but_aspect_error_is_retained(self):
        shape = np.array([[-1., -2., -3.], [1., 2., 3.], [0., 0., 0.]])
        normalized, _ = normalize_mesh(shape)
        moved, _ = normalize_mesh(shape * 7 + [20, -10, 50])
        np.testing.assert_allclose(normalized, moved)
        distorted, _ = normalize_mesh(shape * [2, 1, 1])
        self.assertFalse(np.allclose(normalized, distorted))

    def test_surface_sampling_is_deterministic_and_area_weighted(self):
        vertices = np.array([[0.,0,0], [1,0,0], [0,1,0], [10,0,0], [12,0,0], [10,2,0]])
        triangles = np.array([[0,1,2], [3,4,5]])
        first, normals = sample_surface(vertices, triangles, count=10000, seed=7)
        second, _ = sample_surface(vertices, triangles, count=10000, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float((first[:,0] > 5).mean()), .8, delta=.02)
        np.testing.assert_allclose(normals, np.tile([0,0,1], (10000,1)))

    def test_fscore_catches_displaced_surface(self):
        points = np.array([[0.,0,0], [1,0,0], [0,1,0]])
        same = surface_distance_report(points, points, tolerance=.02)
        shifted = surface_distance_report(points, points + [0,0,.1], tolerance=.02)
        self.assertEqual(same.fscore_tau, 1.)
        self.assertEqual(shifted.fscore_tau, 0.)

    def test_obj_negative_indices_and_quad_triangulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quad.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n")
            vertices, triangles = read_obj(path)
            self.assertEqual(vertices.shape, (4,3))
            np.testing.assert_array_equal(triangles, [[0,1,2],[0,2,3]])

    def test_farthest_sampler_matches_original_selection(self):
        from blender_blocking.primitives.ellipsoid_proxy.initialization import _sample_farthest_points
        points = np.random.default_rng(6).normal(size=(100, 3))
        selected = [points[np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1))]]
        for _ in range(1, 20):
            nearest = np.linalg.norm(points[:,None,:] - np.asarray(selected)[None,:,:], axis=2).min(axis=1)
            selected.append(points[int(np.argmax(nearest))])
        np.testing.assert_array_equal(_sample_farthest_points(points, 20), selected)
        np.testing.assert_array_equal(_sample_farthest_points(points, 1), points[:1])
        np.testing.assert_array_equal(_sample_farthest_points(points, 0), points[:0])

    def test_polygon_projection_measures_a_missing_half(self):
        from blender_blocking.reconstruction.projected_metrics import projected_mesh_metrics
        from blender_blocking.reconstruction.types import Bounds3D
        target = SimpleNamespace(bounds=Bounds3D(0,1,0,1,0,1), constraints=[SimpleNamespace(view="front", mask=np.ones((101,101), bool))])
        vertices = np.array([[0.,0,0], [.5,0,0], [.5,0,1], [0,0,1]])
        result = projected_mesh_metrics(target, vertices, [(0,1,2,3)])["front"]
        self.assertAlmostEqual(result["area_iou"], 51/101, places=6)
        self.assertFalse(result["passed"])
        self.assertEqual(result["candidate_projection_source"], "exported_mesh_polygon_raster")

    def test_evidence_refreshes_selected_bundle_and_failure_taxonomy(self):
        from blender_blocking.e2e.evidence import attach_evaluation_evidence
        from blender_blocking.evaluation.schemas import EvaluationBundle
        views = {v: {"area_iou": .9, "boundary_iou": .8, "signed_distance_loss": .01, "required": True, "passed": True} for v in ("front", "side", "top")}
        bundle = {"candidate_id": "selected", "status": "fail", "metric_groups": [
            {"name": "silhouette", "status": "fail", "metrics": []},
            {"name": "geometry", "status": "not_applicable", "metrics": []}],
            "failures": [{"code": "silhouette_required_metrics_missing", "severity": "fail", "subsystem": "silhouette"}]}
        untouched = dict(bundle, candidate_id="other")
        payload = {"validation_mode": "render-iou", "views": views, "average_iou": .9, "min_view_iou": .9,
                   "backend_result": {"selected": {"candidate_id": "selected"}}, "evaluation_bundles": [bundle, untouched]}
        result = attach_evaluation_evidence(payload, geometry_payload={"geometry_true": {"source": "bbox_uniform_normalized_mesh_surface", "chamfer_l1": .01, "fscore_tau": .95}})
        parsed = EvaluationBundle.from_dict(result["evaluation_bundle"])
        self.assertEqual(parsed.status, "pass")
        self.assertEqual(parsed.metric_index()["geometry.true.chamfer_l1_normalized"].value, .01)
        self.assertNotIn("silhouette_required_metrics_missing", {f.code for f in parsed.failures})
        self.assertEqual(result["evaluation_bundles"][1], untouched)

    def test_render_collection_includes_all_shape_program_parts(self):
        from unittest.mock import patch
        from blender_blocking.integration.blender_ops import silhouette_render
        mesh_a = SimpleNamespace(type="MESH", get=lambda key, default=None: default)
        mesh_b = SimpleNamespace(type="MESH", get=lambda key, default=None: default)
        root = SimpleNamespace(type="EMPTY", children_recursive=[mesh_a, mesh_b])
        with patch.object(silhouette_render, "BLENDER_AVAILABLE", True):
            result = silhouette_render.collect_target_objects(None, [root, mesh_a])
        self.assertEqual(result, [mesh_a, mesh_b])

    def test_geometry_uses_externally_rendered_artifact_before_backend_proxy(self):
        from blender_blocking.e2e.ground_truth import _mesh_path_from_payload
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proxy, rendered = root / "proxy.obj", root / "rendered.obj"
            proxy.touch(); rendered.touch()
            payload = {"backend_result": {"mesh_path": str(proxy)}, "mesh_path": str(rendered)}
            self.assertEqual(_mesh_path_from_payload(payload), rendered)

    def test_validator_resolves_engine_alias_and_guards_removed_sample_property(self):
        from unittest.mock import patch
        from blender_blocking.e2e import validator
        render = SimpleNamespace(engine="BLENDER_WORKBENCH", image_settings=SimpleNamespace(),
                                 bl_rna=SimpleNamespace(properties={"engine": SimpleNamespace(enum_items=[SimpleNamespace(identifier="BLENDER_EEVEE_NEXT")])}))
        scene = SimpleNamespace(render=render, eevee=SimpleNamespace())
        instance = object.__new__(validator.E2EValidator)
        instance.render_config = SimpleNamespace(transparent_bg=True, color_mode="RGBA", resolution=(64,64), engine="BLENDER_EEVEE", samples=64)
        with patch.object(validator, "bpy", SimpleNamespace(context=SimpleNamespace(scene=scene))):
            instance.setup_render_settings()
        self.assertEqual(render.engine, "BLENDER_EEVEE_NEXT")
        self.assertIsNone(instance.render_engine_evidence["samples_applied"])


if __name__ == "__main__":
    unittest.main()
