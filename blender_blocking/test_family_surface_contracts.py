"""Continuous reference bounds and independent, source-conditioned policy checks."""
import copy
import math
import unittest

import numpy as np

from blender_blocking.evaluation.canonical_artifacts import CANONICAL_VIEWS
from blender_blocking.evaluation.family_surface_contracts import (
    evaluate_family_surface_contract, freeze_family_surface_contract,
    reference_facet_certificate,
)
from blender_blocking.reconstruction.native_geometry import GeometryArrays


def octahedron(radii=(1., 1., 1.)):
    vertices = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    faces = [[0,2,4],[2,1,4],[1,3,4],[3,0,4],
             [2,0,5],[1,2,5],[3,1,5],[0,3,5]]
    return GeometryArrays.capture(vertices * radii, faces)


def cameras():
    record = {"projection": "ORTHO", "matrix_world": np.eye(4).tolist(),
              "ortho_scale": 2., "shift_x": 0., "shift_y": 0.,
              "clip_start": .1, "clip_end": 100., "resolution": [512,512],
              "pixel_aspect": [1.,1.]}
    return {view: copy.deepcopy(record) for view in CANONICAL_VIEWS}


def revolved(rb=1., rt=1., height=2., count=8):
    angle = np.arange(count) * 2 * math.pi / count
    vertices = [[r * math.cos(a), r * math.sin(a), z]
                for r, z in ((rb, -height/2), (rt, height/2)) for a in angle]
    vertices.extend([[0,0,-height/2],[0,0,height/2]])
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend([[i,j,count+j],[i,count+j,count+i],
                      [2*count,j,i],[2*count+1,count+i,count+j]])
    return GeometryArrays.capture(vertices, faces)


def raw(contract, distance=0., angle=0.):
    return {**contract["metric"], "reference_geometry_hash": contract["reference_geometry_hash"],
            "candidate_geometry_hash": "a" * 64,
            "normal_orientation": "oriented; opposite normals are 180 degrees",
            "normal_interpretation": "geometric face normals, not shading normals",
            "units": "unchanged Blender world coordinates; no fitting or normalization",
            "symmetric_mean_distance_world": distance, "distance_p95_world": distance,
            "sampled_max_distance_world": distance,
            "normal_angle_mean_degrees": angle, "normal_angle_p95_degrees": angle}


class FamilySurfaceContracts(unittest.TestCase):
    def test_sphere_bound_includes_facet_interior_not_only_exact_vertices(self):
        certificate = reference_facet_certificate("sphere", {"radius": 1.}, octahedron())
        self.assertAlmostEqual(certificate["distance_bound_world"], 1 - 1 / math.sqrt(3))
        self.assertAlmostEqual(certificate["normal_correspondence_bound_degrees"], math.degrees(math.acos(1/math.sqrt(3))))
        self.assertEqual(certificate["vertex_construction_error_world"], 0.)
        self.assertFalse(certificate["sampled"])
        self.assertFalse(certificate["candidate_boundary_qualified"])

    def test_ellipsoid_bound_scales_world_distance_and_preserves_normal_scope(self):
        certificate = reference_facet_certificate("anisotropic_ellipsoid", {"radii": [2.,1.,.5]}, octahedron([2.,1.,.5]))
        self.assertAlmostEqual(certificate["distance_bound_world"], 2 * (1 - 1/math.sqrt(3)))
        self.assertIn("not nearest-point", certificate["normal_correspondence"])

    def test_cylinder_and_frustum_bound_continuous_side_arc_and_flat_caps(self):
        for family, parameters, data in (
                ("cylinder", {"radius": 1., "height": 2.}, revolved()),
                ("tapered_frustum", {"radius_bottom": 1., "radius_top": .5, "height": 2.}, revolved(rt=.5))):
            certificate = reference_facet_certificate(family, parameters, data)
            self.assertAlmostEqual(certificate["distance_bound_world"], 1 - math.cos(math.pi/8))
            self.assertLessEqual(certificate["normal_correspondence_bound_degrees"], 22.5 + 1e-8)

    def test_capsule_piecewise_bound_preserves_actual_hemisphere_joins(self):
        from blender_blocking.primitives.capsule import CapsulePrimitive
        mesh = CapsulePrimitive(radius=.4, segment_height=1.2).to_mesh_data(48)
        vertices, faces = mesh.vertices, mesh.faces
        certificate = reference_facet_certificate("capsule", {"radius": .4, "segment_height": 1.2}, GeometryArrays.capture(vertices, faces))
        self.assertGreater(certificate["distance_bound_world"], 0.)
        self.assertLess(certificate["distance_bound_world"], .01)

    def test_open_flipped_and_parameter_drift_cannot_certify(self):
        data = octahedron()
        for other in (GeometryArrays.capture(data.vertices, data.faces[:-1]),
                      GeometryArrays.capture(data.vertices, data.faces[:, ::-1]),
                      GeometryArrays.capture(data.vertices * 1.01, data.faces)):
            with self.assertRaises(ValueError):
                reference_facet_certificate("sphere", {"radius": 1.}, other)

    def test_fixed_source_pixel_budget_and_anisotropic_camera_pitch(self):
        frames = cameras()
        frames["front"]["resolution"] = [256,512]
        contract = freeze_family_surface_contract("sphere", {"radius":1.}, octahedron(), frames)
        self.assertAlmostEqual(contract["reconstruction_allowance"]["world"], .5 * 2/512)
        self.assertEqual(contract["artist_acceptance_limits"], None)
        self.assertEqual(contract["engineering_status"], "frozen")
        self.assertEqual(frames["front"]["resolution"], [256,512])

    def test_missing_actual_clipping_or_camera_frame_refuses_freeze(self):
        for edit in (lambda frames: frames["front"].pop("clip_start"),
                     lambda frames: frames.pop("side"),
                     lambda frames: frames["top"]["matrix_world"][0].__setitem__(0, 2.)):
            frames = cameras(); edit(frames)
            with self.assertRaises(ValueError):
                freeze_family_surface_contract("sphere", {"radius":1.}, octahedron(), frames)

    def test_changed_candidate_metric_never_changes_contract_or_grants_artist_pass(self):
        contract = freeze_family_surface_contract("sphere", {"radius":1.}, octahedron(), cameras())
        original = copy.deepcopy(contract)
        for distance, angle, expected in ((0., 0., True), (2., 179., False)):
            verdict = evaluate_family_surface_contract(contract, raw(contract, distance, angle), candidate_geometry_hash="a" * 64)
            self.assertEqual(verdict["engineering_surface_passed"], expected)
            self.assertIsNone(verdict["artist_surface_passed"])
            self.assertFalse(verdict["aggregate_passed"])
        self.assertEqual(original, contract)

    def test_mismatched_or_invalid_observation_and_contract_refused(self):
        contract = freeze_family_surface_contract("sphere", {"radius":1.}, octahedron(), cameras())
        for key, value in (("seed", 1), ("reference_geometry_hash", "b"*64),
                           ("normal_interpretation", "shading"),
                           ("normal_angle_p95_degrees", float("nan")),
                           ("symmetric_mean_distance_world", True)):
            observation = raw(contract); observation[key] = value
            with self.assertRaises(ValueError):
                evaluate_family_surface_contract(contract, observation, candidate_geometry_hash="a"*64)
        contract["engineering_limits"]["normal_angle_p95_degrees_max"] = 180.
        with self.assertRaisesRegex(ValueError, "digest"):
            evaluate_family_surface_contract(contract, raw(contract), candidate_geometry_hash="a"*64)

    def test_torus_policy_uses_only_certified_frozen_source_and_refuses_other_geometry(self):
        from unittest.mock import patch
        data = octahedron()
        parameters = {"primitive": "torus", "major_radius": .7, "minor_radius": .22}
        proof = {"status": "certified", "reference_geometry_hash": data.content_hash,
                 "maximum_source_facet_distance_world": .003,
                 "maximum_normal_angle_degrees": 8.,
                 "maximum_vertex_construction_shift_world": 1e-6,
                 "normal_correspondence": "analytic parameter cell correspondence",
                 "distance_correspondence": "continuous bidirectional parameter cover"}
        with patch("blender_blocking.evaluation.torus_reference.torus_reference_certificate", return_value=proof):
            contract = freeze_family_surface_contract("torus", parameters, data, cameras())
        self.assertEqual(contract["reference_certificate"]["torus_parameter_cover"], proof)
        self.assertAlmostEqual(contract["engineering_limits"]["symmetric_mean_distance_world_max"], .003 + .5*2/512)
        self.assertEqual(contract["engineering_limits"]["normal_angle_p95_degrees_max"], 9.)
        self.assertIsNone(contract["artist_acceptance_limits"])
        changed = freeze_family_surface_contract("torus", parameters, data, cameras())
        self.assertIsNone(changed["engineering_limits"])
        self.assertIn("frozen original authored torus identity", changed["reason"])
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            reference_facet_certificate("torus", {**parameters, "major_radius": .71}, data)

    def test_unsupported_reference_remains_unqualified_instead_of_copying_vase(self):
        contract = freeze_family_surface_contract("rounded_triangle_dot", {"thickness":.48}, octahedron(), cameras())
        self.assertIsNone(contract["engineering_limits"])
        verdict = evaluate_family_surface_contract(contract, raw(contract), candidate_geometry_hash="a"*64)
        self.assertIsNone(verdict["engineering_surface_passed"])
        self.assertFalse(verdict["aggregate_passed"])


if __name__ == "__main__":
    unittest.main()
