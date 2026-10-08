"""Extraction diagnostics follow actual connectivity, never geometric validity."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from blender_blocking.metrics.topology import mesh_topology_report
from blender_blocking.metrics.topology_receipt import (
    ConnectivityTopologyReceipt, topology_for_geometry,
)
from blender_blocking.reconstruction.native_geometry import GeometryArrays, GeometryCache
from reconstruction.backends.visual_hull import VisualHullBackend
from reconstruction.types import CandidateRequest
from test_volume import _build_full_view_target
from volume import Bounds3D, DenseVolumeGrid, extract_mesh


def tetra():
    return GeometryArrays.capture(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


class TopologyReuseTests(unittest.TestCase):
    def test_receipt_matches_same_indices_after_coordinate_update(self):
        original = tetra()
        receipt = ConnectivityTopologyReceipt.capture(original.vertices, original.faces)
        translated = GeometryArrays.capture(original.vertices+3, original.faces)
        with patch('blender_blocking.metrics.topology_receipt.mesh_topology_report',
                   side_effect=AssertionError('duplicate topology evaluation')):
            report, reused = topology_for_geometry(translated, receipt)
            cache = GeometryCache()
            self.assertEqual(cache.topology_report(translated, receipt), report)
            self.assertEqual(cache.receipt_reuses, 1)
        self.assertTrue(reused)
        self.assertTrue(report['watertight'])

    def test_connectivity_vertex_count_and_evaluator_changes_invalidate(self):
        original = tetra()
        receipt = ConnectivityTopologyReceipt.capture(original.vertices, original.faces)
        changed = [
            (GeometryArrays.capture(original.vertices, original.faces[:-1]), receipt),
            (GeometryArrays.capture(np.r_[original.vertices, [[2, 2, 2]]], original.faces), receipt),
            (original, replace(receipt, evaluator='different_evaluator')),
            (original, receipt.report.to_dict()),
        ]
        with patch('blender_blocking.metrics.topology_receipt.mesh_topology_report',
                   wraps=mesh_topology_report) as evaluate:
            for data, stale in changed:
                report, reused = topology_for_geometry(data, stale)
                self.assertFalse(reused)
                self.assertEqual(report, mesh_topology_report(data.vertices, data.faces).to_dict())
            self.assertEqual(evaluate.call_count, 4)

    def test_coordinate_collapse_does_not_relabel_index_report_as_solid_guard(self):
        from reconstruction.grouped_solids import solid_guard
        original = tetra()
        receipt = ConnectivityTopologyReceipt.capture(original.vertices, original.faces)
        collapsed = GeometryArrays.capture(np.zeros_like(original.vertices), original.faces)
        report, reused = topology_for_geometry(collapsed, receipt)
        self.assertTrue(reused)
        self.assertTrue(report['watertight'])
        self.assertFalse(solid_guard(collapsed)['valid_solid'])

    def test_actual_extraction_receipt_matches_and_mutated_faces_invalidate(self):
        data = np.zeros((8, 8, 8), bool)
        data[2:6, 2:6, 2:6] = True
        mesh = extract_mesh(DenseVolumeGrid(data, Bounds3D(-1, 1, -1, 1, -1, 1)))
        actual = GeometryArrays.capture(mesh.vertices, mesh.faces)
        report, reused = topology_for_geometry(actual, mesh.topology_receipt)
        self.assertTrue(reused)
        self.assertEqual(report, mesh_topology_report(actual.vertices, actual.faces).to_dict())
        changed_faces = mesh.faces.copy()
        changed_faces[0] = changed_faces[1]
        changed = GeometryArrays.capture(mesh.vertices, changed_faces)
        self.assertFalse(topology_for_geometry(changed, mesh.topology_receipt)[1])

    def test_visual_hull_backend_reuses_actual_receipt_and_keeps_artifacts(self):
        for with_cache in (False, True):
            with self.subTest(with_cache=with_cache), tempfile.TemporaryDirectory() as root:
                cache = GeometryCache() if with_cache else None
                request = CandidateRequest(
                    candidate_id='bounded-topology-fixture', backend_name='visual_hull_voxel',
                    target=_build_full_view_target(), config={'resolution': 8},
                    artifact_root=Path(root), context=SimpleNamespace(geometry_cache=cache))
                with patch('blender_blocking.metrics.topology_receipt.mesh_topology_report',
                           wraps=mesh_topology_report) as evaluate:
                    result = VisualHullBackend().reconstruct(request)
                self.assertIsNotNone(result.geometry, result.errors)
                self.assertEqual(evaluate.call_count, 1)
                extras = result.metric_result.extras
                self.assertTrue(extras['topology_reuse']['extraction_receipt_reused'])
                self.assertEqual(extras['topology'],
                                 mesh_topology_report(result.geometry.vertices, result.geometry.faces).to_dict())
                self.assertTrue(result.mesh_path.is_file())
                self.assertTrue((result.volume_path/'volume.npz').is_file())
                if cache is not None:
                    self.assertEqual(cache.receipt_reuses, 1)

    def test_coordinate_refinement_reuses_indices_but_discards_stale_normals(self):
        data = np.zeros((8, 8, 8), bool)
        data[2:6, 2:6, 2:6] = True
        source = extract_mesh(DenseVolumeGrid(data, Bounds3D(-1, 1, -1, 1, -1, 1)))
        self.assertIsNotNone(source.normals)
        changed = GeometryArrays.capture(source.vertices+.01, source.faces)
        request = CandidateRequest(
            candidate_id='coordinate-reuse-fixture', backend_name='visual_hull_voxel',
            target=_build_full_view_target(), config={'resolution': 8, 'adaptive_hull': True})
        with patch('volume.extract_mesh', return_value=source), patch(
                'reconstruction.adaptive_geometry.refine_hull_boundary',
                return_value=(changed, {'accepted': True})):
            result = VisualHullBackend().reconstruct(request)
        extras = result.metric_result.extras
        np.testing.assert_array_equal(result.geometry.vertices, changed.vertices)
        self.assertTrue(extras['topology_reuse']['extraction_receipt_reused'])
        self.assertFalse(extras['boundary_mesh_result']['has_normals'])
        self.assertFalse(extras['mesh']['has_normals'])


if __name__ == '__main__':
    unittest.main()
