#!/usr/bin/env python3
"""Pure tests for signed-distance volume projection."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from reconstruction.backends.visual_hull import VisualHullBackend
from reconstruction.types import (
    Bounds3D as ReconstructionBounds3D,
    CandidateRequest,
    OrthographicCameraSpec,
    ReconstructionTarget,
    ViewConstraint,
)
from volume import (
    Bounds3D,
    DenseVolumeGrid,
    SIGN_CONVENTION,
    extract_mesh,
    occupancy_grid_from_signed_distance,
    signed_distance_grid_from_volume,
)


class SDFProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = Bounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def test_numpy_fallback_preserves_sign_convention_and_round_trips(self) -> None:
        occupancy = np.zeros((5, 5, 5), dtype=bool)
        occupancy[1:4, 1:4, 1:4] = True
        grid = DenseVolumeGrid(occupancy, self.bounds, chunk_size=2)

        result = signed_distance_grid_from_volume(
            grid,
            prefer_scipy=False,
            numpy_max_voxels=10_000,
            narrow_band_voxels=2,
        )

        field = result.grid.to_dense()
        self.assertEqual(result.report.method, "numpy_bruteforce")
        self.assertEqual(result.report.sign_convention, SIGN_CONVENTION)
        self.assertLess(float(field[2, 2, 2]), 0.0)
        self.assertGreater(float(field[0, 0, 0]), 0.0)
        self.assertGreater(result.report.surface_voxels, 0)
        self.assertEqual(result.report.narrow_band_voxels, 2)
        self.assertLessEqual(
            float(np.max(np.abs(field))),
            float(result.report.narrow_band_world) + 1e-6,
        )

        recovered = occupancy_grid_from_signed_distance(result.grid)
        np.testing.assert_array_equal(recovered.to_dense(), occupancy)

    def test_sdf_grid_meshes_at_zero_level_when_skimage_is_available(self) -> None:
        try:
            import skimage  # noqa: F401
        except Exception as exc:
            self.skipTest(f"skimage unavailable: {exc}")

        occupancy = np.zeros((8, 8, 8), dtype=bool)
        occupancy[2:6, 2:6, 2:6] = True
        grid = DenseVolumeGrid(occupancy, self.bounds)
        sdf = signed_distance_grid_from_volume(grid)

        mesh = extract_mesh(sdf.grid, method="marching_cubes")

        self.assertEqual(mesh.status, "ok")
        self.assertGreater(int(mesh.vertices.shape[0]), 0)
        self.assertGreater(int(mesh.faces.shape[0]), 0)
        self.assertEqual(mesh.metrics["level"], 0.0)
        self.assertLess(mesh.metrics["field_min"], 0.0)
        self.assertGreater(mesh.metrics["field_max"], 0.0)

    def test_visual_hull_backend_can_emit_and_mesh_sdf_volume(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        target = ReconstructionTarget(
            constraints=(
                ViewConstraint(
                    view="front",
                    mask=mask,
                    camera=OrthographicCameraSpec(
                        view_name="front",
                        axis="z",
                        azimuth_deg=0.0,
                    ),
                ),
            ),
            bounds=ReconstructionBounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        )
        backend = VisualHullBackend()
        with tempfile.TemporaryDirectory() as tmpdir:
            request = CandidateRequest(
                candidate_id="vh-sdf",
                backend_name="visual_hull_voxel",
                target=target,
                config={
                    "resolution": 8,
                    "chunk_size": 4,
                    "backend": "chunked",
                    "mesh_method": "points",
                    "mesh_from_sdf": True,
                    "sdf_projection": True,
                    "sdf_backend": "dense",
                },
                artifact_root=Path(tmpdir),
            )
            result = backend.reconstruct(request)

        self.assertEqual(result.status, "success")
        extras = result.metric_result.extras
        self.assertEqual(extras["mesh_source"]["kind"], "signed_distance")
        self.assertEqual(
            extras["sdf_projection"]["sign_convention"],
            SIGN_CONVENTION,
        )
        self.assertEqual(extras["sdf_volume_metadata"]["value_type"], "signed_distance")
        self.assertIn("sdf_volume_metadata", result.artifacts)
        self.assertIn("sdf_volume_npz", result.artifacts)


if __name__ == "__main__":
    unittest.main()
