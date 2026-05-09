#!/usr/bin/env python3
"""Pure tests for volume grid contracts and interchange."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

from reconstruction.backends.visual_hull import VisualHullBackend, _postprocess_mesh
from reconstruction.point_cloud import visual_hull_grid_from_target
from reconstruction.types import (
    CandidateRequest,
    OrthographicCameraSpec,
    Bounds3D as ReconBounds3D,
    ReconstructionTarget,
    ViewConstraint,
)
from volume import (
    Bounds3D,
    ChunkKey,
    ChunkedVolumeGrid,
    DenseVolumeGrid,
    SparseHashVolumeGrid,
    OpenVDBVolumeGrid,
    MeshExtractionResult,
    VoxelTransform,
    export_to_openvdb,
    extract_mesh,
    extract_surface_voxels,
    import_from_openvdb,
    load_volume,
    save_volume,
)


def _build_full_view_target() -> ReconstructionTarget:
    camera = OrthographicCameraSpec(view_name="front", axis="z", azimuth_deg=0.0)
    return ReconstructionTarget(
        constraints=(
            ViewConstraint(
                view="front",
                mask=np.ones((8, 8), dtype=bool),
                camera=camera,
            ),
        ),
        bounds=ReconBounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )


class VolumeGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = Bounds3D(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def test_dense_sampling_and_chunks(self) -> None:
        data = np.zeros((4, 4, 4), dtype=bool)
        data[1, 1, 1] = True
        grid = DenseVolumeGrid(data, self.bounds, chunk_size=2)

        world_point = grid.transform.index_to_world(np.array([[1, 1, 1]]))
        self.assertTrue(bool(grid.sample_world(world_point)[0]))
        self.assertEqual(grid.active_voxel_count(), 1)
        self.assertEqual(len(list(grid.iter_active_chunks())), 1)

    def test_chunked_dense_conversion(self) -> None:
        data = np.zeros((5, 4, 3), dtype=np.float32)
        data[4, 3, 2] = 0.75
        grid = ChunkedVolumeGrid.from_dense(
            data,
            self.bounds,
            value_type="occupancy_prob",
            default_value=0.0,
            chunk_size=2,
        )

        np.testing.assert_array_equal(grid.to_dense(), data)
        self.assertEqual(grid.active_voxel_count(), 1)
        self.assertEqual(grid.get_chunk(ChunkKey(2, 1, 1)).shape, (2, 2, 2))

    def test_sparse_hash_round_trip(self) -> None:
        data = np.zeros((6, 6, 6), dtype=bool)
        data[1:3, 1:3, 1:3] = True
        grid = SparseHashVolumeGrid.from_dense(data, self.bounds, chunk_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = save_volume(grid, tmpdir, generation_seed=123)
            loaded = load_volume(tmpdir)

        self.assertEqual(metadata.backend, "sparse_hash")
        np.testing.assert_array_equal(loaded.to_dense(), data)

    def test_vectorized_surface_extraction(self) -> None:
        cube = np.ones((3, 3, 3), dtype=bool)
        surface = extract_surface_voxels(cube, prefer_scipy=False)
        self.assertEqual(int(surface.sum()), 26)
        self.assertFalse(bool(surface[1, 1, 1]))

    def test_meshing_unavailable_or_structured_result(self) -> None:
        data = np.zeros((3, 3, 3), dtype=bool)
        data[1, 1, 1] = True
        grid = DenseVolumeGrid(data, self.bounds)
        result = extract_mesh(grid, method="point_cloud_only")

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.method, "points")
        self.assertEqual(result.requested_method, "point_cloud_only")
        self.assertEqual(result.faces.shape, (0, 3))

    def test_marching_cubes_pads_full_occupancy_volume(self) -> None:
        try:
            import skimage  # noqa: F401
        except Exception as exc:
            self.skipTest(f"skimage unavailable: {exc}")

        data = np.ones((4, 4, 4), dtype=bool)
        grid = DenseVolumeGrid(data, self.bounds)
        result = extract_mesh(grid, method="marching_cubes")

        self.assertEqual(result.status, "ok")
        self.assertGreater(int(result.vertices.shape[0]), 0)
        self.assertGreater(int(result.faces.shape[0]), 0)
        mins = result.vertices.min(axis=0)
        maxs = result.vertices.max(axis=0)
        np.testing.assert_allclose(mins, np.array([-1.0, -1.0, -1.0]), atol=1e-6)
        np.testing.assert_allclose(maxs, np.array([1.0, 1.0, 1.0]), atol=1e-6)

    def test_direct_visual_hull_chunked_sparse_creation(self) -> None:
        target = _build_full_view_target()
        dense_grid = visual_hull_grid_from_target(
            target,
            resolution=6,
            chunk_size=2,
            use_vectorized=True,
            backend="dense",
        )
        dense_data = dense_grid.to_dense()

        chunked = visual_hull_grid_from_target(
            target,
            resolution=6,
            chunk_size=2,
            use_vectorized=True,
            backend="chunked",
        )
        sparse = visual_hull_grid_from_target(
            target,
            resolution=6,
            chunk_size=2,
            use_vectorized=True,
            backend="sparse_hash",
        )

        self.assertIsInstance(chunked, ChunkedVolumeGrid)
        self.assertIsInstance(sparse, SparseHashVolumeGrid)
        self.assertNotIsInstance(chunked, DenseVolumeGrid)
        self.assertNotIsInstance(sparse, DenseVolumeGrid)
        np.testing.assert_array_equal(chunked.to_dense(), dense_data)
        np.testing.assert_array_equal(sparse.to_dense(), dense_data)
        self.assertEqual(chunked.active_voxel_count(), dense_grid.active_voxel_count())
        self.assertEqual(sparse.active_voxel_count(), dense_grid.active_voxel_count())

    def test_direct_visual_hull_openvdb_creation_marks_interchange(self) -> None:
        target = _build_full_view_target()
        grid = visual_hull_grid_from_target(
            target,
            resolution=6,
            chunk_size=2,
            use_vectorized=True,
            backend="openvdb",
        )

        self.assertIsInstance(grid, OpenVDBVolumeGrid)
        self.assertEqual(grid.backend, "openvdb")
        metadata = grid.openvdb_metadata()
        self.assertEqual(metadata["storage_backend"], "sparse_hash")
        self.assertEqual(metadata["serialization"], "npz_interchange")

    def test_mesh_extraction_method_normalization_and_metadata(self) -> None:
        data = np.zeros((3, 3, 3), dtype=bool)
        data[1, 1, 1] = True
        grid = DenseVolumeGrid(data, self.bounds)
        result = extract_mesh(grid, method="POiNT_CLOUD")

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.method, "points")
        self.assertEqual(result.requested_method, "POiNT_CLOUD")
        payload = result.to_dict()

        self.assertEqual(payload["status"], "skipped")
        self.assertEqual(payload["method"], "points")
        self.assertEqual(payload["requested_method"], "POiNT_CLOUD")
        self.assertEqual(payload["vertices"], int(result.vertices.shape[0]))
        self.assertIn("surface_points", payload["metrics"])
        self.assertEqual(payload["topology"]["topology_style"], "none")

    def test_visual_hull_postprocess_warning_status(self) -> None:
        backend = VisualHullBackend()
        target = _build_full_view_target()

        for postprocess, expect_warning in (("none", False), ("poisson", True)):
            with (
                self.subTest(postprocess=postprocess),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                request = CandidateRequest(
                    candidate_id=f"vh-post-{postprocess}",
                    backend_name="visual_hull_voxel",
                    target=target,
                    config={
                        "resolution": 6,
                        "chunk_size": 2,
                        "backend": "chunked",
                        "mesh_method": "points",
                        "postprocess": postprocess,
                    },
                    artifact_root=Path(tmpdir),
                )
                result = backend.reconstruct(request)
            self.assertEqual(result.status, "success")
            postprocess_status = result.metric_result.extras["mesh_postprocess"]
            self.assertEqual(postprocess_status["method"], postprocess)
            self.assertEqual(postprocess_status["status"], "skipped")
            has_postprocess_warning = any(
                "postprocess" in warning for warning in result.warnings
            )
            self.assertEqual(has_postprocess_warning, expect_warning)

    def test_visual_hull_topology_repair_postprocess_is_pure_and_structured(self) -> None:
        mesh = MeshExtractionResult(
            status="ok",
            method="fixture",
            requested_method="fixture",
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [4.0, 4.0, 4.0],
                ],
                dtype=float,
            ),
            faces=np.array(
                [
                    [0, 1, 2],
                    [0, 1, 1],
                ],
                dtype=np.int64,
            ),
        )

        repaired, status = _postprocess_mesh(
            mesh,
            "topology_repair",
            config={},
        )

        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["implementation"], "metrics.topology.safe_topology_repair")
        self.assertTrue(status["repair"]["changed"])
        self.assertEqual(status["repair"]["after"]["loose_vertices"], 0)
        self.assertEqual(len(repaired.vertices), 3)
        self.assertEqual(len(repaired.faces), 1)
        self.assertEqual(repaired.topology["degenerate_faces"], 0)

    def test_visual_hull_mesh_unavailable_degrades_or_fails_without_point_fallback(self) -> None:
        backend = VisualHullBackend()
        target = _build_full_view_target()

        for require_mesh, expected_status in ((False, "degraded"), (True, "failed")):
            with (
                self.subTest(require_mesh=require_mesh),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                request = CandidateRequest(
                    candidate_id=f"vh-mesh-required-{require_mesh}",
                    backend_name="visual_hull_voxel",
                    target=target,
                    config={
                        "resolution": 6,
                        "chunk_size": 2,
                        "backend": "chunked",
                        "mesh_method": "dual_contouring",
                        "require_mesh": require_mesh,
                    },
                    artifact_root=Path(tmpdir),
                )
                result = backend.reconstruct(request)

            self.assertEqual(result.status, expected_status)
            self.assertEqual(result.degraded, expected_status == "degraded")
            self.assertIn("mesh_extraction", result.metric_result.extras)
            self.assertIn("mesh extraction did not produce a mesh", "\n".join(result.warnings))
            if require_mesh:
                self.assertTrue(result.errors)

    def test_openvdb_metadata_round_trip(self) -> None:
        grid = OpenVDBVolumeGrid.unavailable(
            shape=(6, 6, 4),
            bounds=self.bounds,
            chunk_size=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = save_volume(
                grid,
                tmpdir,
                source_candidate_id="openvdb-case",
                source_masks=("front",),
                source_views=("front",),
                generation_seed=123,
                extra={"backend": "openvdb-test"},
            )
            loaded = load_volume(tmpdir)

        self.assertEqual(metadata.backend, "openvdb")
        self.assertEqual(metadata.source_candidate_id, "openvdb-case")
        self.assertEqual(metadata.source_masks, ("front",))
        self.assertEqual(metadata.source_views, ("front",))
        self.assertEqual(metadata.generation_seed, 123)
        self.assertEqual(metadata.extra["backend"], "openvdb-test")
        self.assertEqual(metadata.extra["storage_backend"], "sparse_hash")
        self.assertEqual(metadata.extra["serialization"], "npz_interchange")
        self.assertIn("openvdb", metadata.extra)
        self.assertIn("metadata_without_hashes", metadata.hashes)
        self.assertIn("npz_sha256", metadata.hashes)
        self.assertEqual(loaded.backend, "openvdb")
        np.testing.assert_array_equal(loaded.to_dense(), np.zeros((6, 6, 4), dtype=bool))

    def test_openvdb_binding_export_import_round_trip_with_fake_module(self) -> None:
        fake_module = _FakeOpenVDBModule("pyopenvdb")
        previous_pyopenvdb = sys.modules.get("pyopenvdb")
        previous_openvdb = sys.modules.get("openvdb")
        sys.modules["pyopenvdb"] = fake_module
        sys.modules.pop("openvdb", None)
        try:
            data = np.zeros((4, 4, 4), dtype=bool)
            data[1:3, 1:3, 1:3] = True
            grid = OpenVDBVolumeGrid.from_dense(
                data,
                self.bounds,
                chunk_size=2,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "shape.vdb"
                export_status = export_to_openvdb(grid, path)
                self.assertEqual(export_status.status, "exported")
                self.assertTrue(path.exists())

                imported = import_from_openvdb(
                    path,
                    bounds=self.bounds,
                    transform=VoxelTransform.from_bounds_shape(self.bounds, data.shape),
                    chunk_size=2,
                )
            self.assertIsInstance(imported, OpenVDBVolumeGrid)
            self.assertEqual(imported.openvdb_status.status, "imported")
            np.testing.assert_array_equal(imported.to_dense(), data)
        finally:
            if previous_pyopenvdb is None:
                sys.modules.pop("pyopenvdb", None)
            else:
                sys.modules["pyopenvdb"] = previous_pyopenvdb
            if previous_openvdb is None:
                sys.modules.pop("openvdb", None)
            else:
                sys.modules["openvdb"] = previous_openvdb


class _FakeOpenVDBModule(types.ModuleType):
    __version__ = "test"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._last_grid = None

    class BoolGrid:
        def __init__(self, background: bool = False) -> None:
            self.background = background
            self.name = ""
            self.active_voxels = {}

        def getAccessor(self):
            return self

        def setValueOn(self, coord, value) -> None:
            self.active_voxels[tuple(int(v) for v in coord)] = bool(value)

        def iter_active_values(self):
            return tuple(self.active_voxels.items())

    class FloatGrid(BoolGrid):
        def setValueOn(self, coord, value) -> None:
            self.active_voxels[tuple(int(v) for v in coord)] = float(value)

    def createLinearTransform(self, voxelSize=1.0):
        return {"voxelSize": float(voxelSize)}

    def write(self, path: str, grids) -> None:
        self._last_grid = list(grids)[0]
        Path(path).write_text("fake openvdb", encoding="utf-8")

    def read(self, path: str):
        _ = Path(path).read_text(encoding="utf-8")
        return [self._last_grid]


if __name__ == "__main__":
    unittest.main()
