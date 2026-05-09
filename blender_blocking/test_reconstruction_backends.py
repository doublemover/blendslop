"""Pure tests for reconstruction backend registry and result contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any, Mapping, Sequence

import numpy as np

from config import ReconstructionConfig
from reconstruction.backend import BackendBudget, BackendCapabilities, BaseBackend
from reconstruction.candidate_scoring import select_best
from reconstruction.backends.gaussian_ellipsoid import GaussianEllipsoidBackend
from reconstruction.ensemble import CandidateConfig, EnsembleRunner
from reconstruction.registry import (
    clear_backends_for_tests,
    get_backend,
    list_backend_aliases,
    list_backends,
    register_backend,
    register_builtin_backends,
)
from reconstruction.backends.shape_program import (
    _compiled_appearance_summary,
    build_shape_program_from_target,
)
from reconstruction.types import (
    Bounds3D,
    CandidateMetrics,
    CandidateRequest,
    CandidateResult,
    ProfileBand,
    ProfileIntervalPx,
)
from reconstruction.types import ReconstructionTarget


_CAPABILITY_KEYS = {
    "requires_blender",
    "supports_pure_python",
    "supports_multi_view",
    "supports_top_view",
    "supports_uncertainty",
    "supports_constraints",
    "outputs_mesh",
    "outputs_volume",
    "outputs_primitive_set",
    "supports_gradients",
    "editability_score",
    "optional_dependencies",
}


class _FakeBackend(BaseBackend):
    def __init__(self, name: str = "fake_plugin") -> None:
        super().__init__(
            name=name,
            version="test",
            capabilities=BackendCapabilities(
                supports_multi_view=True,
                outputs_mesh=True,
                optional_dependencies=("__blendslop_missing_optional__",),
                editability_score=0.42,
            ),
        )

    def estimate_budget(
        self, target: Any, config: Mapping[str, Any]
    ) -> BackendBudget:
        return BackendBudget(estimated_seconds=0.01, notes=("fake budget",))

    def reconstruct(self, request: CandidateRequest) -> CandidateResult:
        sleep_s = float(request.config.get("sleep_s", 0.0) or 0.0)
        if sleep_s > 0.0:
            import time

            time.sleep(sleep_s)
        dependency_report = {
            name: {
                "available": False,
                "policy": request.config.get("optional_dependency_policy", "skip"),
            }
            for name in self.capabilities.optional_dependencies
        }
        if request.config.get("force_missing_dependency"):
            message = "optional dependency '__blendslop_missing_optional__' is unavailable"
            metrics = CandidateMetrics(
                extras={"optional_dependencies": dependency_report}
            )
            if request.config.get("optional_dependency_policy") == "fail":
                return CandidateResult(
                    candidate_id=request.candidate_id,
                    backend_name=self.name,
                    status="failed",
                    metric_result=metrics,
                    warnings=(message,),
                    errors=(message,),
                )
            return CandidateResult(
                candidate_id=request.candidate_id,
                backend_name=self.name,
                status="skipped",
                metric_result=metrics,
                warnings=(message,),
            )

        root = request.candidate_artifact_root()
        artifacts = {"manifest": root / "manifest.json"} if root is not None else {}
        degraded = bool(request.config.get("degraded", False))
        return CandidateResult(
            candidate_id=request.candidate_id,
            backend_name=self.name,
            status="degraded" if degraded else "success",
            metric_result=CandidateMetrics(
                per_view={"front": {"area_iou": 0.75, "boundary_iou": 0.5}},
                elapsed_s=0.01,
                extras={
                    "optional_dependencies": dependency_report,
                    "request_budget": request.budget.to_dict(),
                },
            ),
            artifacts=artifacts,
            warnings=("fake-warning",),
            errors=(),
            degraded=degraded,
        )


class _FakeUVLayers:
    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count


class _FakeMeshData:
    def __init__(self, *, polygons: int, uv_layers: int) -> None:
        self.polygons = [object() for _ in range(polygons)]
        self.uv_layers = _FakeUVLayers(uv_layers)


class _FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)


class _FakeNode:
    def __init__(self, node_type: str, name: str, image: Any = None) -> None:
        self.type = node_type
        self.name = name
        self.image = image


class _FakeNodeTree:
    def __init__(self, nodes: Sequence[Any]) -> None:
        self.nodes = tuple(nodes)


class _FakeMaterial:
    def __init__(self, name: str, *, image: Any = None) -> None:
        self.name = name
        self.diffuse_color = (1.0, 1.0, 1.0, 1.0)
        self.roughness = 0.5
        self.metallic = 0.0
        self.node_tree = _FakeNodeTree(
            (
                _FakeNode("TEX_IMAGE", "Base Color", image=image),
                _FakeNode("NORMAL_MAP", "Normal Map"),
            )
        )


class _FakeMaterialSlot:
    def __init__(self, material: Any) -> None:
        self.material = material


class _FakeObject:
    def __init__(
        self,
        *,
        name: str,
        polygons: int,
        uv_layers: int,
        materials: Sequence[Any] = (),
    ) -> None:
        self.name = name
        self.type = "MESH"
        self.data = _FakeMeshData(polygons=polygons, uv_layers=uv_layers)
        self.material_slots = tuple(_FakeMaterialSlot(material) for material in materials)


class _FakeCompiled:
    def __init__(self, objects: Sequence[Any]) -> None:
        self.objects = tuple(objects)


class ReconstructionBackendRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_backends_for_tests()

    def tearDown(self) -> None:
        clear_backends_for_tests()

    def test_builtin_registry_names_capabilities_and_optional_metadata(self) -> None:
        register_builtin_backends()

        infos = {info.name: info for info in list_backends()}
        self.assertEqual(
            set(infos),
            {
                "legacy",
                "profile_loft",
                "silhouette_intersection",
                "visual_hull_voxel",
                "hybrid_loft_hull",
                "primitive_fit_refine",
                "gaussian_ellipsoid_proxy",
                "differentiable_refine",
                "shape_program",
            },
        )
        self.assertEqual(list_backend_aliases(), {"loft_profile": "profile_loft"})

        expected_capabilities = {
            "legacy": {
                "requires_blender": True,
                "supports_pure_python": False,
                "supports_multi_view": False,
                "supports_top_view": False,
                "supports_uncertainty": False,
                "supports_constraints": False,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": False,
                "supports_gradients": False,
                "editability_score": 0.25,
                "optional_dependencies": [],
            },
            "profile_loft": {
                "requires_blender": True,
                "supports_pure_python": False,
                "supports_multi_view": True,
                "supports_top_view": True,
                "supports_uncertainty": True,
                "supports_constraints": False,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": False,
                "supports_gradients": False,
                "editability_score": 0.55,
                "optional_dependencies": [],
            },
            "silhouette_intersection": {
                "requires_blender": True,
                "supports_pure_python": False,
                "supports_multi_view": True,
                "supports_top_view": False,
                "supports_uncertainty": False,
                "supports_constraints": True,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": False,
                "supports_gradients": False,
                "editability_score": 0.35,
                "optional_dependencies": [],
            },
            "visual_hull_voxel": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": True,
                "supports_uncertainty": True,
                "supports_constraints": False,
                "outputs_mesh": True,
                "outputs_volume": True,
                "outputs_primitive_set": False,
                "supports_gradients": False,
                "editability_score": 0.15,
                "optional_dependencies": ["skimage", "open3d", "openvdb"],
            },
            "hybrid_loft_hull": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": True,
                "supports_uncertainty": True,
                "supports_constraints": False,
                "outputs_mesh": True,
                "outputs_volume": True,
                "outputs_primitive_set": False,
                "supports_gradients": False,
                "editability_score": 0.45,
                "optional_dependencies": [],
            },
            "primitive_fit_refine": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": False,
                "supports_uncertainty": True,
                "supports_constraints": True,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": True,
                "supports_gradients": False,
                "editability_score": 0.9,
                "optional_dependencies": [],
            },
            "gaussian_ellipsoid_proxy": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": True,
                "supports_uncertainty": True,
                "supports_constraints": True,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": True,
                "supports_gradients": False,
                "editability_score": 0.75,
                "optional_dependencies": [],
            },
            "differentiable_refine": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": False,
                "supports_uncertainty": True,
                "supports_constraints": True,
                "outputs_mesh": True,
                "outputs_volume": False,
                "outputs_primitive_set": False,
                "supports_gradients": True,
                "editability_score": 0.65,
                "optional_dependencies": ["nvdiffrast", "torch"],
            },
            "shape_program": {
                "requires_blender": False,
                "supports_pure_python": True,
                "supports_multi_view": True,
                "supports_top_view": True,
                "supports_uncertainty": True,
                "supports_constraints": True,
                "outputs_mesh": False,
                "outputs_volume": False,
                "outputs_primitive_set": True,
                "supports_gradients": False,
                "editability_score": 0.95,
                "optional_dependencies": [],
            },
        }
        for name, expected in expected_capabilities.items():
            with self.subTest(backend=name):
                payload = infos[name].capabilities.to_dict()
                self.assertEqual(set(payload), _CAPABILITY_KEYS)
                self.assertEqual(payload, expected)
                self.assertEqual(infos[name].to_dict()["capabilities"], expected)
                self.assertTrue(infos[name].version)

    def test_shape_program_preserves_profile_curve_rows(self) -> None:
        bands = (
            ProfileBand(
                t=0.0,
                intervals=(ProfileIntervalPx(10.0, 20.0),),
                center_x=15.0,
                width_px=10.0,
                source_view="front",
            ),
            ProfileBand(
                t=0.5,
                intervals=(ProfileIntervalPx(6.0, 24.0),),
                center_x=15.0,
                width_px=18.0,
                source_view="front",
            ),
            ProfileBand(
                t=1.0,
                intervals=(ProfileIntervalPx(11.0, 19.0),),
                center_x=15.0,
                width_px=8.0,
                source_view="front",
            ),
        )
        target = ReconstructionTarget(
            profile_bands={"front": bands},
            bounds=Bounds3D.from_min_max((-1.0, -0.5, 0.0), (1.0, 0.5, 2.0)),
        )

        program, diagnostics = build_shape_program_from_target(
            target,
            config={"root_strategy": "profile_lathe", "residual_policy": "report"},
            program_id="shape-curve-test",
        )

        curve = program.root_nodes[0].parameters["profile_curve"]
        self.assertEqual(len(curve), 3)
        self.assertEqual(diagnostics["profile_curve_rows"], 3)
        self.assertGreater(curve[1]["radius_x_world"], curve[0]["radius_x_world"])
        self.assertGreater(curve[1]["radius_y_world"], curve[2]["radius_y_world"])

    def test_shape_program_realizes_residual_patch_nodes(self) -> None:
        bands = (
            ProfileBand(
                t=0.25,
                intervals=(
                    ProfileIntervalPx(4.0, 12.0),
                    ProfileIntervalPx(28.0, 36.0),
                ),
                center_x=20.0,
                width_px=32.0,
                source_view="front",
            ),
            ProfileBand(
                t=0.75,
                intervals=(ProfileIntervalPx(5.0, 35.0),),
                holes=(ProfileIntervalPx(17.0, 23.0),),
                center_x=20.0,
                width_px=30.0,
                source_view="front",
                confidence=0.7,
            ),
        )
        target = ReconstructionTarget(
            profile_bands={"front": bands},
            bounds=Bounds3D.from_min_max((-1.0, -0.5, -1.0), (1.0, 0.5, 1.0)),
        )

        program, diagnostics = build_shape_program_from_target(
            target,
            config={
                "root_strategy": "profile_lathe",
                "residual_policy": "suggest_patches",
                "max_nodes": 8,
            },
            program_id="shape-residual-node-test",
        )

        residual_nodes = [
            node for node in program.root_nodes if node.node_id.startswith("residual_")
        ]
        self.assertEqual(len(program.residual_patches), 2)
        self.assertEqual(len(residual_nodes), 2)
        self.assertEqual(diagnostics["suggested_residual_node_count"], 2)
        self.assertEqual(diagnostics["realized_residual_node_count"], 2)
        self.assertEqual(
            diagnostics["realized_residual_node_ids"],
            [node.node_id for node in residual_nodes],
        )
        self.assertTrue(
            all(patch.suggested_node is not None for patch in program.residual_patches)
        )
        self.assertEqual(residual_nodes[0].operation, "attach")
        self.assertEqual(residual_nodes[1].operation, "difference")
        self.assertEqual(residual_nodes[0].parameters["interval_count"], 2)
        self.assertEqual(residual_nodes[1].parameters["hole_count"], 1)
        self.assertEqual(
            residual_nodes[1].parameters["suggested_boolean_role"],
            "subtract",
        )

    def test_gaussian_proxy_distills_to_editable_shape_program(self) -> None:
        surface_points = np.array(
            [
                [-1.0, -0.5, -0.25],
                [-1.0, 0.5, -0.25],
                [1.0, -0.5, 0.25],
                [1.0, 0.5, 0.25],
                [-0.5, 0.0, 0.75],
                [0.5, 0.0, 0.75],
                [0.0, -0.25, -0.75],
                [0.0, 0.25, -0.75],
            ],
            dtype=float,
        )
        target = ReconstructionTarget(extras={"surface_points": surface_points})

        with tempfile.TemporaryDirectory() as tmp:
            result = GaussianEllipsoidBackend().reconstruct(
                CandidateRequest(
                    candidate_id="gaussian-editable",
                    backend_name="gaussian_ellipsoid_proxy",
                    target=target,
                    config={
                        "family": "gaussian",
                        "primitive_count": 2,
                        "target_point_count": 8,
                        "export_mesh_proxy": False,
                    },
                    artifact_root=Path(tmp),
                )
            )

        editable = result.metric_result.extras["editable_proxy"]
        program = editable["shape_program"]
        node = program["root_nodes"][0]
        stages = {
            record["stage"]
            for record in result.metric_result.extras["objective_history"]
        }

        self.assertEqual(result.status, "success")
        self.assertIn("editable_proxy_shape_program", result.artifacts)
        self.assertIn("proxy_distillation", result.artifacts)
        self.assertIn("proxy_distillation_npz", result.artifacts)
        self.assertEqual(editable["node_count"], 2)
        self.assertEqual(editable["validation_errors"], [])
        self.assertEqual(node["primitive_type"], "ellipsoid")
        self.assertEqual(len(node["parameters"]["rotation_row_major"]), 9)
        self.assertGreater(result.metric_result.editability_score, 0.7)
        self.assertIn("editable_proxy_distillation", stages)
        distillation = result.metric_result.extras["proxy_distillation"]
        self.assertGreater(distillation["arbitration_score"], 0.0)
        self.assertEqual(distillation["primitive_count"], 2)

    def test_shape_program_compiled_appearance_summary_reads_uv_materials(self) -> None:
        image = _FakeImage(256, 128)
        material = _FakeMaterial("BodyPaint", image=image)
        compiled = _FakeCompiled(
            (
                _FakeObject(
                    name="body",
                    polygons=12,
                    uv_layers=1,
                    materials=(material,),
                ),
            )
        )

        appearance = _compiled_appearance_summary(
            compiled,
            {
                "evaluate_texture_materials": True,
                "uv_strict": True,
                "material_target": "pbr",
                "max_texture_memory_mb": 1.0,
            },
        )

        self.assertIsNotNone(appearance)
        self.assertEqual(appearance["uv"]["has_uv_map"], True)
        self.assertEqual(appearance["uv"]["missing_uv_faces"], 0)
        self.assertEqual(appearance["materials"]["material_slot_count"], 1)
        self.assertEqual(appearance["materials"]["named_material_ratio"], 1.0)
        self.assertEqual(
            appearance["materials"]["pbr_channel_coverage"],
            {
                "base_color": True,
                "roughness": True,
                "metallic": True,
                "normal": True,
            },
        )
        self.assertGreater(appearance["texture"]["texture_memory_mb"], 0.0)
        self.assertNotIn("errors", appearance)

    def test_shape_program_appearance_summary_fails_required_uncompiled_asset(self) -> None:
        appearance = _compiled_appearance_summary(
            None,
            {
                "evaluate_texture_materials": True,
                "uv_strict": True,
                "material_target": "pbr",
            },
        )

        self.assertEqual(appearance["uv_valid"], False)
        self.assertIn("compiled_blender_asset_missing", appearance["errors"])

    def test_shape_program_backend_validates_appearance_options(self) -> None:
        from reconstruction.backends.shape_program import ShapeProgramBackend

        backend = ShapeProgramBackend()

        self.assertEqual(
            backend.validate_config(
                {
                    "evaluate_texture_materials": True,
                    "uv_strict": True,
                    "material_target": "pbr",
                    "max_texture_memory_mb": 64.0,
                    "texture_reference_dir": "temp/texture_refs",
                }
            ),
            [],
        )
        self.assertIn(
            "shape_program.material_target must be pbr/simple/none",
            backend.validate_config({"material_target": "radiance"}),
        )

    def test_duplicate_missing_and_alias_registration_errors(self) -> None:
        backend = _FakeBackend()
        register_backend(backend, aliases=("fake_alias",))

        self.assertIs(get_backend("fake_plugin"), backend)
        self.assertIs(get_backend("fake_alias"), backend)
        self.assertEqual(list_backend_aliases(), {"fake_alias": "fake_plugin"})

        with self.assertRaisesRegex(ValueError, "fake_plugin"):
            register_backend(_FakeBackend())
        with self.assertRaisesRegex(ValueError, "fake_alias"):
            register_backend(_FakeBackend("other_fake"), aliases=("fake_alias",))
        with self.assertRaisesRegex(ValueError, "conflicts with registered backend"):
            register_backend(_FakeBackend("alias_conflict"), aliases=("fake_plugin",))
        with self.assertRaisesRegex(ValueError, "duplicates canonical"):
            register_backend(_FakeBackend("self_alias"), aliases=("self_alias",))
        with self.assertRaisesRegex(KeyError, "unknown backend 'missing_backend'"):
            get_backend("missing_backend")

    def test_alias_behavior_from_config_and_candidate_request(self) -> None:
        register_builtin_backends()
        ReconstructionConfig(reconstruction_mode="loft_profile").validate()
        self.assertIs(get_backend("loft_profile"), get_backend("profile_loft"))

        clear_backends_for_tests()
        register_backend(_FakeBackend("canonical_fake"), aliases=("request_alias",))
        result = EnsembleRunner().run(
            target=ReconstructionTarget(),
            candidates=(
                CandidateConfig(
                    backend_name="request_alias",
                    candidate_id="alias-request",
                ),
            ),
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].status, "success")
        self.assertEqual(result.candidates[0].backend_name, "canonical_fake")

    def test_fake_backend_optional_dependency_skip_and_fail_policy(self) -> None:
        register_backend(_FakeBackend())
        runner = EnsembleRunner()
        requests = runner.build_requests(
            target=ReconstructionTarget(),
            candidates=(
                CandidateConfig(
                    backend_name="fake_plugin",
                    candidate_id="skip-missing",
                    config={
                        "force_missing_dependency": True,
                        "optional_dependency_policy": "skip",
                    },
                ),
                CandidateConfig(
                    backend_name="fake_plugin",
                    candidate_id="fail-missing",
                    config={
                        "force_missing_dependency": True,
                        "optional_dependency_policy": "fail",
                    },
                ),
            ),
        )

        result = runner.run_requests(requests)
        by_id = {candidate.candidate_id: candidate for candidate in result.candidates}

        self.assertEqual(by_id["skip-missing"].status, "skipped")
        self.assertIn("__blendslop_missing_optional__", by_id["skip-missing"].warnings[0])
        self.assertEqual(by_id["skip-missing"].errors, ())
        self.assertFalse(by_id["skip-missing"].succeeded)
        self.assertEqual(by_id["fail-missing"].status, "failed")
        self.assertIn("__blendslop_missing_optional__", by_id["fail-missing"].errors[0])
        self.assertFalse(by_id["fail-missing"].succeeded)

    def test_ensemble_total_timeout_bounds_candidates_and_skips_remaining(self) -> None:
        register_backend(_FakeBackend())
        runner = EnsembleRunner()
        requests = runner.build_requests(
            target=ReconstructionTarget(),
            candidates=(
                CandidateConfig(
                    backend_name="fake_plugin",
                    candidate_id="slow-first",
                    config={"sleep_s": 0.02},
                ),
                CandidateConfig(
                    backend_name="fake_plugin",
                    candidate_id="skipped-second",
                ),
            ),
        )

        result = runner.run_requests(requests, total_timeout_s=0.001)
        by_id = {candidate.candidate_id: candidate for candidate in result.candidates}

        self.assertEqual(by_id["slow-first"].status, "success")
        self.assertTrue(by_id["slow-first"].degraded)
        self.assertIn(
            "ensemble total timeout elapsed during candidate",
            by_id["slow-first"].warnings,
        )
        self.assertLessEqual(
            by_id["slow-first"].metric_result.extras["request_budget"]["timeout_s"],
            0.001,
        )
        self.assertEqual(by_id["skipped-second"].status, "skipped")
        self.assertIn("total timeout exhausted", by_id["skipped-second"].warnings[0])

    def test_candidate_artifact_root_is_scoped_and_rejects_escape(self) -> None:
        self.assertIsNone(
            CandidateRequest(
                candidate_id="no-root",
                backend_name="fake_plugin",
                target=ReconstructionTarget(),
            ).candidate_artifact_root()
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = CandidateRequest(
                candidate_id="candidate-01",
                backend_name="fake_plugin",
                target=ReconstructionTarget(),
                artifact_root=root,
            )
            self.assertEqual(
                request.candidate_artifact_root(),
                (root / "candidate-01").resolve(),
            )

            escaping_candidate_ids = (
                "../escape",
                str(root.parent / "absolute-escape"),
            )
            for candidate_id in escaping_candidate_ids:
                with self.subTest(candidate_id=candidate_id):
                    escaping = CandidateRequest(
                        candidate_id=candidate_id,
                        backend_name="fake_plugin",
                        target=ReconstructionTarget(),
                        artifact_root=root,
                    )
                    with self.assertRaisesRegex(ValueError, "escapes artifact root"):
                        escaping.candidate_artifact_root()

    def test_fake_backend_result_serializes_required_metadata_fields(self) -> None:
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            request = CandidateRequest(
                candidate_id="result-shape",
                backend_name=backend.name,
                target=ReconstructionTarget(),
                config={"degraded": True},
                artifact_root=Path(tmp),
            )
            result = backend.reconstruct(request)

        payload = result.to_dict()
        self.assertEqual(result.status, "degraded")
        self.assertTrue(result.succeeded)
        self.assertEqual(
            set(payload),
            {
                "candidate_id",
                "backend_name",
                "status",
                "mesh_path",
                "primitive_path",
                "volume_path",
                "render_paths",
                "metric_result",
                "artifacts",
                "warnings",
                "errors",
                "degraded",
            },
        )
        self.assertEqual(payload["metric_result"]["area_iou_min"], 0.75)
        self.assertEqual(payload["metric_result"]["area_iou_mean"], 0.75)
        self.assertEqual(payload["metric_result"]["boundary_iou_mean"], 0.5)
        self.assertIn("optional_dependencies", payload["metric_result"]["extras"])
        self.assertIn("manifest", payload["artifacts"])
        self.assertEqual(payload["warnings"], ["fake-warning"])
        self.assertEqual(payload["errors"], [])
        self.assertTrue(payload["degraded"])

    def test_quality_first_selection_penalizes_hidden_view_failures(self) -> None:
        pretty_bad = CandidateResult(
            candidate_id="pretty-bad",
            backend_name="gaussian_ellipsoid_proxy",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.95,
                        "boundary_iou": 0.80,
                        "signed_distance_loss": 0.04,
                        "required": True,
                        "passed": True,
                    },
                    "top": {
                        "area_iou": 0.05,
                        "boundary_iou": 0.02,
                        "signed_distance_loss": 0.70,
                        "required": True,
                        "passed": False,
                    },
                },
                editability_score=0.9,
            ),
        )
        boring_good = CandidateResult(
            candidate_id="boring-good",
            backend_name="visual_hull_voxel",
            status="success",
            metric_result=CandidateMetrics(
                per_view={
                    "front": {
                        "area_iou": 0.76,
                        "boundary_iou": 0.62,
                        "signed_distance_loss": 0.10,
                        "required": True,
                        "passed": True,
                    },
                    "top": {
                        "area_iou": 0.74,
                        "boundary_iou": 0.60,
                        "signed_distance_loss": 0.11,
                        "required": True,
                        "passed": True,
                    },
                },
                topology_score=0.7,
                editability_score=0.4,
            ),
        )

        selected, ranked = select_best(
            (pretty_bad, boring_good),
            policy="quality_first",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.candidate_id, "boring-good")
        scores = {score.candidate_id: score for _, score in ranked}
        bad_terms = {term.name: term.weighted for term in scores["pretty-bad"].terms}
        self.assertLess(bad_terms["failed_required_views"], 0.0)


if __name__ == "__main__":
    unittest.main()
