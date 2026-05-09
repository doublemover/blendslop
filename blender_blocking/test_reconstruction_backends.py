"""Pure tests for reconstruction backend registry and result contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any, Mapping

from config import ReconstructionConfig
from reconstruction.backend import BackendBudget, BackendCapabilities, BaseBackend
from reconstruction.ensemble import CandidateConfig, EnsembleRunner
from reconstruction.registry import (
    clear_backends_for_tests,
    get_backend,
    list_backend_aliases,
    list_backends,
    register_backend,
    register_builtin_backends,
)
from reconstruction.backends.shape_program import build_shape_program_from_target
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
                extras={"optional_dependencies": dependency_report},
            ),
            artifacts=artifacts,
            warnings=("fake-warning",),
            errors=(),
            degraded=degraded,
        )


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


if __name__ == "__main__":
    unittest.main()
