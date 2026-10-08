from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

import numpy as np

try:
    from utils.optional_deps import optional_policy_decision, probe_dependency
except Exception:  # pragma: no cover
    from ...utils.optional_deps import optional_policy_decision, probe_dependency

try:
    from primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from primitives.primitive_protocol import MeshData
    from primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from primitives.superfrustum import SuperFrustum
    from reconstruction.artifacts import write_json
except ImportError:  # pragma: no cover - package import path.
    from ...primitives.analytic_primitives import (
        AnisotropicGaussianPrimitive,
        EllipsoidPrimitive,
        SuperquadricPrimitive,
    )
    from ...primitives.primitive_protocol import MeshData
    from ...primitives.soft_silhouette import (
        OrthographicCamera,
        render_projected_soft_silhouette,
        soft_mask_metrics,
    )
    from ...primitives.superfrustum import SuperFrustum
    from ...reconstruction.artifacts import write_json
from .contracts import CameraSpec, GradientBatch, LossResult, LossWeights, ReconstructionTarget, RenderBatch, RenderableScene
from .losses import evaluate_render_loss
from .mesh_projection import _project_vertices_to_clip, _scene_mesh_arrays


class NvdiffrastBackend:
    """Lazy optional nvdiffrast adapter with explicit availability reporting."""

    name = "nvdiffrast"
    supports_gradients = True
    supports_silhouette = True
    supports_depth = True

    def __init__(self) -> None:
        self._module = None
        self._dr = None
        self._nvdiffrast = probe_dependency("nvdiffrast")
        self._torch = probe_dependency("torch")
        self._nvdiffrast_torch = None
        self._gpu_runtime_status: dict[str, object] = {
            "available": False,
            "status": "not_checked",
            "message": "GPU runtime has not been checked",
        }
        self._unmet_dependencies: list[str] = []
        if not self._nvdiffrast.available:
            self._unmet_dependencies.append(self._nvdiffrast.skip_reason)
        if not self._torch.available:
            self._unmet_dependencies.append(self._torch.skip_reason)
        elif self._torch.module is not None:
            self._gpu_runtime_status = self._detect_torch_gpu_runtime(self._torch.module)
            if not bool(self._gpu_runtime_status.get("available")):
                self._unmet_dependencies.append(
                    "torch GPU runtime: "
                    f"{self._gpu_runtime_status.get('message', 'unavailable')}"
                )
        self._module = self._nvdiffrast.module if self._nvdiffrast.available else None
        if self._module is not None:
            self._nvdiffrast_torch = probe_dependency("nvdiffrast.torch")
            if self._nvdiffrast_torch.available:
                self._dr = self._nvdiffrast_torch.module
            else:
                self._unmet_dependencies.append(self._nvdiffrast_torch.skip_reason)
        self.unavailable_reason: str | None = None
        if self._unmet_dependencies:
            self.unavailable_reason = "; ".join(self._unmet_dependencies)
        self._contexts: dict[str, object] = {}
        self._last_render_batch: RenderBatch | None = None
        self._last_tensors: dict[str, object] = {}

    @property
    def available(self) -> bool:
        return (
            self._module is not None
            and self._dr is not None
            and self._torch.available
            and bool(self._gpu_runtime_status.get("available"))
        )

    @property
    def dependency_report(self) -> str:
        if self.available:
            return "dependencies satisfied"
        return "; ".join(self._unmet_dependencies)

    @property
    def dependency_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "nvdiffrast": self._nvdiffrast.to_dict(),
            "torch": self._torch.to_dict(),
            "gpu_runtime": dict(self._gpu_runtime_status),
        }
        if self._nvdiffrast_torch is not None:
            state["nvdiffrast.torch"] = self._nvdiffrast_torch.to_dict()
        return state

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "nvdiffrast is unavailable; use CpuSoftSilhouetteBackend or "
                f"BlenderFiniteDifferenceBackend instead ({self.dependency_report})"
            )

    def render(self, scene: RenderableScene, cameras: Sequence[CameraSpec]) -> RenderBatch:
        self._require_available()
        torch = self._torch.require()
        dr = self._dr
        vertices, faces = _scene_mesh_arrays(scene)
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError("nvdiffrast render requires a non-empty triangle mesh")
        device_name = "cuda"
        device = torch.device(device_name)
        faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=device)
        ctx = self._context_for_device(dr, torch, device_name)

        silhouettes: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        view_stats: list[dict[str, object]] = []
        tensor_records: dict[str, object] = {}
        for camera in cameras:
            width, height = (int(camera.image_size[0]), int(camera.image_size[1]))
            clip_vertices = _project_vertices_to_clip(vertices, camera)
            pos = torch.as_tensor(
                clip_vertices,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            rast, _ = dr.rasterize(
                ctx,
                pos,
                faces_tensor,
                resolution=[height, width],
            )
            hard_mask = (rast[..., 3:4] > 0).to(torch.float32)
            try:
                mask_tensor = dr.antialias(hard_mask, rast, pos, faces_tensor)[0, ..., 0]
            except Exception:
                mask_tensor = hard_mask[0, ..., 0]
            depth_tensor = torch.where(
                hard_mask[0, ..., 0] > 0.0,
                rast[0, ..., 2],
                torch.zeros_like(rast[0, ..., 2]),
            )
            mask_np = mask_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            depth_np = depth_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            silhouettes[camera.name] = mask_np
            depths[camera.name] = depth_np
            tensor_records[camera.name] = {
                "position": pos,
                "raster": rast,
                "mask": mask_tensor,
                "depth": depth_tensor,
            }
            view_stats.append(
                {
                    "view": camera.name,
                    "image_size": (width, height),
                    "coverage": float(mask_np.mean()),
                    "area": float(mask_np.sum()),
                    "device": device_name,
                }
            )

        metadata = {
            "backend": self.name,
            "device": device_name,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "camera_count": int(len(cameras)),
            "view_stats": view_stats,
            "gradient_path": "nvdiffrast.torch",
        }
        batch = RenderBatch(
            silhouettes=silhouettes,
            depths=depths,
            metadata=metadata,
        )
        self._last_render_batch = batch
        self._last_tensors = tensor_records
        return batch

    def loss(
        self,
        render_batch: RenderBatch,
        target: ReconstructionTarget,
        weights: LossWeights = LossWeights(),
        view_weights: Mapping[str, float] | None = None,
    ) -> LossResult:
        self._require_available()
        return evaluate_render_loss(render_batch, target, weights, view_weights=view_weights)

    def backward(self, loss: LossResult) -> GradientBatch:
        self._require_available()
        _ = loss
        return GradientBatch(
            gradients={},
            epsilon=0.0,
            warnings=(
                "nvdiffrast raster tensors were retained for external torch "
                "optimization; scalar LossResult gradients are not materialized by "
                "this candidate wrapper",
            ),
        )

    def _context_for_device(self, dr: Any, torch: Any, device_name: str) -> object:
        cached = self._contexts.get(device_name)
        if cached is not None:
            return cached
        errors: list[str] = []
        if device_name == "cuda" and hasattr(dr, "RasterizeCudaContext"):
            try:
                context = dr.RasterizeCudaContext(device=torch.device(device_name))
                self._contexts[device_name] = context
                return context
            except Exception as exc:
                errors.append(f"RasterizeCudaContext: {exc}")
        if hasattr(dr, "RasterizeGLContext"):
            try:
                context = dr.RasterizeGLContext()
                self._contexts[device_name] = context
                return context
            except Exception as exc:
                errors.append(f"RasterizeGLContext: {exc}")
        raise RuntimeError("no usable nvdiffrast raster context: " + "; ".join(errors))

    @staticmethod
    def _detect_torch_gpu_runtime(torch: Any) -> dict[str, object]:
        version = getattr(torch, "version", None)
        cuda_version = getattr(version, "cuda", None)
        hip_version = getattr(version, "hip", None)
        cuda_api = getattr(torch, "cuda", None)
        cuda_available = False
        device_count = 0
        try:
            cuda_available = (
                bool(cuda_api.is_available()) if cuda_api is not None else False
            )
        except Exception as exc:
            return {
                "available": False,
                "status": "unusable",
                "message": f"torch.cuda availability check failed: {exc}",
                "error_type": type(exc).__name__,
                "cuda_version": cuda_version,
                "hip_version": hip_version,
            }
        try:
            device_count = (
                int(cuda_api.device_count())
                if cuda_api is not None and cuda_available
                else 0
            )
        except Exception:
            device_count = 0
        if hip_version and not cuda_version:
            return {
                "available": False,
                "status": "rocm_unsupported",
                "message": (
                    "ROCm/HIP PyTorch runtime detected; nvdiffrast has no official "
                    "ROCm backend and must not silently fall back to CPU"
                ),
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "device_count": device_count,
                "supports_rocm": False,
            }
        if not cuda_available:
            return {
                "available": False,
                "status": "cuda_unavailable",
                "message": (
                    "torch.cuda is unavailable; nvdiffrast requires an NVIDIA CUDA "
                    "runtime for this backend"
                ),
                "cuda_version": cuda_version,
                "hip_version": hip_version,
                "device_count": device_count,
                "supports_rocm": False,
            }
        return {
            "available": True,
            "status": "cuda_available",
            "message": "torch CUDA runtime is available for nvdiffrast",
            "cuda_version": cuda_version,
            "hip_version": hip_version,
            "device_count": device_count,
            "supports_rocm": False,
        }
