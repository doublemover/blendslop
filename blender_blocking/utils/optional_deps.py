"""Lazy optional dependency detection helpers.

Ambitious reconstruction modes should report precise skip reasons when heavy
libraries are absent.  This module centralizes those probes so backends do not
hide optional import failures behind broad exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from types import ModuleType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


_VALID_OPTIONAL_POLICIES = {"skip", "fail"}


@dataclass(frozen=True)
class DependencySpec:
    """Static metadata for a known optional dependency."""

    name: str
    import_names: Tuple[str, ...] = ()
    package_name: Optional[str] = None
    pypi_name: Optional[str] = None
    purpose: str = ""
    install_hint: str = ""
    docs_url: str = ""
    source_url: str = ""
    optional_for: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    gpu_vendor: Optional[str] = None
    accelerator: Optional[str] = None
    supports_rocm: Optional[bool] = None
    rocm_alternative: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def candidates(self) -> Tuple[str, ...]:
        return self.import_names or (self.name,)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "import_names": list(self.candidates),
            "package_name": self.package_name or self.name,
            "pypi_name": self.pypi_name,
            "purpose": self.purpose,
            "install_hint": self.install_hint,
            "docs_url": self.docs_url,
            "source_url": self.source_url,
            "optional_for": list(self.optional_for),
            "aliases": list(self.aliases),
            "gpu_vendor": self.gpu_vendor,
            "accelerator": self.accelerator,
            "supports_rocm": self.supports_rocm,
            "rocm_alternative": self.rocm_alternative,
            "notes": list(self.notes),
        }


KNOWN_DEPENDENCIES: Dict[str, DependencySpec] = {
    "numpy": DependencySpec(
        name="numpy",
        purpose="core array operations",
        install_hint="pip install numpy",
    ),
    "cv2": DependencySpec(
        name="cv2",
        package_name="opencv-python",
        pypi_name="opencv-python",
        purpose="image loading and silhouette preprocessing",
        install_hint="pip install opencv-python",
        aliases=("opencv", "opencv-python"),
    ),
    "PIL": DependencySpec(
        name="PIL",
        package_name="Pillow",
        pypi_name="Pillow",
        purpose="image loading and Pillow C extension checks",
        install_hint="pip install Pillow",
        aliases=("pillow",),
    ),
    "scipy": DependencySpec(
        name="scipy",
        purpose="distance transforms, morphology, and numeric helpers",
        install_hint="pip install scipy",
        optional_for=("distance_transforms", "morphology"),
    ),
    "skimage": DependencySpec(
        name="skimage",
        package_name="scikit-image",
        pypi_name="scikit-image",
        purpose="marching cubes mesh extraction and SSIM metrics",
        install_hint="pip install scikit-image",
        optional_for=("lewiner_marching_cubes", "ssim"),
        aliases=("scikit-image",),
    ),
    "open3d": DependencySpec(
        name="open3d",
        purpose="Poisson and screened-Poisson mesh postprocessing",
        install_hint="pip install open3d",
        optional_for=("poisson_postprocess", "geometry_metrics"),
    ),
    "trimesh": DependencySpec(
        name="trimesh",
        purpose="external mesh evaluation and interchange helpers",
        install_hint="pip install trimesh",
        optional_for=("external_mesh_eval",),
    ),
    "torch": DependencySpec(
        name="torch",
        package_name="PyTorch",
        pypi_name="torch",
        purpose="LPIPS, differentiable research paths, and optional GPU tensor execution",
        install_hint="pip install torch torchvision",
        optional_for=("lpips", "gpu_research"),
    ),
    "torchvision": DependencySpec(
        name="torchvision",
        package_name="torchvision",
        purpose="LPIPS model support",
        install_hint="pip install torchvision",
        optional_for=("lpips",),
    ),
    "lpips": DependencySpec(
        name="lpips",
        purpose="learned perceptual image metric for novel-view evaluation",
        install_hint="pip install lpips",
        optional_for=("novel_view_lpips",),
    ),
    "nvdiffrast": DependencySpec(
        name="nvdiffrast",
        package_name="nvdiffrast",
        pypi_name=None,
        purpose="NVIDIA CUDA/OpenGL differentiable rasterization for mesh silhouette refinement",
        install_hint=(
            "pip install setuptools wheel ninja; "
            "pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
        ),
        docs_url="https://nvlabs.github.io/nvdiffrast/",
        source_url="https://github.com/NVlabs/nvdiffrast",
        optional_for=("gpu_differentiable",),
        gpu_vendor="NVIDIA",
        accelerator="CUDA/OpenGL",
        supports_rocm=False,
        rocm_alternative=(
            "No official ROCm nvdiffrast build exists; use the CPU soft-silhouette "
            "backend for mesh refinement or a separate ROCm GSplat experiment for "
            "Gaussian splatting research."
        ),
        notes=(
            "Not a normal PyPI wheel in this environment.",
            "Requires a compatible NVIDIA driver/CUDA toolchain for the PyTorch path.",
        ),
    ),
    "nvdiffrast.torch": DependencySpec(
        name="nvdiffrast.torch",
        package_name="nvdiffrast",
        pypi_name=None,
        purpose="PyTorch binding for nvdiffrast rasterization",
        install_hint=(
            "pip install setuptools wheel ninja; "
            "pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation"
        ),
        docs_url="https://nvlabs.github.io/nvdiffrast/",
        source_url="https://github.com/NVlabs/nvdiffrast",
        optional_for=("gpu_differentiable",),
        gpu_vendor="NVIDIA",
        accelerator="CUDA/OpenGL",
        supports_rocm=False,
    ),
    "openvdb": DependencySpec(
        name="openvdb",
        import_names=("pyopenvdb", "openvdb"),
        package_name="OpenVDB Python bindings",
        pypi_name=None,
        purpose="direct .vdb sparse volume interchange",
        install_hint=(
            "Install OpenVDB Python bindings for the active Python/Blender Python; "
            "plain 'pip install openvdb' usually has no Windows wheel, so conda/source "
            "builds or Blender-bundled bindings are typically required."
        ),
        source_url="https://github.com/AcademySoftwareFoundation/openvdb",
        optional_for=("sparse_volume_interchange",),
        aliases=("pyopenvdb",),
        notes=(
            "The repo falls back to sparse NPZ interchange when bindings are absent.",
            "Both 'pyopenvdb' and 'openvdb' import names are probed.",
        ),
    ),
    "pyopenvdb": DependencySpec(
        name="pyopenvdb",
        import_names=("pyopenvdb",),
        package_name="OpenVDB Python bindings",
        pypi_name=None,
        purpose="direct .vdb sparse volume interchange",
        install_hint="Install OpenVDB Python bindings that expose the pyopenvdb module.",
        source_url="https://github.com/AcademySoftwareFoundation/openvdb",
        optional_for=("sparse_volume_interchange",),
        aliases=("openvdb",),
    ),
}


@dataclass(frozen=True)
class OptionalDependency:
    """Availability record for a lazily imported dependency."""

    name: str
    available: bool
    module: Optional[ModuleType] = None
    import_name: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    attempts: Tuple[Mapping[str, object], ...] = ()
    spec: Optional[DependencySpec] = None
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def skip_reason(self) -> str:
        if self.available:
            return ""
        message = f"optional dependency {self.name!r} is unavailable"
        if self.error:
            message += f": {self.error}"
        if self.spec and self.spec.install_hint:
            message += f" | install: {self.spec.install_hint}"
        if self.spec and self.spec.supports_rocm is False:
            message += " | ROCm/AMD: unsupported by this dependency"
            if self.spec.rocm_alternative:
                message += f"; {self.spec.rocm_alternative}"
        return message

    def require(self) -> ModuleType:
        """Return the imported module or raise with a clear message."""
        if self.module is None:
            raise RuntimeError(self.skip_reason)
        return self.module

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe availability record."""
        spec_payload: Dict[str, object] = self.spec.to_dict() if self.spec else {}
        return {
            "module_name": self.name,
            "import_name": self.import_name,
            "resolved_module_name": getattr(self.module, "__name__", None)
            if self.module is not None
            else None,
            "available": self.available,
            "module_version": getattr(self.module, "__version__", None)
            if self.module is not None
            else None,
            "module_file": getattr(self.module, "__file__", None)
            if self.module is not None
            else None,
            "error_type": self.error_type,
            "error": self.error,
            "attempts": [dict(attempt) for attempt in self.attempts],
            "package_name": spec_payload.get("package_name", self.name),
            "pypi_name": spec_payload.get("pypi_name"),
            "purpose": spec_payload.get("purpose", ""),
            "install_hint": spec_payload.get("install_hint", ""),
            "docs_url": spec_payload.get("docs_url", ""),
            "source_url": spec_payload.get("source_url", ""),
            "optional_for": spec_payload.get("optional_for", []),
            "aliases": spec_payload.get("aliases", []),
            "gpu_vendor": spec_payload.get("gpu_vendor"),
            "accelerator": spec_payload.get("accelerator"),
            "supports_rocm": spec_payload.get("supports_rocm"),
            "rocm_alternative": spec_payload.get("rocm_alternative"),
            "notes": spec_payload.get("notes", []),
            "details": dict(self.details),
        }


_CACHE: Dict[str, OptionalDependency] = {}


def dependency_spec(name: str) -> Optional[DependencySpec]:
    """Return static dependency metadata for a known import or package alias."""
    if name in KNOWN_DEPENDENCIES:
        return KNOWN_DEPENDENCIES[name]
    normalized = str(name).strip().lower()
    for spec in KNOWN_DEPENDENCIES.values():
        candidates = {value.lower() for value in spec.candidates}
        aliases = {value.lower() for value in spec.aliases}
        package = {str(spec.package_name or "").lower(), str(spec.pypi_name or "").lower()}
        if normalized in candidates or normalized in aliases or normalized in package:
            return spec
    return None


def probe_dependency(import_name: str, *, cache: bool = True) -> OptionalDependency:
    """Probe one import path without making it a hard runtime dependency."""
    if cache and import_name in _CACHE:
        return _CACHE[import_name]
    spec = dependency_spec(import_name)
    candidates: Sequence[str] = spec.candidates if spec else (import_name,)
    attempts: list[Mapping[str, object]] = []
    result: OptionalDependency | None = None
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
            attempts.append(
                {
                    "module_name": candidate,
                    "status": "available",
                    "module_version": getattr(module, "__version__", None),
                    "module_file": getattr(module, "__file__", None),
                }
            )
            result = OptionalDependency(
                import_name,
                True,
                module=module,
                import_name=candidate,
                attempts=tuple(attempts),
                spec=spec,
            )
            break
        except Exception as exc:
            attempts.append(
                {
                    "module_name": candidate,
                    "status": "import_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
    if result is None:
        last = attempts[-1] if attempts else {}
        attempted_names = ", ".join(str(item.get("module_name", "")) for item in attempts)
        error = str(last.get("error", "")) or f"attempted imports: {attempted_names}"
        result = OptionalDependency(
            import_name,
            False,
            import_name=None,
            error=error,
            error_type=str(last.get("error_type", "ImportError")),
            attempts=tuple(attempts),
            spec=spec,
        )
    if cache:
        _CACHE[import_name] = result
    return result


def dependency_report(import_names: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """Return JSON-safe availability records for multiple dependencies."""
    report: Dict[str, Dict[str, object]] = {}
    for name in import_names:
        dep = probe_dependency(name)
        report[name] = dep.to_dict()
    return report


def require_dependency(import_name: str) -> ModuleType:
    """Import an optional dependency or raise a skip-ready RuntimeError."""
    return probe_dependency(import_name).require()


def optional_policy_decision(
    dependency: str | OptionalDependency,
    *,
    policy: str = "skip",
    feature: str = "",
) -> Dict[str, object]:
    """Return a structured skip/fail decision for one optional dependency."""
    selected_policy = str(policy or "skip").strip().lower()
    if selected_policy not in _VALID_OPTIONAL_POLICIES:
        selected_policy = "skip"
    dep = probe_dependency(dependency) if isinstance(dependency, str) else dependency
    if dep.available:
        return {
            "status": "available",
            "result_status": "available",
            "policy": selected_policy,
            "feature": feature,
            "message": f"optional dependency {dep.name!r} is available",
            "dependency": dep.to_dict(),
        }
    result_status = "failed" if selected_policy == "fail" else "skipped"
    feature_prefix = f"{feature}: " if feature else ""
    return {
        "status": result_status,
        "result_status": result_status,
        "policy": selected_policy,
        "feature": feature,
        "message": feature_prefix + dep.skip_reason,
        "dependency": dep.to_dict(),
    }


def clear_dependency_cache() -> None:
    """Reset cached probes for tests."""
    _CACHE.clear()
