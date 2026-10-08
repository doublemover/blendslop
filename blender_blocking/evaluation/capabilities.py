"""Capability and optional dependency reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    from blender_blocking.utils.optional_deps import probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import probe_dependency


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    kind: str
    required_for: tuple[str, ...] = ()
    optional_for: tuple[str, ...] = ()
    min_version: str | None = None
    fail_policy: str = "fail"
    degrade_to: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required_for": list(self.required_for),
            "optional_for": list(self.optional_for),
            "min_version": self.min_version,
            "fail_policy": self.fail_policy,
            "degrade_to": self.degrade_to,
        }


@dataclass(frozen=True)
class CapabilityReport:
    capability: str
    available: bool
    version: str | None = None
    path: str | None = None
    status: str = "available"
    policy: str = "optional"
    effect: str = "none"
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "available": self.available,
            "version": self.version,
            "path": self.path,
            "status": self.status,
            "policy": self.policy,
            "effect": self.effect,
            "message": self.message,
        }


DEFAULT_CAPABILITIES = (
    CapabilityRequirement("numpy", "python_package", required_for=("core",)),
    CapabilityRequirement("PIL", "python_package", required_for=("image_inputs",)),
    CapabilityRequirement("scipy", "python_package", optional_for=("distance_transforms", "morphology"), fail_policy="skip"),
    CapabilityRequirement("skimage", "python_package", optional_for=("ssim", "lewiner_marching_cubes"), fail_policy="skip"),
    CapabilityRequirement("trimesh", "python_package", optional_for=("external_mesh_eval",), fail_policy="skip"),
    CapabilityRequirement("open3d", "python_package", optional_for=("poisson_postprocess",), fail_policy="skip"),
    CapabilityRequirement("torch", "python_package", optional_for=("lpips", "gpu_research"), fail_policy="skip"),
    CapabilityRequirement("lpips", "python_package", optional_for=("novel_view_lpips",), fail_policy="skip"),
    CapabilityRequirement("nvdiffrast", "python_package", optional_for=("gpu_differentiable",), fail_policy="skip"),
    CapabilityRequirement("openvdb", "python_package", optional_for=("sparse_volume_interchange",), fail_policy="skip"),
)


def probe_capability(requirement: CapabilityRequirement, *, policy: str | None = None) -> CapabilityReport:
    dep = probe_dependency(requirement.name)
    selected_policy = policy or ("required" if requirement.required_for else "optional")
    if dep.available:
        return CapabilityReport(
            capability=requirement.name,
            available=True,
            version=str(getattr(dep.module, "__version__", "")) or None,
            path=str(getattr(dep.module, "__file__", "")) or None,
            status="available",
            policy=selected_policy,
            effect="none",
        )
    effect = "fail" if selected_policy == "required" else requirement.fail_policy
    return CapabilityReport(
        capability=requirement.name,
        available=False,
        status="missing",
        policy=selected_policy,
        effect=effect,
        message=dep.skip_reason,
    )


def capability_report(
    requirements: Iterable[CapabilityRequirement] = DEFAULT_CAPABILITIES,
) -> dict[str, dict[str, object]]:
    return {item.name: probe_capability(item).to_dict() for item in requirements}


def backend_dependency_state(optional_dependencies: Iterable[str]) -> dict[str, Any]:
    requirements = [
        CapabilityRequirement(str(name), "python_package", optional_for=("backend",), fail_policy="skip")
        for name in optional_dependencies
    ]
    return capability_report(requirements)


def unavailable_effects(report: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    effects: dict[str, str] = {}
    for name, payload in report.items():
        if not bool(payload.get("available")):
            effects[str(name)] = str(payload.get("effect", "skip"))
    return effects
