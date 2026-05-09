"""Novel-view and image-based reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    from blender_blocking.utils.optional_deps import probe_dependency
except Exception:  # pragma: no cover - script-style imports
    from utils.optional_deps import probe_dependency


@dataclass(frozen=True)
class NovelViewMetricReport:
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    mse: float | None = None
    image_count: int = 0
    warnings: tuple[str, ...] = ()
    dependency_state: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "mse": self.mse,
            "image_count": self.image_count,
            "warnings": list(self.warnings),
            "dependency_state": dict(self.dependency_state),
        }


def image_pair_report(
    reference: Any,
    candidate: Any,
    *,
    compute_ssim: bool = True,
    compute_lpips: bool = False,
) -> NovelViewMetricReport:
    np = _np()
    ref = _image(reference)
    cand = _image(candidate)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate images must have matching shapes")
    mse_value = float(np.mean((ref - cand) ** 2))
    psnr_value = psnr_from_mse(mse_value)
    warnings: list[str] = []
    dependencies: dict[str, Any] = {}

    ssim_value = None
    if compute_ssim:
        skimage_dependency = probe_dependency("skimage")
        dependencies["skimage"] = skimage_dependency.to_dict()
        if not skimage_dependency.available:
            warnings.append(f"SSIM unavailable: {skimage_dependency.skip_reason}")
        else:
            try:
                from skimage.metrics import structural_similarity

                channel_axis = -1 if ref.ndim == 3 and ref.shape[-1] > 1 else None
                ssim_value = float(
                    structural_similarity(
                        ref,
                        cand,
                        data_range=1.0,
                        channel_axis=channel_axis,
                    )
                )
            except Exception as exc:
                dependencies["skimage"] = {
                    **dependencies["skimage"],
                    "available": False,
                    "status": "unusable",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                warnings.append(f"SSIM unavailable: {exc}")

    lpips_value = None
    if compute_lpips:
        lpips_value, lpips_state, lpips_warning = _lpips_score(ref, cand)
        dependencies["lpips"] = lpips_state
        if lpips_warning:
            warnings.append(lpips_warning)

    return NovelViewMetricReport(
        psnr=psnr_value,
        ssim=ssim_value,
        lpips=lpips_value,
        mse=mse_value,
        image_count=1,
        warnings=tuple(warnings),
        dependency_state=dependencies,
    )


def image_set_report(
    pairs: Mapping[str, tuple[Any, Any]],
    *,
    compute_ssim: bool = True,
    compute_lpips: bool = False,
) -> NovelViewMetricReport:
    reports = [
        image_pair_report(
            ref,
            cand,
            compute_ssim=compute_ssim,
            compute_lpips=compute_lpips,
        )
        for ref, cand in pairs.values()
    ]
    if not reports:
        return NovelViewMetricReport(warnings=("no image pairs supplied",))
    return NovelViewMetricReport(
        psnr=_mean_optional(report.psnr for report in reports),
        ssim=_mean_optional(report.ssim for report in reports),
        lpips=_mean_optional(report.lpips for report in reports),
        mse=_mean_optional(report.mse for report in reports),
        image_count=len(reports),
        warnings=tuple(warning for report in reports for warning in report.warnings),
        dependency_state={
            str(index): dict(report.dependency_state)
            for index, report in enumerate(reports)
            if report.dependency_state
        },
    )


def psnr_from_mse(mse: float, *, max_value: float = 1.0) -> float:
    np = _np()
    mse = max(0.0, float(mse))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(float(max_value)) - 10.0 * np.log10(mse))


def report_from_mapping(payload: Mapping[str, Any]) -> NovelViewMetricReport:
    return NovelViewMetricReport(
        psnr=_optional_float(payload.get("psnr")),
        ssim=_optional_float(payload.get("ssim")),
        lpips=_optional_float(payload.get("lpips")),
        mse=_optional_float(payload.get("mse")),
        image_count=int(payload.get("image_count", 0) or 0),
        warnings=tuple(str(item) for item in payload.get("warnings", ()) or ()),
        dependency_state=payload.get("dependency_state", {})
        if isinstance(payload.get("dependency_state", {}), Mapping)
        else {},
    )


def _lpips_score(reference: Any, candidate: Any) -> tuple[float | None, dict[str, Any], str]:
    torch_dependency = probe_dependency("torch")
    lpips_dependency = probe_dependency("lpips")
    state = {
        "torch": torch_dependency.to_dict(),
        "lpips": lpips_dependency.to_dict(),
    }
    missing = [
        dependency.skip_reason
        for dependency in (torch_dependency, lpips_dependency)
        if not dependency.available
    ]
    if missing:
        return None, state, "LPIPS unavailable: " + "; ".join(missing)
    try:
        torch = torch_dependency.require()
        lpips = lpips_dependency.require()

        loss = lpips.LPIPS(net="alex")
        ref_tensor = _lpips_tensor(reference, torch)
        cand_tensor = _lpips_tensor(candidate, torch)
        with torch.no_grad():
            value = float(loss(ref_tensor, cand_tensor).item())
        return value, state, ""
    except Exception as exc:
        state["runtime"] = {
            "available": False,
            "status": "unusable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return None, state, f"LPIPS unavailable: {exc}"


def _lpips_tensor(image: Any, torch: Any) -> Any:
    np = _np()
    arr = _image(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    arr = arr.transpose(2, 0, 1)[None, :, :, :]
    return torch.from_numpy(arr.astype("float32") * 2.0 - 1.0)


def _image(value: Any) -> Any:
    np = _np()
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("image arrays cannot be empty")
    arr = arr.astype("float32", copy=False)
    if float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _mean_optional(values: Any) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _np() -> Any:
    import numpy as np

    return np
