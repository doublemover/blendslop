"""2D adversarial masks and capture-style degradation generators."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

from .analytic_sdf import OptionalDependencyUnavailable, require_numpy


def pillow_available() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        return False
    return True


def generate_adversarial_mask(
    kind: str,
    resolution: tuple[int, int] = (256, 256),
    seed: int = 0,
) -> tuple[Any, dict[str, object]]:
    np = require_numpy()
    width, height = resolution
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.bool_)
    yy, xx = np.ogrid[:height, :width]

    if kind == "off_center_dark":
        cx, cy, rx, ry = int(width * 0.68), int(height * 0.38), int(width * 0.18), int(height * 0.26)
        mask[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = True
    elif kind == "border_touching":
        mask[int(height * 0.2) : int(height * 0.82), 0 : int(width * 0.42)] = True
    elif kind == "single_outlier_pixel":
        mask[int(height * 0.35) : int(height * 0.65), int(width * 0.35) : int(width * 0.65)] = True
        mask[max(0, height - 3), min(width - 1, 2)] = True
    elif kind == "dust_clusters":
        mask[int(height * 0.3) : int(height * 0.7), int(width * 0.34) : int(width * 0.66)] = True
        for _ in range(16):
            cx = int(rng.integers(0, width))
            cy = int(rng.integers(0, height))
            r = int(rng.integers(1, 4))
            mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True
    elif kind == "thin_diagonal_struts":
        for offset in (-36, 0, 36):
            distance = abs((yy - height // 2) - (xx - width // 2) * 0.65 - offset)
            mask[distance <= 1.25] = True
        mask[int(height * 0.25) : int(height * 0.75), int(width * 0.46) : int(width * 0.54)] = True
    elif kind == "holed_silhouette":
        outer = ((xx - width // 2) / (width * 0.28)) ** 2 + ((yy - height // 2) / (height * 0.34)) ** 2 <= 1.0
        inner = ((xx - width // 2) / (width * 0.11)) ** 2 + ((yy - height // 2) / (height * 0.15)) ** 2 <= 1.0
        mask[outer & ~inner] = True
    elif kind == "disconnected_components":
        mask[((xx - int(width * 0.35)) / (width * 0.13)) ** 2 + ((yy - height // 2) / (height * 0.22)) ** 2 <= 1.0] = True
        mask[((xx - int(width * 0.68)) / (width * 0.1)) ** 2 + ((yy - int(height * 0.57)) / (height * 0.16)) ** 2 <= 1.0] = True
    elif kind == "zero_radius_rings":
        for r in (0, 0, 3, 28, 29, 0, 47):
            if r == 0:
                mask[height // 2, width // 2] = True
            else:
                distance = np.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)
                mask[np.abs(distance - r) <= 1.0] = True
    elif kind == "tiny_object_large_canvas":
        mask[((xx - width // 2) ** 2 + (yy - height // 2) ** 2) <= 5 * 5] = True
    elif kind == "full_canvas_near_threshold":
        mask[:, :] = True
    elif kind == "ambiguous_polarity_pair":
        mask[((xx - width // 2) / (width * 0.25)) ** 2 + ((yy - height // 2) / (height * 0.25)) ** 2 <= 1.0] = True
    elif kind == "inconsistent_front_side":
        mask[int(height * 0.23) : int(height * 0.78), int(width * 0.3) : int(width * 0.7)] = True
        mask[int(height * 0.42) : int(height * 0.58), int(width * 0.48) : int(width * 0.93)] = True
    else:
        raise ValueError(f"Unknown adversarial mask kind: {kind}")

    metadata = {
        "mask_kind": kind,
        "resolution": [width, height],
        "seed": seed,
        "foreground_true": True,
        "expected_effect": _mask_effect(kind),
    }
    return mask, metadata


def mask_to_uint8(mask: Any, foreground: int = 0, background: int = 255) -> Any:
    np = require_numpy()
    return np.where(mask, foreground, background).astype(np.uint8)


def apply_degradation(image: Any, kind: str, seed: int = 0) -> tuple[Any, dict[str, object]]:
    np = require_numpy()
    rng = np.random.default_rng(seed)
    arr = np.asarray(image).astype(np.float32)
    params: dict[str, object] = {"degradation": kind, "seed": seed}

    if kind == "jpeg_artifacts":
        quality = 35
        params["quality"] = quality
        arr = _jpeg_roundtrip(arr.astype(np.uint8), quality).astype(np.float32)
    elif kind == "blur":
        radius = 1.6
        params["radius"] = radius
        arr = _box_blur(arr, radius=2)
    elif kind == "paper_sketch_lines":
        line_count = 22
        params["line_count"] = line_count
        for _ in range(line_count):
            x0 = int(rng.integers(0, arr.shape[1]))
            y0 = int(rng.integers(0, arr.shape[0]))
            length = int(rng.integers(12, 48))
            angle = float(rng.uniform(0, math.pi))
            _draw_line(arr, x0, y0, int(x0 + math.cos(angle) * length), int(y0 + math.sin(angle) * length), 90)
    elif kind == "partial_occlusion":
        x0 = int(arr.shape[1] * 0.43)
        y0 = int(arr.shape[0] * 0.22)
        arr[y0 : y0 + int(arr.shape[0] * 0.22), x0 : x0 + int(arr.shape[1] * 0.38)] = 255
        params["occluder"] = [x0, y0, int(arr.shape[1] * 0.38), int(arr.shape[0] * 0.22)]
    elif kind == "uneven_lighting":
        gradient = np.linspace(0.75, 1.22, arr.shape[1], dtype=np.float32)
        arr = np.clip(arr * gradient[None, ...], 0, 255)
        params["gradient"] = "horizontal_0.75_1.22"
    elif kind == "low_contrast":
        arr = 118.0 + (arr - 128.0) * 0.18
        params["contrast_scale"] = 0.18
    elif kind == "alpha_premultiplication":
        alpha = np.linspace(0.45, 1.0, arr.shape[0], dtype=np.float32)[:, None]
        arr = np.clip(arr * alpha + 255.0 * (1.0 - alpha), 0, 255)
        params["alpha_range"] = [0.45, 1.0]
    elif kind == "transparent_rgb_noise":
        noise = rng.normal(0.0, 9.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        params["rgb_noise_sigma"] = 9.0
    elif kind == "missing_top_view":
        params["expected_effect"] = "view omitted by artifact writer"
    elif kind == "tilted_input":
        params["tilt_degrees"] = 7.5
        arr = _shift_rows(arr, max_shift=10)
    else:
        raise ValueError(f"Unknown degradation kind: {kind}")

    params.setdefault("expected_effect", _degradation_effect(kind))
    return arr.astype(np.uint8), params


def save_png_or_pgm(path: Path, image: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        fallback_path = path.with_suffix(".pgm")
        _write_pgm(fallback_path, image)
        return "Pillow unavailable; wrote portable graymap fallback"
    Image.fromarray(image).save(path)
    return "png"


def _jpeg_roundtrip(arr: Any, quality: int) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyUnavailable("Pillow is required for JPEG artifact degradation.") from exc
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return require_numpy().asarray(Image.open(buffer).convert("L"))


def _box_blur(arr: Any, radius: int) -> Any:
    np = require_numpy()
    padded = np.pad(arr, radius, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return out / float(size * size)


def _draw_line(arr: Any, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    np = require_numpy()
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps).astype(int)
    ys = np.linspace(y0, y1, steps).astype(int)
    valid = (xs >= 0) & (ys >= 0) & (xs < arr.shape[1]) & (ys < arr.shape[0])
    arr[ys[valid], xs[valid]] = value


def _shift_rows(arr: Any, max_shift: int) -> Any:
    np = require_numpy()
    out = np.empty_like(arr)
    for y in range(arr.shape[0]):
        shift = int((y / max(1, arr.shape[0] - 1) - 0.5) * 2 * max_shift)
        out[y] = np.roll(arr[y], shift)
    return out


def _write_pgm(path: Path, image: Any) -> None:
    np = require_numpy()
    arr = np.asarray(image, dtype=np.uint8)
    with path.open("wb") as handle:
        handle.write(f"P5\n{arr.shape[1]} {arr.shape[0]}\n255\n".encode("ascii"))
        handle.write(arr.tobytes())


def _mask_effect(kind: str) -> str:
    return {
        "off_center_dark": "checks polarity and off-center bbox handling",
        "border_touching": "checks border clipping assumptions",
        "single_outlier_pixel": "checks outlier rejection before bbox normalization",
        "dust_clusters": "checks small-component filtering",
        "thin_diagonal_struts": "checks anti-aliased thin support preservation",
        "holed_silhouette": "checks hole preservation",
        "disconnected_components": "checks multi-component foreground handling",
        "zero_radius_rings": "checks degenerate profile-band handling",
        "tiny_object_large_canvas": "checks tiny-object scale normalization",
        "full_canvas_near_threshold": "checks full-canvas foreground rejection",
        "ambiguous_polarity_pair": "checks dark/light polarity ambiguity",
        "inconsistent_front_side": "checks deliberate view disagreement",
    }.get(kind, "adversarial mask")


def _degradation_effect(kind: str) -> str:
    return {
        "jpeg_artifacts": "introduces block artifacts around edges",
        "blur": "softens foreground boundary",
        "paper_sketch_lines": "adds non-object sketch strokes",
        "partial_occlusion": "hides a known foreground section",
        "uneven_lighting": "moves threshold by image position",
        "low_contrast": "compresses foreground/background separation",
        "alpha_premultiplication": "fades object through alpha blending",
        "transparent_rgb_noise": "adds faint RGB noise in transparent-like regions",
        "missing_top_view": "marks top view as unavailable",
        "tilted_input": "marks non-axis-aligned capture input",
    }.get(kind, "capture degradation")
