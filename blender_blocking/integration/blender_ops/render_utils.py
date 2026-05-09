"""Rendering utilities for Blender."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

from integration.blender_ops.camera_framing import (
    compute_bounds_world,
    configure_ortho_camera_for_view,
)
from integration.blender_ops.silhouette_render import (
    collect_target_objects,
    render_silhouette_frame,
    set_camera_orbit,
    silhouette_session,
)


@dataclass(frozen=True)
class RenderViewRecord:
    """Metadata for a single rendered orthographic view."""

    view: str
    path: str
    elapsed_s: float
    camera_location: Tuple[float, float, float]
    camera_rotation: Tuple[float, float, float]
    ortho_scale: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "view": self.view,
            "path": self.path,
            "elapsed_s": self.elapsed_s,
            "camera_location": list(self.camera_location),
            "camera_rotation": list(self.camera_rotation),
            "ortho_scale": self.ortho_scale,
        }


@dataclass(frozen=True)
class RenderResult:
    """Structured render result with paths and render-setting metadata."""

    paths: Dict[str, str]
    requested_engine: Optional[str]
    applied_engine: Optional[str]
    engine_fallback: Optional[str]
    requested_samples: Optional[int]
    applied_samples: Optional[int]
    resolution: Tuple[int, int]
    color_mode: str
    transparent_bg: bool
    views: Tuple[RenderViewRecord, ...]
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        return {
            "paths": dict(self.paths),
            "requested_engine": self.requested_engine,
            "applied_engine": self.applied_engine,
            "engine_fallback": self.engine_fallback,
            "requested_samples": self.requested_samples,
            "applied_samples": self.applied_samples,
            "resolution": list(self.resolution),
            "color_mode": self.color_mode,
            "transparent_bg": self.transparent_bg,
            "views": [view.to_dict() for view in self.views],
            "warnings": list(self.warnings),
        }


def _config_value(config: Optional[Any], name: str, default: Any) -> Any:
    if config is None:
        return default
    return getattr(config, name, default)


def _available_render_engines(scene: Any) -> List[str]:
    try:
        prop = scene.render.bl_rna.properties["engine"]
        return [item.identifier for item in prop.enum_items]
    except Exception:
        current = getattr(scene.render, "engine", None)
        return [current] if current else []


def _engine_candidates(requested: Optional[str]) -> List[str]:
    if not requested:
        return []
    requested = str(requested)
    aliases = {
        "WORKBENCH": ["BLENDER_WORKBENCH", "WORKBENCH"],
        "BLENDER_WORKBENCH": ["BLENDER_WORKBENCH", "WORKBENCH"],
        "EEVEE": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
        "BLENDER_EEVEE": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
        "BLENDER_EEVEE_NEXT": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
    }
    candidates = aliases.get(requested, [requested])
    if requested.startswith("BLENDER_EEVEE") or requested == "EEVEE":
        candidates = candidates + ["BLENDER_WORKBENCH", "WORKBENCH"]
    else:
        candidates = candidates + ["BLENDER_WORKBENCH", "WORKBENCH"]
    deduped: List[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resolve_render_engine(
    scene: Any, requested: Optional[str]
) -> Tuple[Optional[str], Optional[str], Tuple[str, ...]]:
    if requested is None:
        return getattr(scene.render, "engine", None), None, ()

    available = _available_render_engines(scene)
    for candidate in _engine_candidates(requested):
        if not available or candidate in available:
            fallback = candidate if candidate != requested else None
            warnings = ()
            if fallback is not None:
                warnings = (
                    f"render_engine_fallback:{requested}->{candidate}",
                )
            return candidate, fallback, warnings

    current = getattr(scene.render, "engine", None)
    warning = f"render_engine_unavailable:{requested}"
    return current, current if current != requested else None, (warning,)


def _sample_setting_paths(scene: Any) -> List[Tuple[Any, str]]:
    paths: List[Tuple[Any, str]] = []
    for owner_name, attr_name in (
        ("eevee", "taa_render_samples"),
        ("eevee", "taa_samples"),
        ("cycles", "samples"),
        ("display", "render_aa"),
    ):
        owner = getattr(scene, owner_name, None)
        if owner is not None and hasattr(owner, attr_name):
            paths.append((owner, attr_name))
    return paths


def _snapshot_sample_settings(scene: Any) -> Dict[Tuple[int, str], Tuple[Any, str, Any]]:
    snapshot: Dict[Tuple[int, str], Tuple[Any, str, Any]] = {}
    for owner, attr_name in _sample_setting_paths(scene):
        snapshot[(id(owner), attr_name)] = (owner, attr_name, getattr(owner, attr_name))
    return snapshot


def _restore_sample_settings(
    snapshot: Dict[Tuple[int, str], Tuple[Any, str, Any]]
) -> None:
    for owner, attr_name, value in snapshot.values():
        try:
            setattr(owner, attr_name, value)
        except Exception:
            pass


def _apply_render_samples(
    scene: Any, samples: Optional[int]
) -> Tuple[Optional[int], Tuple[str, ...]]:
    if samples is None:
        return None, ()

    warnings: List[str] = []
    applied = False
    for owner, attr_name in _sample_setting_paths(scene):
        current = getattr(owner, attr_name)
        try:
            if isinstance(current, str):
                setattr(owner, attr_name, str(samples))
            else:
                setattr(owner, attr_name, int(samples))
            applied = True
        except Exception as exc:
            warnings.append(f"render_samples_not_applied:{attr_name}:{exc}")

    if not applied:
        warnings.append("render_samples_no_supported_setting")
        return None, tuple(warnings)
    return int(samples), tuple(warnings)


def _snapshot_view_settings(scene: Any) -> Dict[str, Any]:
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is None:
        return {}
    snapshot = {}
    for attr_name in ("view_transform", "look", "exposure", "gamma"):
        if hasattr(view_settings, attr_name):
            snapshot[attr_name] = getattr(view_settings, attr_name)
    return snapshot


def _apply_deterministic_view_settings(scene: Any) -> None:
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is None:
        return
    for attr_name, value in (
        ("view_transform", "Standard"),
        ("look", "None"),
        ("exposure", 0.0),
        ("gamma", 1.0),
    ):
        if hasattr(view_settings, attr_name):
            try:
                setattr(view_settings, attr_name, value)
            except Exception:
                pass


def _restore_view_settings(scene: Any, snapshot: Dict[str, Any]) -> None:
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is None:
        return
    for attr_name, value in snapshot.items():
        try:
            setattr(view_settings, attr_name, value)
        except Exception:
            pass


def _camera_tuple(value: Any) -> Tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def render_orthogonal_views_detailed(
    output_dir: str,
    views: List[str] = ["front", "side", "top"],
    *,
    render_config: Optional[Any] = None,
    target_objects: Optional[List[object]] = None,
    fit_to_bounds: bool = True,
    margin_frac: float = 0.08,
    resolution: Tuple[int, int] = (512, 512),
    color_mode: str = "RGBA",
    transparent_bg: bool = True,
    force_material: bool = False,
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    camera_distance_factor: float = 2.0,
    party_mode: bool = False,
    filename_prefix: Optional[str] = None,
    start_index: int = 1,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> RenderResult:
    """
    Render orthogonal views of the scene.

    Args:
        output_dir: Directory to save renders
        views: List of views to render

    Returns:
        Structured render result with paths and metadata
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return RenderResult(
            paths={},
            requested_engine=_config_value(render_config, "engine", None),
            applied_engine=None,
            engine_fallback=None,
            requested_samples=_config_value(render_config, "samples", None),
            applied_samples=None,
            resolution=_config_value(render_config, "resolution", resolution),
            color_mode=_config_value(render_config, "color_mode", color_mode),
            transparent_bg=_config_value(
                render_config, "transparent_bg", transparent_bg
            ),
            views=(),
            warnings=("blender_unavailable",),
        )

    margin_frac = _config_value(render_config, "margin_frac", margin_frac)
    resolution = tuple(_config_value(render_config, "resolution", resolution))
    color_mode = _config_value(render_config, "color_mode", color_mode)
    transparent_bg = _config_value(
        render_config, "transparent_bg", transparent_bg
    )
    force_material = _config_value(render_config, "force_material", force_material)
    background_color = tuple(
        _config_value(render_config, "background_color", background_color)
    )
    silhouette_color = tuple(
        _config_value(render_config, "silhouette_color", silhouette_color)
    )
    camera_distance_factor = _config_value(
        render_config, "camera_distance_factor", camera_distance_factor
    )
    party_mode = _config_value(render_config, "party_mode", party_mode)
    requested_engine = _config_value(render_config, "engine", None)
    requested_samples = _config_value(render_config, "samples", None)

    output_paths = {}
    view_records: List[RenderViewRecord] = []
    warnings: List[str] = []
    applied_samples: Optional[int] = None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    applied_engine, engine_fallback, engine_warnings = _resolve_render_engine(
        scene, requested_engine
    )
    warnings.extend(engine_warnings)
    target_objects = collect_target_objects(scene, target_objects)

    if not target_objects:
        print("Warning: No renderable mesh objects found.")
        warnings.append("no_renderable_mesh_targets")
        return RenderResult(
            paths={},
            requested_engine=requested_engine,
            applied_engine=applied_engine,
            engine_fallback=engine_fallback,
            requested_samples=requested_samples,
            applied_samples=None,
            resolution=resolution,
            color_mode=color_mode,
            transparent_bg=transparent_bg,
            views=(),
            warnings=tuple(warnings),
        )

    if fit_to_bounds and target_objects:
        bounds_min, bounds_max = compute_bounds_world(target_objects)
    else:
        bounds_min = mathutils.Vector((0.0, 0.0, 0.0))
        bounds_max = mathutils.Vector((1.0, 1.0, 1.0))

    def _unique_path(base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        match = re.match(r"^(.*?)(?:_(\d+))?$", stem)
        if match:
            base_stem = match.group(1)
            start_at = int(match.group(2)) if match.group(2) else 1
        else:
            base_stem = stem
            start_at = 1
        for idx in range(start_at + 1, 10000):
            candidate = base_path.with_name(f"{base_stem}_{idx}{suffix}")
            if not candidate.exists():
                return candidate
        return base_path.with_name(f"{base_stem}_{uuid.uuid4().hex[:8]}{suffix}")

    def _bounded_output_path(base_path: Path, *, max_chars: int = 240) -> Path:
        path_text = str(base_path)
        if len(path_text) <= max_chars:
            return base_path
        parent_text = str(base_path.parent)
        suffix = base_path.suffix
        name_budget = max_chars - len(parent_text) - 1 - len(suffix)
        if name_budget < 16:
            return base_path
        stem = base_path.stem
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
        head_len = max(1, name_budget - len(digest) - 1)
        return base_path.with_name(f"{stem[:head_len]}-{digest}{suffix}")

    try:
        with silhouette_session(
            scene=scene,
            target_objects=target_objects,
            resolution=resolution,
            color_mode=color_mode,
            transparent_bg=transparent_bg,
            engine=applied_engine,
            background_color=background_color,
            silhouette_color=silhouette_color,
            force_material=force_material,
            hide_non_targets=True,
            party_mode=party_mode,
        ) as session:
            sample_snapshot = _snapshot_sample_settings(scene)
            view_snapshot = _snapshot_view_settings(scene)
            try:
                applied_samples, sample_warnings = _apply_render_samples(
                    scene, requested_samples
                )
                warnings.extend(sample_warnings)
                _apply_deterministic_view_settings(scene)
                for view in views:
                    if not _configure_validation_camera(
                        session.camera,
                        view,
                        bounds_min,
                        bounds_max,
                        margin_frac=margin_frac,
                        resolution=resolution,
                        camera_distance_factor=camera_distance_factor,
                    ):
                        warnings.append(f"unsupported_view_skipped:{view}")
                        continue

                    if filename_prefix:
                        stem = f"{filename_prefix}{view}_{start_index}"
                    else:
                        stem = view
                    output_file = _unique_path(
                        _bounded_output_path(output_path / f"{stem}.png")
                    )
                    render_start = time.perf_counter()
                    render_silhouette_frame(session, output_file)
                    elapsed_s = time.perf_counter() - render_start

                    output_paths[view] = str(output_file)
                    view_records.append(
                        RenderViewRecord(
                            view=view,
                            path=str(output_file),
                            elapsed_s=elapsed_s,
                            camera_location=_camera_tuple(session.camera.location),
                            camera_rotation=_camera_tuple(session.camera.rotation_euler),
                            ortho_scale=float(session.camera.data.ortho_scale),
                        )
                    )
                    if progress_callback is not None:
                        progress_callback(1)
            finally:
                _restore_view_settings(scene, view_snapshot)
                _restore_sample_settings(sample_snapshot)
    finally:
        pass

    return RenderResult(
        paths=output_paths,
        requested_engine=requested_engine,
        applied_engine=applied_engine,
        engine_fallback=engine_fallback,
        requested_samples=requested_samples,
        applied_samples=applied_samples,
        resolution=resolution,
        color_mode=color_mode,
        transparent_bg=transparent_bg,
        views=tuple(view_records),
        warnings=tuple(warnings),
    )


def _configure_validation_camera(
    camera: object,
    view: str,
    bounds_min: object,
    bounds_max: object,
    *,
    margin_frac: float,
    resolution: Tuple[int, int],
    camera_distance_factor: float,
) -> bool:
    if view in {"front", "side", "top"}:
        configure_ortho_camera_for_view(
            camera,
            view,
            bounds_min,
            bounds_max,
            margin_frac=margin_frac,
            resolution=resolution,
            distance_factor=camera_distance_factor,
        )
        return True

    angle = parse_orbit_view_degrees(view)
    if angle is None:
        return False

    center = (bounds_min + bounds_max) / 2.0
    width = float(bounds_max.x - bounds_min.x)
    depth = float(bounds_max.y - bounds_min.y)
    height = float(bounds_max.z - bounds_min.z)
    max_dim = max(width, depth, height, 1e-3)
    distance = max_dim * float(camera_distance_factor)
    ortho_scale = max_dim * (1.0 + 2.0 * float(margin_frac))
    import math

    set_camera_orbit(
        camera,
        center,
        distance,
        math.radians(float(angle)),
        ortho_scale,
    )
    return True


def parse_orbit_view_degrees(view: str) -> Optional[float]:
    """Parse a named orbit validation view such as orbit_045 or azimuth-135."""
    value = str(view).strip().lower()
    match = re.match(
        r"^(?:orbit|azimuth|view|angle)[_-]?(-?\d+(?:\.\d+)?)$",
        value,
    )
    if not match:
        return None
    return float(match.group(1)) % 360.0


def render_orthogonal_views(
    output_dir: str,
    views: List[str] = ["front", "side", "top"],
    *,
    render_config: Optional[Any] = None,
    target_objects: Optional[List[object]] = None,
    fit_to_bounds: bool = True,
    margin_frac: float = 0.08,
    resolution: Tuple[int, int] = (512, 512),
    color_mode: str = "RGBA",
    transparent_bg: bool = True,
    force_material: bool = False,
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    silhouette_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    camera_distance_factor: float = 2.0,
    party_mode: bool = False,
    filename_prefix: Optional[str] = None,
    start_index: int = 1,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, str]:
    """Render orthogonal views and return the legacy view-to-path mapping."""
    result = render_orthogonal_views_detailed(
        output_dir,
        views=views,
        render_config=render_config,
        target_objects=target_objects,
        fit_to_bounds=fit_to_bounds,
        margin_frac=margin_frac,
        resolution=resolution,
        color_mode=color_mode,
        transparent_bg=transparent_bg,
        force_material=force_material,
        background_color=background_color,
        silhouette_color=silhouette_color,
        camera_distance_factor=camera_distance_factor,
        party_mode=party_mode,
        filename_prefix=filename_prefix,
        start_index=start_index,
        progress_callback=progress_callback,
    )
    return result.paths


def save_render(output_path: str) -> None:
    """
    Render and save the current scene.

    Args:
        output_path: Path to save the render
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
