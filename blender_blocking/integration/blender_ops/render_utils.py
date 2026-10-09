"""Rendering utilities for Blender."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
from typing import Mapping, Any, Callable, Dict, List, Optional, Tuple

try:
    import bpy
    import mathutils

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

from utils.artifact_publication import publish_file_no_clobber
from utils.run_ownership import OwnedRun

from integration.blender_ops.camera_framing import (
    compute_bounds_world,
    configure_ortho_camera_for_view,
)
from integration.blender_ops.silhouette_render import (
    collect_target_objects,
    render_silhouette_frame,
    set_camera_orbit,
    silhouette_session,
    _suppress_blender_render_stdout,
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
    ownership_receipt: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-safe dictionary."""
        result = {
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
        if self.ownership_receipt is not None:
            result["ownership_receipt"] = self.ownership_receipt
        return result


def _write_render_receipt(owner: OwnedRun, receipt: Mapping[str, Any]) -> None:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    owner.reserve_bytes(len(payload))
    partial = owner.root / "render-result.json.partial"
    partial.write_bytes(payload)
    os.replace(partial, owner.root / "render-result.json")
    owner.register_file("render-result.json", "diagnostic")


def _verified_png(path: Path, resolution: Tuple[int, int]) -> None:
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.size != tuple(resolution):
            raise ValueError("render stage is not a complete PNG at the requested resolution")
        image.verify()


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
        "EEVEE": ["BLENDER_EEVEE"],
        "BLENDER_EEVEE": ["BLENDER_EEVEE"],
    }
    candidates = aliases.get(requested, [requested])
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
        if not os.path.lexists(base_path):
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
            if not os.path.lexists(candidate):
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

    # PNG can retain the scene's existing 16-bit depth: budget all eight RGBA
    # bytes/pixel, filter/compression overhead, plus bounded receipt space.
    width, height = (max(1, int(value)) for value in resolution)
    raw_bytes = width * height * 8
    frame_bytes = raw_bytes + raw_bytes // 100 + height + 65536
    owner = OwnedRun(
        output_path.resolve() / ".render-runs",
        producer="orthographic_render",
        max_generated_bytes=4 * 1024 * 1024 + frame_bytes * len(views),
    )
    receipt = {
        "protocol": "orthographic-render-owned-v1",
        "status": "running",
        "output_dir": str(output_path),
        "publication": "atomic hard link; immutable staging retained",
        "subprocesses_started": 0,
        "frames": [],
        "paths": {},
        "warnings": [],
    }
    error = None
    try:
        _write_render_receipt(owner, receipt)
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
                        calibration=_config_value(render_config, "view_calibration", {}).get(view),
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
                    staged = owner.root / f"frame-{len(receipt['frames']) + 1:04d}.png"
                    frame = {"view": view, "stage": staged.name,
                             "final_candidate": str(output_file), "status": "rendering"}
                    receipt["frames"].append(frame)
                    _write_render_receipt(owner, receipt)
                    owner.reserve_bytes(frame_bytes)
                    try:
                        render_start = time.perf_counter()
                        render_silhouette_frame(session, staged)
                        elapsed_s = time.perf_counter() - render_start
                        _verified_png(staged, resolution)
                        owner.register_file(staged.name, "final_output")
                        collisions = 0
                        for _ in range(128):
                            try:
                                publish_file_no_clobber(staged, output_file)
                                break
                            except FileExistsError:
                                collisions += 1
                                output_file = _unique_path(output_file)
                        else:
                            raise FileExistsError("render publication collision retry bound exceeded")
                        frame.update(
                            status="published", path=str(output_file),
                            bytes=owner.records[staged.name]["bytes"],
                            sha256=owner.records[staged.name]["sha256"],
                            publication_collisions=collisions,
                        )
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
                        _write_render_receipt(owner, receipt)
                    finally:
                        # The legacy synchronous renderer leaves this pointing
                        # at its requested/final output, never private staging.
                        session.scene.render.filepath = str(output_file)
                    if progress_callback is not None:
                        progress_callback(1)
            finally:
                _restore_view_settings(scene, view_snapshot)
                _restore_sample_settings(sample_snapshot)
    except BaseException as exc:
        error = exc
        raise
    finally:
        auxiliary = []
        for frame in receipt["frames"]:
            staged = owner.root / frame["stage"]
            if staged.exists():
                category = "final_output" if frame["status"] == "published" else "diagnostic"
                try:
                    owner.register_file(staged.name, category)
                except Exception as exc:
                    auxiliary.append(exc)
        failure = error or (auxiliary[0] if auxiliary else None)
        receipt.update(
            status=("cancelled" if isinstance(failure, (KeyboardInterrupt, SystemExit))
                    else "failed" if failure is not None else "succeeded"),
            error=None if failure is None else repr(failure),
            paths=dict(output_paths), warnings=list(warnings),
        )
        try:
            _write_render_receipt(owner, receipt)
        except Exception as exc:
            auxiliary.append(exc)
            partial = owner.root / "render-result.json.partial"
            if partial.exists():
                try:
                    owner.register_file(partial.name, "diagnostic")
                except Exception as partial_error:
                    auxiliary.append(partial_error)
        owner.auxiliary_errors.extend(repr(exc) for exc in auxiliary)
        try:
            owner.close(error=error or (auxiliary[0] if auxiliary else None))
        except Exception as exc:
            auxiliary.append(exc)
        if error is not None:
            for secondary in auxiliary:
                if hasattr(error, "add_note"):
                    error.add_note("Render receipt publication also failed: " + repr(secondary))
        elif auxiliary:
            raise auxiliary[0]

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
        ownership_receipt=str(owner.root / "render-result.json"),
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
    calibration: Optional[Mapping[str, Any]] = None,
) -> bool:
    if calibration is not None:
        from reconstruction.projection_contract import validate_view_calibration
        validate_view_calibration({view: calibration})
        xmin, xmax, ymin, ymax = calibration["world_bounds"]
        aspect = resolution[0]/resolution[1]
        if abs((xmax-xmin)/(ymax-ymin)-aspect) > 1e-5:
            raise ValueError("calibrated camera aspect differs from render resolution")
        configure_ortho_camera_for_view(camera, view, bounds_min, bounds_max,
            margin_frac=0., resolution=resolution, distance_factor=camera_distance_factor)
        axes = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}[view]
        camera.location[axes[0]] = (xmin+xmax)*.5
        camera.location[axes[1]] = (ymin+ymax)*.5
        # Blender ortho_scale is horizontal extent with landscape aspect, vertical with portrait.
        camera.data.ortho_scale = max(xmax-xmin, ymax-ymin)
        return True
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
    """Save one current-scene still through an owned stage, without clobbering.

    Scene format, extension, frame, camera and presentation settings are kept.
    Blender supplies the actual still filename; that exact name is published
    once, without a naming retry. An existing final raises FileExistsError.
    The legacy return is None. Success leaves the requested filepath; failure
    restores its previous value and retains the stage and ownership receipt.

    Single-file still outputs are supported. Movie/multiview outputs and path
    templates in the parent directory require a separate publication contract.
    Compositor File Output nodes and external engine outputs remain outside
    this main-image publication contract. The 252 MiB stage bound is accounting,
    not a native render memory/disk quota; oversized output is never published.
    """
    if not BLENDER_AVAILABLE:
        print("Warning: Blender API not available")
        return

    requested = os.fspath(output_path)
    if not isinstance(requested, str) or not requested or requested.endswith(("/", "\\")):
        raise ValueError("save_render requires an explicit still filename")
    leaf = requested.replace("\\", "/").rsplit("/", 1)[-1]
    if leaf in {"", ".", ".."}:
        raise ValueError("save_render requires a non-directory filename leaf")
    scene = bpy.context.scene
    if getattr(scene.render, "is_movie_format", False) or getattr(scene.render, "use_multiview", False):
        raise ValueError("save_render supports a single-file still, not movie/multiview output")
    target = Path(bpy.path.abspath(requested)).absolute()
    if target.name in {"", ".", ".."}:
        raise ValueError("save_render requires a non-directory filename leaf")
    if any(character in str(target.parent) for character in "{}"):
        raise ValueError("save_render does not publish parent-directory path templates")
    parent = target.parent.resolve()
    previous_filepath = scene.render.filepath
    owner = OwnedRun(parent / ".render-runs", producer="save_render",
                     max_generated_bytes=256 * 1024 * 1024)
    stage_dir = owner.root / "stage"
    staged = None
    final = None
    published = False
    primary = None
    auxiliary = []
    started = time.perf_counter()
    receipt = {
        "protocol": "save-render-owned-v1", "status": "running",
        "run_id": owner.run_id, "requested_path": requested,
        "resolved_parent": str(parent), "previous_filepath": previous_filepath,
        "file_format": scene.render.image_settings.file_format,
        "use_file_extension": bool(scene.render.use_file_extension),
        "frame_current": int(scene.frame_current),
        "stage_byte_bound": 252 * 1024 * 1024,
        "subprocesses_started": 0, "native_call": "synchronous current-scene write_still",
        "publication": "atomic no-clobber; exact native filename; no suffix",
        "published": False, "stage": None, "final_path": None,
        "reclamation": "unimplemented; all files retained",
    }

    def stage_entries():
        entries = []
        if stage_dir.is_dir():
            for entry in stage_dir.iterdir():
                if len(entries) >= 64:
                    raise ValueError("single-still stage inventory exceeds 64 entries")
                entries.append(entry)
        return entries

    try:
        stage_dir.mkdir()
        owner.reserve_bytes(receipt["stage_byte_bound"])
        _write_render_receipt(owner, receipt)
        # Keep the original leaf: native write_still performs format-extension
        # and filename-template handling. frame_path is an animation path API.
        scene.render.filepath = str(stage_dir / target.name)
        with _suppress_blender_render_stdout():
            result = bpy.ops.render.render(write_still=True)
        receipt["operator_status"] = sorted(result)
        if "FINISHED" not in result:
            raise RuntimeError("current-scene still render did not finish")
        entries = stage_entries()
        if (len(entries) != 1 or entries[0].is_symlink() or
                getattr(entries[0], "is_junction", lambda: False)() or not entries[0].is_file()):
            raise ValueError("single-still render did not produce exactly one regular staged file")
        staged = entries[0]
        if staged.stat().st_size == 0 or staged.stat().st_size > receipt["stage_byte_bound"]:
            raise ValueError("single-still stage is empty or exceeds its byte bound")
        if receipt["file_format"] == "PNG":
            from PIL import Image
            with Image.open(staged) as image:
                if image.format != "PNG":
                    raise ValueError("current-scene PNG stage has the wrong image format")
                image.verify()
            receipt["stage_validation"] = "closed regular nonempty file; PNG readability"
        else:
            receipt["stage_validation"] = "closed regular nonempty native output; no format-specific pixel proof"
        relative = staged.relative_to(owner.root).as_posix()
        owner.register_file(relative, "diagnostic")
        receipt["stage"] = dict(owner.records[relative.casefold() if os.name == "nt" else relative])
        final = parent / staged.name
        receipt["final_path"] = str(final)
        publish_file_no_clobber(staged, final)
        published = receipt["published"] = True
    except BaseException as exc:
        primary = exc
    finally:
        # The synchronous operator has returned/raised. No child is started or
        # adopted here. Partial files are retained even when rendering failed.
        try:
            for entry in stage_entries():
                if entry.is_file() and not entry.is_symlink():
                    category = "final_output" if published and entry == staged else "diagnostic"
                    owner.register_file(entry.relative_to(owner.root).as_posix(), category)
        except BaseException as exc:
            auxiliary.append(exc)
        if staged is not None:
            key = staged.relative_to(owner.root).as_posix()
            key = key.casefold() if os.name == "nt" else key
            if key in owner.records:
                receipt["stage"] = dict(owner.records[key])
        try:
            scene.render.filepath = requested if primary is None and not auxiliary else previous_filepath
        except BaseException as exc:
            auxiliary.append(exc)
        receipt.update(
            status=("cancelled" if isinstance(primary, (KeyboardInterrupt, SystemExit)) else
                    "failed" if primary is not None or auxiliary else "succeeded"),
            elapsed_s=time.perf_counter() - started,
            error=repr(primary)[:4096] if primary is not None else None,
            auxiliary_errors=[repr(exc)[:4096] for exc in auxiliary],
        )
        try:
            _write_render_receipt(owner, receipt)
        except BaseException as exc:
            auxiliary.append(exc)
        owner.auxiliary_errors.extend(repr(exc)[:4096] for exc in auxiliary)
        if primary is None and auxiliary:
            try:
                scene.render.filepath = previous_filepath
            except BaseException as exc:
                auxiliary.append(exc)
                owner.auxiliary_errors.append(repr(exc)[:4096])
        try:
            owner.close(error=primary if primary is not None else auxiliary[0] if auxiliary else None)
        except BaseException as exc:
            auxiliary.append(exc)
    if primary is not None:
        if hasattr(primary, "add_note"):
            primary.add_note("save_render stage and receipt retained at " + str(owner.root))
            for exc in auxiliary:
                primary.add_note("save_render finalization also failed: " + repr(exc))
        raise primary.with_traceback(primary.__traceback__)
    if auxiliary:
        if hasattr(auxiliary[0], "add_note"):
            auxiliary[0].add_note("save_render stage and receipt retained at " + str(owner.root))
        raise auxiliary[0]
