"""Retain canonical inspection evidence without conflating it with acceptance.

An inventory observes existing files only. It never renders missing passes or
infers family surface limits from another family. All five cameras are retained
independently of any smaller presentation gallery.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

CANONICAL_VIEWS = ("front", "side", "top", "oblique_35_28", "oblique_145_40")
INSPECTION_PASSES = ("mask", "neutral", "normals")
REQUIRED_PASSES = ("mask", "neutral")
PASS_STATES = {"completed", "unrun", "unavailable"}


def camera_record(camera, *, resolution=(512, 512), pixel_aspect=(1., 1.)):
    """Capture the actual native orthographic camera after framing/replay."""
    if camera.data.type != "ORTHO":
        raise ValueError("canonical inspection requires an orthographic camera")
    return {"projection": "ORTHO", "matrix_world": [list(row) for row in camera.matrix_world],
            "ortho_scale": float(camera.data.ortho_scale),
            "shift_x": float(camera.data.shift_x), "shift_y": float(camera.data.shift_y),
            "clip_start": float(camera.data.clip_start), "clip_end": float(camera.data.clip_end),
            "resolution": list(resolution), "pixel_aspect": list(pixel_aspect)}


def _camera_identity(record, resolution):
    if not isinstance(record, dict):
        raise ValueError("camera record is unavailable")
    matrix = record.get("matrix_world")
    if (not isinstance(matrix, (list, tuple)) or len(matrix) != 4 or
            any(not isinstance(row, (list, tuple)) or len(row) != 4 for row in matrix)):
        raise ValueError("camera matrix must have shape (4, 4)")
    matrix = [[float(value) for value in row] for row in matrix]
    scale = float(record["ortho_scale"])
    shifts = [float(record.get("shift_x", 0.)), float(record.get("shift_y", 0.))]
    if (record.get("projection", "ORTHO") != "ORTHO" or scale <= 0 or
            not all(math.isfinite(value) for row in matrix for value in row) or
            not all(math.isfinite(value) for value in [scale, *shifts])):
        raise ValueError("camera record has invalid orthographic projection")
    size = tuple(record.get("resolution", resolution))
    if len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in size):
        raise ValueError("camera resolution must contain two positive integers")
    aspect = [float(value) for value in record.get("pixel_aspect", [1., 1.])]
    if len(aspect) != 2 or any(not math.isfinite(value) or value <= 0 for value in aspect):
        raise ValueError("pixel aspect must contain two positive finite values")
    identity = {"projection": "ORTHO", "matrix_world": matrix, "ortho_scale": scale,
                "shift_x": shifts[0], "shift_y": shifts[1], "resolution": list(size),
                "pixel_aspect": aspect}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode("utf-8")).hexdigest()
    return identity, digest


def camera_frame_sha256(record, *, resolution=(512, 512)):
    """Hash projection/frame identity; clipping qualification is separate."""
    return _camera_identity(record, resolution)[1]


def _retained_file(directory, name, budget, *, png=False):
    path = directory / name
    row = {"path": name, "status": "unavailable", "sha256": None, "bytes": None}
    try:
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("redirected artifact paths are refused")
        if not path.resolve().is_relative_to(directory):
            raise ValueError("artifact escapes inventory directory")
        if not path.is_file():
            row["reason"] = "expected retained file is missing"
            return row
        size = path.stat().st_size
        if size > budget["per_file"] or budget["used"] + size > budget["total"]:
            raise ValueError("retained hash byte budget exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.read(24)
            digest.update(header)
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        budget["used"] += size
        row.update(status="available", sha256=digest.hexdigest(), bytes=size)
        if png:
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError("retained pass is not a PNG with a readable IHDR")
            row["resolution"] = list(struct.unpack(">II", header[16:24]))
            from PIL import Image
            with Image.open(path) as retained_image:
                if retained_image.format != "PNG" or list(retained_image.size) != row["resolution"]:
                    raise ValueError("retained PNG frame is inconsistent")
                retained_image.verify()
    except (OSError, ValueError) as exc:
        row.update(status="unavailable", reason=str(exc))
    return row


def canonical_artifact_inventory(directory, *, geometry_hash, camera_records=None,
                                 pass_states=None, resolution=(512, 512),
                                 reference_camera_records=None,
                                 per_file_byte_limit=67108864, total_byte_limit=268435456):
    """Inspect exact geometry and all expected passes, with explicit gaps.

    ``pass_states`` declares completed/unrun/unavailable for each render pass;
    an existing file never turns a declared unrun pass into completed evidence.
    Legacy camera records are normalized using the producer's declared
    resolution. Reference comparison uses projection/matrix/scale/shifts/frame,
    excluding each mask's PNG hash and camera clipping planes.
    """
    supplied = Path(directory).absolute()
    root = supplied.resolve()
    if supplied != root or supplied.is_symlink() or getattr(supplied, "is_junction", lambda: False)():
        raise ValueError("inventory directory must not be redirected")
    if not isinstance(geometry_hash, str) or len(geometry_hash) != 64 or any(c not in "0123456789abcdef" for c in geometry_hash):
        raise ValueError("evaluated geometry requires a SHA-256 identity")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (per_file_byte_limit, total_byte_limit)):
        raise ValueError("hash byte limits must be positive integers")
    states = {name: "unavailable" for name in INSPECTION_PASSES}
    states.update(pass_states or {})
    if set(states) != set(INSPECTION_PASSES) or any(value not in PASS_STATES for value in states.values()):
        raise ValueError("inspection pass states must be completed, unrun or unavailable")
    budget = {"per_file": per_file_byte_limit, "total": total_byte_limit, "used": 0}
    geometry = {"geometry_hash": geometry_hash, "identity_status": "unavailable", "files": {}}
    for name in ("evaluated.obj", "evaluated-exact.npz"):
        geometry["files"][name] = _retained_file(root, name, budget)
    archive = geometry["files"]["evaluated-exact.npz"]
    if archive["status"] == "available":
        try:
            import numpy as np
            from blender_blocking.reconstruction.native_geometry import GeometryArrays
            with zipfile.ZipFile(root / archive["path"]) as zipped:
                if sum(item.file_size for item in zipped.infolist()) > total_byte_limit:
                    raise ValueError("exact geometry decompression byte budget exceeded")
            with np.load(root / archive["path"], allow_pickle=False) as stored:
                data = GeometryArrays.capture(stored["vertices"], stored["faces"])
            geometry["archive_geometry_hash"] = data.content_hash
            geometry["identity_status"] = "verified" if data.content_hash == geometry_hash else "mismatch"
            geometry["vertex_count"], geometry["triangle_count"] = len(data.vertices), len(data.faces)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            geometry["reason"] = str(exc)
    views, retained, gaps = {}, [], []
    for view in CANONICAL_VIEWS:
        record = (camera_records or {}).get(view)
        camera = {"status": "unavailable", "sha256": None, "record": None,
                  "reference_match": "unavailable", "geometry_binding": "unavailable",
                  "clipping_status": "unavailable", "reference_clipping_match": "unavailable",
                  "reference_match_scope": "projection/frame only; clipping qualification is independent"}
        try:
            identity, digest = _camera_identity(record, resolution)
            camera.update(status="available", sha256=digest, record=identity,
                          source="retained producer camera record",
                          clipping={name: record[name] for name in ("clip_start", "clip_end") if name in record},
                          geometry_binding=("verified_by_producer" if record.get("geometry_hash") == geometry_hash and
                                            record.get("geometry_unchanged_after_passes") is True else "unavailable"))
            if record.get("geometry_hash") is not None and record["geometry_hash"] != geometry_hash:
                camera.update(status="unavailable", geometry_binding="mismatch",
                              reason="render camera receipt belongs to different evaluated geometry")
            clipping = camera["clipping"]
            if set(clipping) == {"clip_start", "clip_end"} and all(math.isfinite(float(value)) for value in clipping.values()) and 0 < float(clipping["clip_start"]) < float(clipping["clip_end"]):
                camera["clipping_status"] = "available"
            if reference_camera_records is not None:
                _, reference_digest = _camera_identity(reference_camera_records.get(view), resolution)
                camera["reference_match"] = "matched" if digest == reference_digest else "mismatch"
                camera["reference_sha256"] = reference_digest
                reference_clipping = {name: reference_camera_records[view][name] for name in ("clip_start", "clip_end") if name in reference_camera_records[view]}
                if camera["clipping_status"] == "available" and set(reference_clipping) == {"clip_start", "clip_end"}:
                    camera["reference_clipping_match"] = "matched" if clipping == reference_clipping else "mismatch"
        except (ValueError, KeyError, TypeError) as exc:
            camera["reason"] = str(exc)
        artifacts = {}
        for name in INSPECTION_PASSES:
            path = view + "-" + name + ".png"
            row = _retained_file(root, path, budget, png=True)
            row.update(pass_name=name, geometry_hash=geometry_hash, camera_sha256=camera["sha256"],
                       producer_status=states[name], geometry_identity_status=geometry["identity_status"])
            if states[name] != "completed":
                row["retained_file_status"] = row["status"]
                row["status"] = states[name]
                row["reason"] = "producer did not execute this pass" if states[name] == "unrun" else "producer availability was not established"
            elif row["status"] == "available" and camera["status"] == "available" and row["resolution"] != camera["record"]["resolution"]:
                row.update(status="unavailable", reason="PNG resolution differs from retained camera frame")
            if row["status"] == "available" and name == "mask" and record and record.get("png_sha256") is not None and row["sha256"] != record["png_sha256"]:
                row.update(status="unavailable", reason="mask hash differs from retained camera receipt")
            row["render_settings"] = (record or {}).get("render_passes", {}).get(name)
            binding = (record or {}).get("pass_artifacts", {}).get(name)
            row["geometry_binding"] = "unavailable"
            if isinstance(binding, dict):
                row["producer_binding"] = dict(binding)
                matches = (camera["geometry_binding"] == "verified_by_producer" and
                           binding.get("geometry_hash") == geometry_hash and
                           binding.get("camera_sha256") == camera["sha256"] and
                           binding.get("sha256") == row["sha256"])
                row["geometry_binding"] = "verified_by_producer" if matches else "mismatch"
            if row["status"] == "available":
                retained.append(path)
            if name in REQUIRED_PASSES and (row["status"] != "available" or camera["status"] != "available" or
                                              geometry["identity_status"] != "verified" or
                                              row["geometry_binding"] != "verified_by_producer" or
                                              camera["clipping_status"] != "available" or
                                              camera["reference_match"] == "mismatch" or
                                              camera["reference_clipping_match"] == "mismatch"):
                gaps.append({"view": view, "pass_name": name, "artifact_status": row["status"],
                             "camera_status": camera["status"], "reference_match": camera["reference_match"],
                             "geometry_identity_status": geometry["identity_status"],
                             "geometry_binding": row["geometry_binding"], "clipping_status": camera["clipping_status"],
                             "reference_clipping_match": camera["reference_clipping_match"]})
            artifacts[name] = row
        views[view] = {"camera": camera, "artifacts": artifacts}
    return {"schema_version": 1, "protocol": "canonical_matched_inspection_v1", "directory": str(root),
            "canonical_views": list(CANONICAL_VIEWS), "required_passes": list(REQUIRED_PASSES),
            "geometry": geometry, "views": views, "retained_paths": retained, "gaps": gaps,
            "status": "complete" if not gaps else "incomplete", "hashed_bytes": budget["used"],
            "acceptance": "independent; complete retained inspection does not establish quality",
            "presentation_gallery": "curation never removes canonical inventory entries"}


def raw_surface_observation(reference, candidate, **metric_options):
    """Evaluate actual arrays while withholding unqualified family gate claims."""
    if reference is None or candidate is None:
        return {"metric_status": "unrun", "surface_status": "unqualified", "qualified_limits": None,
                "reason": "actual reference and candidate geometry are both required"}
    from blender_blocking.evaluation.surface_quality import compare_surface_arrays
    result = dict(compare_surface_arrays(reference, candidate, **metric_options))
    result.pop("surface_passed", None)
    result.pop("frozen_limits", None)
    result.update(metric_status="measured", surface_status="unqualified", qualified_limits=None,
                  surface_qualification="independent family reference tessellation/noise limits are unavailable; raw observations only")
    return result


def triangle_diagnostic_control_screen(observation):
    """Keep the old deliberate-control screen separate from family acceptance."""
    thresholds = {"symmetric_mean_distance_world_max": .003,
                  "normal_angle_p95_degrees_max": 2.5}
    mean = float(observation["symmetric_mean_distance_world"])
    normal = float(observation["normal_angle_p95_degrees"])
    if not math.isfinite(mean) or not math.isfinite(normal):
        raise ValueError("diagnostic control observations must be finite")
    return {"status": "measured_diagnostic", "borrowed_thresholds": thresholds,
            "diagnostic_passed": mean <= thresholds["symmetric_mean_distance_world_max"] and
                                 normal <= thresholds["normal_angle_p95_degrees_max"],
            "scope": "borrowed vase thresholds; no triangle family acceptance qualification",
            "family_acceptance": "blocked; independent triangle tolerance remains unavailable"}
