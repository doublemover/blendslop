"""Blender export/reimport QA helpers for editable reconstruction outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import bpy

    BLENDER_AVAILABLE = True
except Exception:  # pragma: no cover - pure Python environments
    bpy = None  # type: ignore
    BLENDER_AVAILABLE = False

try:
    from blender_blocking.evaluation.export_qa import ExportQAReport
except Exception:  # pragma: no cover - script-style imports
    from evaluation.export_qa import ExportQAReport  # type: ignore


_SUPPORTED_TARGETS = {"obj", "glb", "gltf"}


def run_export_roundtrip_qa(
    objects: Sequence[Any],
    output_root: str | Path,
    *,
    targets: Sequence[str] = ("obj", "glb"),
    cleanup_imports: bool = True,
) -> tuple[ExportQAReport, ...]:
    """Export selected objects, reimport them, and summarize asset health."""
    if not BLENDER_AVAILABLE:
        return (
            ExportQAReport(
                target="blender",
                status="skipped",
                reimport_status="skipped",
                warnings=("Blender export QA requires bpy",),
            ),
        )
    selected_objects = tuple(obj for obj in objects if obj is not None)
    if not selected_objects:
        return (
            ExportQAReport(
                target="blender",
                status="fail",
                reimport_status="skipped",
                errors=("no objects provided for export QA",),
            ),
        )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for target in _normalize_targets(targets):
        reports.append(
            _roundtrip_one(
                selected_objects,
                root,
                target,
                cleanup_imports=cleanup_imports,
            )
        )
    return tuple(reports)


def _roundtrip_one(
    objects: Sequence[Any],
    root: Path,
    target: str,
    *,
    cleanup_imports: bool,
) -> ExportQAReport:
    path = root / f"roundtrip.{_extension_for_target(target)}"
    warnings: list[str] = []
    errors: list[str] = []
    try:
        _select_only(objects)
        _export_selected(target, path)
    except Exception as exc:
        return ExportQAReport(
            target=target,
            status="fail",
            exported_path=str(path),
            reimport_status="skipped",
            warnings=tuple(warnings),
            errors=(f"export failed: {exc}",),
        )

    before = set(bpy.data.objects.keys())
    imported: tuple[Any, ...] = ()
    try:
        _import_file(target, path)
        imported = tuple(
            obj for obj in bpy.data.objects if getattr(obj, "name", "") not in before
        )
        if not imported:
            errors.append("reimport completed but produced no objects")
    except Exception as exc:
        errors.append(f"reimport failed: {exc}")

    counts = _object_counts(imported)
    if cleanup_imports and imported:
        _remove_objects(imported)
    return ExportQAReport(
        target=target,
        status="pass" if path.exists() and not errors else "fail",
        exported_path=str(path),
        reimport_status="pass" if imported and not errors else "fail",
        object_count=counts["object_count"],
        vertex_count=counts["vertex_count"],
        face_count=counts["face_count"],
        material_count=counts["material_count"],
        bounds=counts["bounds"],
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _normalize_targets(targets: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for target in targets:
        value = str(target).strip().lower()
        if value == "gltf":
            value = "glb"
        if value and value in _SUPPORTED_TARGETS and value not in normalized:
            normalized.append(value)
    return tuple(normalized or ("obj",))


def _extension_for_target(target: str) -> str:
    if target == "obj":
        return "obj"
    return "glb" if target in {"glb", "gltf"} else target


def _select_only(objects: Sequence[Any]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    active = None
    for obj in objects:
        try:
            obj.select_set(True)
            if active is None:
                active = obj
        except Exception:
            continue
    if active is not None:
        bpy.context.view_layer.objects.active = active


def _export_selected(target: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if target == "obj":
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(
                filepath=str(path),
                export_selected_objects=True,
            )
        else:
            bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)
        return
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
    )


def _import_file(target: str, path: Path) -> None:
    if target == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
        return
    bpy.ops.import_scene.gltf(filepath=str(path))


def _object_counts(objects: Sequence[Any]) -> dict[str, Any]:
    vertex_count = 0
    face_count = 0
    material_count = 0
    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    for obj in objects:
        data = getattr(obj, "data", None)
        vertices = getattr(data, "vertices", ()) or ()
        polygons = getattr(data, "polygons", ()) or ()
        materials = getattr(data, "materials", ()) or ()
        vertex_count += len(vertices)
        face_count += len(polygons)
        material_count += len(materials)
        bound_box = getattr(obj, "bound_box", None)
        matrix = getattr(obj, "matrix_world", None)
        if bound_box is None:
            continue
        points = []
        for corner in bound_box:
            point = _transform_point(corner, matrix)
            points.append((float(point[0]), float(point[1]), float(point[2])))
        if points:
            mins.append(tuple(min(point[index] for point in points) for index in range(3)))
            maxs.append(tuple(max(point[index] for point in points) for index in range(3)))
    bounds = None
    if mins and maxs:
        min_all = tuple(min(point[index] for point in mins) for index in range(3))
        max_all = tuple(max(point[index] for point in maxs) for index in range(3))
        bounds = tuple(max_all[index] - min_all[index] for index in range(3))
    return {
        "object_count": len(objects),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "material_count": material_count,
        "bounds": bounds,
    }


def _remove_objects(objects: Sequence[Any]) -> None:
    for obj in objects:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _transform_point(point: Any, matrix: Any) -> Any:
    if matrix is None:
        return point
    try:
        return matrix @ point
    except Exception:
        try:
            from mathutils import Vector

            return matrix @ Vector(point)
        except Exception:
            return point
