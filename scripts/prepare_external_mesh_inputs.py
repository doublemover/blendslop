"""Prepare an external GLB in Blender, with a review gate before reference images.

Run one mesh per Blender process. Inspection copies evaluated world geometry and
saves an editable imported scene without rendering benchmark images. After
inspecting that scene, supply a source-hash-matched review receipt to --render.
This does not reconstruct, score, time, download or install anything.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender_blocking")]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("input preparation is restricted to the authorized repository")
    return path


def reviewed_source(source, receipt_path):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (receipt.get("visual_reviewed") is not True or receipt.get("canonical_axes_reviewed") is not True or
            receipt.get("source_sha256") != file_hash(source)):
        raise ValueError("reference images require a hash-matched visual and canonical-axis review")
    return receipt


def canonical_calibration(lower, upper):
    """Match the existing frozen orthographic input and novel-camera contract."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
        raise ValueError("finite three-dimensional bounds required")
    center = (lower + upper) * .5
    scale = float((upper - lower).max()) * 1.2
    if np.any(upper < lower) or scale <= 0:
        raise ValueError("bounds must have positive spatial extent")
    calibration = {}
    for view, axes in {"front": (0,2), "side": (1,2), "top": (0,1)}.items():
        u, v = center[axes[0]], center[axes[1]]
        calibration[view] = {"projection": "orthographic", "axes": list(axes),
            "world_bounds": [float(u-scale/2), float(u+scale/2), float(v-scale/2), float(v+scale/2)],
            "world_units": "metres", "orientation": "canonical_positive_axes", "source": "recorded_capture_camera"}
    return calibration, center, scale


def prepare(args):
    source, output = repository_path(args.source), repository_path(args.output)
    if source.suffix.lower() != ".glb":
        raise ValueError("use self-contained GLB inputs; external glTF dependencies are not inferred")
    if output.exists():
        raise FileExistsError("refusing to replace an existing input preparation directory")
    review = None
    if args.render:
        if args.review_receipt is None:
            raise ValueError("--render requires --review-receipt after actual mesh inspection")
        review = reviewed_source(source, repository_path(args.review_receipt))
    elif args.review_receipt:
        raise ValueError("a review receipt is only consumed by --render")
    # Validate admission before importing anything or creating output files.
    import bpy
    from blender_blocking.verify_setup import configure_dependency_paths
    configure_dependency_paths()
    from blender_blocking.reconstruction.native_geometry import evaluated_arrays
    from blender_blocking.reconstruction.mesh_io import write_obj
    from blender_blocking.metrics.topology import mesh_topology_report
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("source contains no mesh geometry")
    if any(obj.hide_render for obj in meshes):
        raise ValueError("hidden source meshes require an explicit visibility decision before preparation")
    roots = [obj for obj in bpy.data.objects if obj.parent is None and
             (obj.type == "MESH" or any(child.type == "MESH" for child in obj.children_recursive))]
    raw_faces = sum(len(obj.data.polygons) for obj in meshes)
    if raw_faces > args.max_triangles:
        raise ValueError("source exceeds explicit input polygon budget")
    vertices, faces = [], []
    vertex_count = 0
    for obj in roots:
        data = evaluated_arrays(obj)
        if sum(len(part) for part in faces) + len(data.faces) > args.max_triangles:
            raise ValueError("evaluated source exceeds explicit triangle budget")
        vertices.append(data.vertices)
        faces.append(data.faces + vertex_count)
        vertex_count += len(data.vertices)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if float((upper - lower).max()) <= 1e-12:
        raise ValueError("source has no spatial extent")
    output.mkdir(parents=True, exist_ok=False)
    mesh_path = write_obj(output / "ground_truth.obj", {"vertices": vertices, "faces": faces})
    scene_path = output / "inspection.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path), check_existing=False)
    result = {"schema": "blendslop_external_mesh_input_v1", "case_id": args.case_id,
              "dataset": args.dataset, "selection_note": args.selection_note,
              "source": str(source), "source_sha256": file_hash(source),
              "ground_truth": str(mesh_path), "ground_truth_sha256": file_hash(Path(mesh_path)),
              "inspection_scene": str(scene_path), "reference_images_available": False,
              "frame": "Blender imported glTF world coordinates, canonical Z-up",
              "unit_convention": "glTF nominal metres; artist physical scale unverified",
              "bounds": [lower.tolist(), upper.tolist()], "vertex_count": len(vertices), "triangle_count": len(faces),
              "topology": mesh_topology_report(vertices, faces).to_dict(),
              "objects": [{"name": obj.name, "materials": len(obj.material_slots),
                           "uv_layers": len(obj.data.uv_layers), "modifiers": [mod.type for mod in obj.modifiers],
                           "animated": obj.animation_data is not None} for obj in meshes],
              "evaluation_frame": bpy.context.scene.frame_current,
              "blender_version": bpy.app.version_string, "review_receipt": review,
              "native_paper_compatibility": "unverified; capability subset only"}
    if args.render or getattr(args, "inspect_render", False):
        from blender_blocking.config import RenderConfig
        from blender_blocking.integration.blender_ops.render_utils import render_orthogonal_views_detailed
        calibration, center, scale = canonical_calibration(lower, upper)
        inspection = not args.render
        image_size = 256 if inspection else 512
        config = RenderConfig(resolution=(image_size,image_size), view_calibration=calibration, force_material=not inspection)
        rendered = render_orthogonal_views_detailed(str(output), target_objects=meshes, render_config=config)
        from blender_blocking.integration.blender_ops.silhouette_render import (
            silhouette_session, set_camera_orbit, render_silhouette_frame)
        from mathutils import Vector
        import math
        novel = output / "orbit45.png"
        with silhouette_session(target_objects=meshes, resolution=(image_size,image_size), color_mode="RGBA",
                                transparent_bg=True, force_material=not inspection, engine="BLENDER_EEVEE") as session:
            set_camera_orbit(session.camera, Vector(center), scale * 2., math.radians(-45.), scale)
            render_silhouette_frame(session, novel)
        result.update(reference_images_available=bool(args.render), inspection_images_available=inspection,
                      views=rendered.paths, calibration=calibration,
                      novel=str(novel), novel_center=center.tolist(), novel_scale=scale,
                      render_metadata=rendered.to_dict(),
                      hashes={key: file_hash(Path(path)) for key, path in rendered.paths.items()})
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection-note", required=True)
    parser.add_argument("--max-triangles", type=int, default=2_000_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--inspect-render", action="store_true", help="material-preserving previews for actual source review")
    parser.add_argument("--review-receipt", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.max_triangles < 1:
        parser.error("--max-triangles must be positive")
    result = prepare(args)
    print(f"Prepared {result['case_id']}; reference images available: {result['reference_images_available']}")


if __name__ == "__main__":
    main()
