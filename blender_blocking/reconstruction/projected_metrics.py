"""Measure exported geometry against observed pixels, independently of confidence."""
from __future__ import annotations
import numpy as np


def projected_mesh_masks(target, vertices, faces):
    from PIL import Image, ImageDraw
    from .projection_contract import project_vertices
    result = {}
    for constraint in target.constraints:
        ref = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
        if ref.ndim != 2:
            continue
        height, width = ref.shape
        xy = project_vertices(target, constraint, vertices)
        canvas = Image.new("1", (width, height))
        draw = ImageDraw.Draw(canvas)
        for face in faces:
            if len(face) >= 3:
                draw.polygon([tuple(xy[i]) for i in face], fill=1)
        result[constraint.view] = np.asarray(canvas, bool)
    return result


def projected_mesh_metrics(target, vertices, faces):
    from .visibility import evaluate_visible_pair
    from blender_blocking.evaluation.silhouette_eval import SilhouetteGateConfig
    masks = projected_mesh_masks(target, vertices, faces)
    result = {}
    for constraint in target.constraints:
        if constraint.view not in masks:
            continue
        ref = np.asarray(getattr(constraint.mask, "mask", constraint.mask), bool)
        metrics = evaluate_visible_pair(ref, masks[constraint.view], constraint, view=constraint.view,
                                       config=SilhouetteGateConfig(min_area_iou=.7), required=True)
        metrics["candidate_projection_source"] = "exported_mesh_polygon_raster"
        result[constraint.view] = metrics
    return result
