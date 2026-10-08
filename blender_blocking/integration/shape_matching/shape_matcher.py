"""Shape matching and comparison utilities."""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Union

from config import CanonicalizeConfig, SilhouetteExtractConfig
from validation.silhouette_iou import (
    canonicalize_mask_cached,
    canonicalize_mask_with_metadata,
    compute_silhouette_metrics,
    compute_mask_iou,
    uncertain_mask_from_image_array,
)


def match_shapes(
    contour1: np.ndarray, contour2: np.ndarray, method: int = cv2.CONTOURS_MATCH_I2
) -> float:
    """
    Compare two contours using shape matching.

    Args:
        contour1: First contour
        contour2: Second contour
        method: OpenCV matching method

    Returns:
        Similarity score (lower is better, 0 is perfect match)
    """
    if contour1 is None or contour2 is None:
        raise ValueError("Contours must not be None")
    if len(contour1) < 3 or len(contour2) < 3:
        raise ValueError("Contours must have at least 3 points to match")
    return cv2.matchShapes(contour1, contour2, method, 0.0)


def compare_silhouettes(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    output_size: int = 256,
    padding_frac: float = 0.1,
    anchor: str = "bottom_center",
    extract_config: Optional[SilhouetteExtractConfig] = None,
    canonicalize_config: Optional[CanonicalizeConfig] = None,
) -> Tuple[float, Dict[str, Union[float, int, str]]]:
    """
    Compare silhouettes from two images.

    Args:
        image1: First binary image
        image2: Second binary image

    Returns:
        Tuple of (similarity_score, comparison_details)
    """
    if image1 is None or image2 is None or image1.size == 0 or image2.size == 0:
        raise ValueError("Images must be non-empty for silhouette comparison")

    uncertain1 = uncertain_mask_from_image_array(image1, extract_config=extract_config)
    uncertain2 = uncertain_mask_from_image_array(image2, extract_config=extract_config)
    mask1 = uncertain1.hard_mask
    mask2 = uncertain2.hard_mask

    canon1 = canonicalize_mask_cached(
        mask1,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        canonicalize_config=canonicalize_config,
    )
    canon2 = canonicalize_mask_cached(
        mask2,
        output_size=output_size,
        padding_frac=padding_frac,
        anchor=anchor,
        canonicalize_config=canonicalize_config,
    )
    if canonicalize_config is None:
        canon_meta1 = canonicalize_mask_with_metadata(
            mask1,
            output_size=output_size,
            padding_frac=padding_frac,
            anchor=anchor,
        )
        canon_meta2 = canonicalize_mask_with_metadata(
            mask2,
            output_size=output_size,
            padding_frac=padding_frac,
            anchor=anchor,
        )
    else:
        canon_meta1 = canonicalize_mask_with_metadata(
            mask1,
            canonicalize_config=canonicalize_config,
        )
        canon_meta2 = canonicalize_mask_with_metadata(
            mask2,
            canonicalize_config=canonicalize_config,
        )

    result = compute_mask_iou(canon1, canon2)
    metrics = compute_silhouette_metrics(
        canon1,
        canon2,
        view="shape_match",
        pass_required=True,
    )
    difference = float(np.abs(canon1.astype(float) - canon2.astype(float)).mean())

    details: Dict[str, Union[float, int, str]] = {
        "iou": result.iou,
        "area_iou": metrics.area_iou,
        "boundary_iou": metrics.boundary_iou if metrics.boundary_iou is not None else -1.0,
        "signed_distance_loss": (
            metrics.signed_distance_loss
            if metrics.signed_distance_loss is not None
            else -1.0
        ),
        "intersection": result.intersection,
        "union": result.union,
        "ref_area": result.ref_area,
        "candidate_area": result.candidate_area,
        "pixel_difference": difference,
        "source1": uncertain1.source,
        "source2": uncertain2.source,
        "threshold1": uncertain1.threshold if uncertain1.threshold is not None else -1.0,
        "threshold2": uncertain2.threshold if uncertain2.threshold is not None else -1.0,
        "canonical_scale1": float(canon_meta1.transform.get("scale", 0.0)),
        "canonical_scale2": float(canon_meta2.transform.get("scale", 0.0)),
    }
    warnings = tuple(result.warnings) + tuple(metrics.warnings)
    if warnings:
        details["warning"] = "; ".join(warnings)

    return result.iou, details
