"""Explicit linear pixel coverage; segmentation probabilities are not coverage.

A measured display transfer may decode a rendered grayscale image. No transform
is guessed from image colors, and the original hard silhouette gates stay intact.
"""
from __future__ import annotations

import numpy as np


def coverage_from_grayscale(image, linear_samples, encoded_samples):
    """Invert a declared black/white display transfer into foreground coverage.

    Samples describe encoded grayscale for linear background fractions 0..1.
    Quantized plateaus use their midpoint. One 8-bit code of endpoint dither is
    tolerated; an incompatible image/transfer fails instead of inventing evidence.
    """
    image = np.asarray(image, float)
    linear = np.asarray(linear_samples, float)
    encoded = np.asarray(encoded_samples, float)
    if (
        linear.ndim != 1 or encoded.shape != linear.shape
        or not 64 <= len(linear) <= 8193
        or not np.isfinite(np.r_[linear, encoded]).all()
        or linear[0] != 0 or linear[-1] != 1 or (np.diff(linear) <= 0).any()
        or (np.diff(encoded) < 0).any() or encoded[0] < 0 or encoded[-1] > 1
        or encoded[-1] <= encoded[0] or image.ndim != 2
        or not np.isfinite(image).all()
        or image.min() < encoded[0] - 1 / 255
        or image.max() > encoded[-1] + 1 / 255
    ):
        raise ValueError("coverage requires a compatible image and declared "
                         "monotone grayscale transfer")
    values, inverse, counts = np.unique(encoded, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=linear) / counts
    # Known pure endpoints remain pure even when their encoded values plateau.
    means[0], means[-1] = 0., 1.
    return 1 - np.interp(image, values, means)


def constraint_coverage(constraint, valid):
    supplied = getattr(constraint, "coverage_mask", None)
    if supplied is None:
        return None
    coverage = np.asarray(supplied, float)
    if (coverage.shape != valid.shape or not np.isfinite(coverage[valid]).all()
            or (coverage[valid] < 0).any() or (coverage[valid] > 1).any()):
        raise ValueError("linear coverage must match the mask and be finite "
                         "in [0, 1] at observed pixels")
    return np.where(valid, coverage, 0.)


def coverage_interval(values, valid=None):
    """Extreme half-coverage crossings in pixel-cell coordinates.

    Pixel i has center i+.5. Edges touching an unknown or external pixel are
    censored. This remains a filtered-edge estimate, not an exact hidden contour.
    """
    values = np.asarray(values, float)
    valid = np.ones(values.shape, bool) if valid is None else np.asarray(valid, bool)
    if (values.ndim != 1 or values.shape != valid.shape
            or not np.isfinite(values[valid]).all()
            or (values[valid] < 0).any() or (values[valid] > 1).any()):
        raise ValueError("coverage row and validity must be matching finite "
                         "bounded arrays")
    foreground = np.flatnonzero((values >= .5) & valid)
    if not len(foreground):
        return None, None
    first, last = int(foreground[0]), int(foreground[-1])
    left = right = None
    if first > 0 and valid[first - 1] and values[first - 1] < .5:
        left = first - .5 + (.5 - values[first - 1]) / (values[first] - values[first - 1])
    if last + 1 < len(values) and valid[last + 1] and values[last + 1] < .5:
        right = last + .5 + (values[last] - .5) / (values[last] - values[last + 1])
    return left, right


def coverage_bbox(coverage):
    """Complete observed half-coverage envelope; clipping remains unavailable."""
    coverage = np.asarray(coverage, float)
    if (coverage.ndim != 2 or not np.isfinite(coverage).all()
            or (coverage < 0).any() or (coverage > 1).any()):
        raise ValueError("coverage image must be finite in [0, 1]")
    left, right = coverage_interval(coverage.max(axis=0))
    top, bottom = coverage_interval(coverage.max(axis=1))
    if any(v is None for v in (left, right, top, bottom)):
        raise ValueError("coverage bounds are empty or censored by the image edge")
    return left, top, right, bottom
