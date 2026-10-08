from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BLENDER_BLOCKING_ROOT.parent
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.progress import progress_bar


def _now() -> float:
    return time.perf_counter()


def _make_rect_silhouette(width: int, height: int, ratio: float = 0.4) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rect_w = max(1, int(width * ratio))
    x0 = (width - rect_w) // 2
    mask[:, x0 : x0 + rect_w] = True
    return mask


def _make_circle_silhouette(width: int, height: int, ratio: float = 0.35) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cx = width // 2
    cy = height // 2
    radius = max(1, int(min(width, height) * ratio))
    y, x = np.ogrid[:height, :width]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius**2] = True
    return mask


def _sample_cylinder_points(
    radius: float,
    height: float,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    side_count = int(num_points * 0.7)
    cap_count = num_points - side_count

    theta = rng.uniform(0.0, 2.0 * np.pi, size=side_count)
    z = rng.uniform(-height / 2.0, height / 2.0, size=side_count)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    side = np.column_stack([x, y, z])

    r = np.sqrt(rng.uniform(0.0, radius**2, size=cap_count))
    theta_cap = rng.uniform(0.0, 2.0 * np.pi, size=cap_count)
    x_cap = r * np.cos(theta_cap)
    y_cap = r * np.sin(theta_cap)
    z_cap = rng.choice([-height / 2.0, height / 2.0], size=cap_count)
    caps = np.column_stack([x_cap, y_cap, z_cap])

    return np.vstack([side, caps])


def _sample_cone_points(
    radius_bottom: float,
    radius_top: float,
    height: float,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    t = rng.uniform(0.0, 1.0, size=num_points)
    z = -height / 2.0 + t * height
    radii = radius_bottom * (1.0 - t) + radius_top * t
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.column_stack([x, y, z])


def _sample_sphere_points(
    radius: float, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=num_points)
    v = rng.uniform(0.0, 1.0, size=num_points)
    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _sample_rotated_cube_points(
    half_extent: float,
    num_points: int,
    rng: np.random.Generator,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    faces = rng.integers(0, 6, size=num_points)
    coords = rng.uniform(-half_extent, half_extent, size=(num_points, 3))
    coords[np.where(faces == 0), 0] = -half_extent
    coords[np.where(faces == 1), 0] = half_extent
    coords[np.where(faces == 2), 1] = -half_extent
    coords[np.where(faces == 3), 1] = half_extent
    coords[np.where(faces == 4), 2] = -half_extent
    coords[np.where(faces == 5), 2] = half_extent

    if rotation is None:
        rotation = np.array(
            [
                [0.8660254, -0.3535534, 0.3535534],
                [0.5, 0.6123724, -0.6123724],
                [0.0, 0.7071068, 0.7071068],
            ]
        )

    return coords @ rotation.T
