"""Small pure-Numpy binary morphology helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def as_bool_mask(mask: Any, *, name: str = "mask", allow_empty: bool = True) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array.astype(bool, copy=False)


def binary_erosion(mask: Any, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(neighborhood_3x3(padded, result.shape))
    return result


def binary_dilation(mask: Any, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(1, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(neighborhood_3x3(padded, result.shape))
    return result


def boundary_band(mask: Any, radius: int = 2) -> np.ndarray:
    hard = np.asarray(mask, dtype=bool)
    if not hard.any():
        return np.zeros(hard.shape, dtype=bool)
    eroded = binary_erosion(hard, radius=1)
    boundary = np.logical_xor(hard, eroded)
    return binary_dilation(boundary, radius=max(1, int(radius)))


def neighborhood_3x3(padded: np.ndarray, shape: Sequence[int]) -> list[np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    return [
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
    ]
