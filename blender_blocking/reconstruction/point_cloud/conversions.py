from __future__ import annotations

import numpy as np


def boolean_dense(array: object) -> np.ndarray:
    return np.asarray(array, dtype=bool)
