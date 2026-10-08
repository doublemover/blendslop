from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import PointCloudArtifactStatus


def point_cloud_status_payload(
    points: np.ndarray,
    *,
    source: str,
    degraded_substitute: bool = False,
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    points = np.asarray(points)
    status = PointCloudArtifactStatus(
        source=source,
        point_count=int(len(points)) if points.ndim >= 1 else 0,
        degraded_substitute=degraded_substitute,
        reason=reason,
        metadata=dict(metadata or {}),
    )
    return status.to_dict()
