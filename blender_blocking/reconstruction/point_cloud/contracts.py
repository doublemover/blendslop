from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PointCloudArtifactStatus:
    source: str
    point_count: int
    degraded_substitute: bool = False
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "point_count": int(self.point_count),
            "degraded_substitute": bool(self.degraded_substitute),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
