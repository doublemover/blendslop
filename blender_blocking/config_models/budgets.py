from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *


@dataclass
class QualityBudgetConfig:
    """Configuration for quality/performance budget checks."""

    budget_json: Optional[str] = None
    compare_baseline: Optional[str] = None
    fail_on_regression: bool = False
    environment_compatibility: str = "warn"

    def validate(self) -> None:
        if self.environment_compatibility not in {"warn", "strict", "ignore"}:
            raise ValueError("environment_compatibility must be warn/strict/ignore")

    def to_dict(self) -> Dict[str, object]:
        return {
            "budget_json": self.budget_json,
            "compare_baseline": self.compare_baseline,
            "fail_on_regression": self.fail_on_regression,
            "environment_compatibility": self.environment_compatibility,
        }
