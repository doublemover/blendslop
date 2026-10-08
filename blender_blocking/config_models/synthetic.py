from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *


@dataclass
class SyntheticFactoryConfig:
    """Configuration for synthetic fixture generation."""

    suite: str = "smoke"
    seed: int = 1234
    output_root: str = "temp/synthetic"
    commit_small_fixtures_only: bool = True
    keep_heavy_artifacts: bool = False

    def validate(self) -> None:
        if not self.suite:
            raise ValueError("synthetic suite is required")
        if not self.output_root:
            raise ValueError("synthetic output_root is required")

    def to_dict(self) -> Dict[str, object]:
        return {
            "suite": self.suite,
            "seed": self.seed,
            "output_root": self.output_root,
            "commit_small_fixtures_only": self.commit_small_fixtures_only,
            "keep_heavy_artifacts": self.keep_heavy_artifacts,
        }
