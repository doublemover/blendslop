from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .dependencies import *


@dataclass
class RefinementLabConfig:
    """Configuration for local reconstruction refinement experiments."""

    default_output_root: str = "temp/refinement-runs"
    default_suite: str = "default-vase"
    default_track: str = "profile-loft-refinement"
    default_search: str = "grid"
    default_objective: str = "quality_win"
    max_runs: Optional[int] = None
    top_k: int = 10
    html_report: bool = True
    write_overlays: bool = True
    write_bounds_debug: bool = True
    write_autopsy: bool = True
    append_leaderboard: bool = True
    fail_on_all_failed: bool = True
    allow_subprocess_blender: bool = False
    blender_executable: Optional[str] = None
    copy_references: bool = False
    report_failures: str = "top"
    stop_on_first_error: bool = False

    def validate(self) -> None:
        if not self.default_output_root:
            raise ValueError("refinement_lab.default_output_root is required")
        if not self.default_suite:
            raise ValueError("refinement_lab.default_suite is required")
        if not self.default_track:
            raise ValueError("refinement_lab.default_track is required")
        if self.default_search not in _VALID_REFINEMENT_SEARCH:
            raise ValueError(
                f"refinement default_search must be one of {_VALID_REFINEMENT_SEARCH}"
            )
        if self.default_objective not in _VALID_REFINEMENT_OBJECTIVES:
            raise ValueError(
                f"refinement default_objective must be one of {_VALID_REFINEMENT_OBJECTIVES}"
            )
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("refinement max_runs must be >= 1 when provided")
        if self.top_k < 1:
            raise ValueError("refinement top_k must be >= 1")
        if self.report_failures not in _VALID_REFINEMENT_REPORT_FAILURES:
            raise ValueError(
                "refinement report_failures must be one of "
                f"{_VALID_REFINEMENT_REPORT_FAILURES}"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "default_output_root": self.default_output_root,
            "default_suite": self.default_suite,
            "default_track": self.default_track,
            "default_search": self.default_search,
            "default_objective": self.default_objective,
            "max_runs": self.max_runs,
            "top_k": self.top_k,
            "html_report": self.html_report,
            "write_overlays": self.write_overlays,
            "write_bounds_debug": self.write_bounds_debug,
            "write_autopsy": self.write_autopsy,
            "append_leaderboard": self.append_leaderboard,
            "fail_on_all_failed": self.fail_on_all_failed,
            "allow_subprocess_blender": self.allow_subprocess_blender,
            "blender_executable": self.blender_executable,
            "copy_references": self.copy_references,
            "report_failures": self.report_failures,
            "stop_on_first_error": self.stop_on_first_error,
        }
