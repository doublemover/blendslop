from __future__ import annotations

from .adaptive import _cmd_adapt, _cmd_loop
from .autopsy import _cmd_autopsy
from .common import _add_plan_args, _add_run_args
from .human import _cmd_label, _cmd_promote, _cmd_score_study, _cmd_study_pack
from .list_presets import _cmd_list_suites, _cmd_list_tracks
from .masks import _cmd_calibrate_masks, _cmd_patch_masks
from .report import _cmd_rank, _cmd_report
from .run import _cmd_plan, _cmd_run
from .surrogate import _cmd_surrogate

__all__ = [name for name in globals() if name.startswith("_cmd_") or name.startswith("_add_")]
