from __future__ import annotations

from .suites import SUITES
from .tracks import TRACKS
from .types import SuitePreset, TrackPreset


def list_suites() -> tuple[str, ...]:
    return tuple(sorted(SUITES))


def list_tracks() -> tuple[str, ...]:
    return tuple(sorted(TRACKS))


def get_suite_preset(name: str) -> SuitePreset:
    try:
        return SUITES[name]
    except KeyError as exc:
        raise KeyError(f"unknown refinement suite {name!r}; available: {', '.join(list_suites())}") from exc


def get_track_preset(name: str) -> TrackPreset:
    try:
        return TRACKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown refinement track {name!r}; available: {', '.join(list_tracks())}") from exc


__all__ = [
    "SUITES",
    "TRACKS",
    "SuitePreset",
    "TrackPreset",
    "get_suite_preset",
    "get_track_preset",
    "list_suites",
    "list_tracks",
]
