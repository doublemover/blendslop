"""Tests for shared reconstruction target signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

import numpy as np

BLENDER_BLOCKING_ROOT = Path(__file__).resolve().parent
if str(BLENDER_BLOCKING_ROOT) not in sys.path:
    sys.path.insert(0, str(BLENDER_BLOCKING_ROOT))

try:
    from reconstruction.target_signals import collect_target_signals
    from reconstruction.types import ProfileBand, ProfileIntervalPx, ReconstructionTarget
except ModuleNotFoundError:  # pragma: no cover - package unittest path
    from blender_blocking.reconstruction.target_signals import collect_target_signals
    from blender_blocking.reconstruction.types import (
        ProfileBand,
        ProfileIntervalPx,
        ReconstructionTarget,
    )


@dataclass(frozen=True)
class _Uncertainty:
    confidence: object
    boundary_uncertainty: object


@dataclass(frozen=True)
class _Constraint:
    view: str
    kind: str
    uncertainty: _Uncertainty | None = None


class TargetSignalsTests(unittest.TestCase):
    def test_primitive_fit_signal_includes_profile_rows(self) -> None:
        target = _fixture_target()

        signals = collect_target_signals(
            target,
            include_profile_rows=True,
            include_surface_density=False,
            include_constraint_kinds=False,
            include_topology_details=False,
        )

        self.assertEqual(set(signals), {"surface", "profile", "constraints", "uncertainty", "topology"})
        self.assertNotIn("density_hint", signals["surface"])
        self.assertNotIn("constraint_kinds", signals["constraints"])
        self.assertIn("rows", signals["profile"])
        rows = signals["profile"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["view"], "front")
        self.assertGreater(rows[0]["width_px"], 0.0)

    def test_gaussian_signal_includes_surface_constraint_and_topology_details(self) -> None:
        target = _fixture_target()

        signals = collect_target_signals(
            target,
            include_profile_rows=False,
            include_surface_density=True,
            include_constraint_kinds=True,
            include_topology_details=True,
        )

        self.assertIn("density_hint", signals["surface"])
        self.assertEqual(signals["constraints"]["constraint_kinds"], {"scribble": 1})
        self.assertNotIn("rows", signals["profile"])
        self.assertEqual(signals["topology"]["score"], 0.82)
        self.assertEqual(signals["topology"]["detail"], "extras.topology")
        self.assertTrue(signals["uncertainty"]["available"])


def _fixture_target() -> ReconstructionTarget:
    interval = ProfileIntervalPx(10.0, 18.0, confidence=0.9)
    band = ProfileBand(
        t=0.5,
        intervals=(interval,),
        width_px=12.0,
        confidence=0.8,
        source_view="front",
    )
    constraint = _Constraint(
        view="front",
        kind="scribble",
        uncertainty=_Uncertainty(
            confidence=np.asarray([0.7, 0.9]),
            boundary_uncertainty=np.asarray([0.1, 0.3]),
        ),
    )
    return ReconstructionTarget(
        constraints=(constraint,),
        profile_bands={"front": (band,)},
        extras={
            "surface_points": np.zeros((64, 3), dtype=float),
            "constraint_payload": {"constraints": [{"kind": "axis"}]},
            "topology": {"score": 0.82, "holes": 0},
        },
    )


if __name__ == "__main__":
    unittest.main()
