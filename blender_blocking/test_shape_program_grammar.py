"""Tests for editable shape-program grammar, DSL, and search."""

from __future__ import annotations

import unittest

from primitives.grammar import default_shape_program_grammar, validate_grammar
from primitives.program_search import search_shape_program_candidates
from primitives.shape_dsl import program_from_dsl, program_to_dsl
from primitives.shape_program import ShapeNode, ShapeProgram, validate_shape_program
from reconstruction.backends.shape_program import build_shape_program_from_target
from reconstruction.types import (
    Bounds3D,
    ProfileBand,
    ProfileIntervalPx,
    ReconstructionTarget,
)


def _seed_program() -> ShapeProgram:
    return ShapeProgram(
        schema_version="shape-program-v1",
        program_id="seed",
        root_nodes=(
            ShapeNode(
                node_id="root_profile_00",
                operation="add",
                primitive_type="lathe_profile",
                name="front profile lathe",
                parameters={
                    "profile_curve": [{"t": 0.0, "left": -0.5, "right": 0.5}],
                    "width_world": 1.0,
                    "depth_world": 0.8,
                    "height_world": 1.2,
                    "band_count": 3,
                    "preserves_multiple_intervals": True,
                },
            ),
        ),
        metadata={"bounds": {"min_x": -0.5, "max_x": 0.5}},
    )


class ShapeProgramGrammarTests(unittest.TestCase):
    def test_default_grammar_validates(self) -> None:
        grammar = default_shape_program_grammar()

        self.assertEqual(validate_grammar(grammar), ())
        self.assertIn("superquadric_proxy_root", grammar.rule_ids())

    def test_shape_dsl_round_trips_program(self) -> None:
        program = _seed_program()
        dsl = program_to_dsl(program)
        restored = program_from_dsl(dsl)

        self.assertEqual(restored.to_dict(), program.to_dict())
        self.assertEqual(validate_shape_program(restored), ())

    def test_program_search_returns_valid_ranked_candidates(self) -> None:
        result = search_shape_program_candidates(
            _seed_program(),
            max_candidates=3,
            objective="part_aware",
        )

        self.assertGreaterEqual(len(result.candidates), 2)
        self.assertEqual(result.selected, result.candidates[0])
        self.assertEqual(validate_shape_program(result.selected.program), ())
        self.assertIn("dsl", result.selected.to_dict())

    def test_backend_records_grammar_search_and_dsl(self) -> None:
        bands = (
            ProfileBand(
                t=0.0,
                intervals=(
                    ProfileIntervalPx(10.0, 30.0),
                    ProfileIntervalPx(40.0, 55.0),
                ),
                center_x=32.0,
                width_px=45.0,
                holes=(ProfileIntervalPx(31.0, 39.0),),
                confidence=0.8,
                source_view="front",
            ),
            ProfileBand(
                t=1.0,
                intervals=(ProfileIntervalPx(12.0, 50.0),),
                center_x=31.0,
                width_px=38.0,
                confidence=0.9,
                source_view="front",
            ),
        )
        target = ReconstructionTarget(
            profile_bands={"front": bands},
            bounds=Bounds3D(-0.5, 0.5, -0.4, 0.4, 0.0, 1.2),
        )

        program, diagnostics = build_shape_program_from_target(
            target,
            config={
                "root_strategy": "hybrid_profile_bounds",
                "residual_policy": "suggest_patches",
                "program_search_candidates": 3,
                "program_search_objective": "part_aware",
            },
            program_id="backend-program",
        )

        self.assertIn("grammar_search", diagnostics)
        self.assertIn("program_dsl", program.metadata)
        self.assertEqual(validate_shape_program(program), ())
        self.assertGreaterEqual(
            diagnostics["grammar_search"]["candidate_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
