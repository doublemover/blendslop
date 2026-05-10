"""Experiment plan construction for refinement runs."""

from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentVariant,
    ParameterSpec,
    safe_slug,
    stable_hash,
)
from .preset_catalog import SuitePreset, TrackPreset, get_suite_preset, get_track_preset

try:
    from blender_blocking.synthetic.quality_targets import quality_targets_for
    from blender_blocking.synthetic.registry import specs_for_suite
except ImportError:  # pragma: no cover - direct blender_blocking/ execution
    from synthetic.quality_targets import quality_targets_for
    from synthetic.registry import specs_for_suite


def build_experiment_plan(
    *,
    suite: str,
    track: str,
    search: str,
    objective: str,
    output_root: Path,
    seed: int = 0,
    modes: Sequence[str] | None = None,
    max_runs: int | None = None,
    top_k: int = 10,
    custom_reference_paths: Mapping[str, Path] | None = None,
    external_variants: Sequence[ExperimentVariant | Mapping[str, Any]] | None = None,
    external_variant_mode: str = "append",
) -> ExperimentPlan:
    suite_preset = get_suite_preset(suite)
    track_preset = get_track_preset(track)
    run_id = _run_id(suite, track)
    cases = resolve_suite_cases(
        suite_preset,
        seed=seed or suite_preset.default_seed,
        count=suite_preset.default_count,
        custom_reference_paths=custom_reference_paths,
    )
    effective_modes = tuple(modes or track_preset.modes)
    effective_max_runs = (
        max_runs if max_runs is not None else track_preset.max_runs_hint
    )
    variants = build_variants_for_track(
        track_preset,
        search=search,
        modes=effective_modes,
        seed=seed,
        max_runs=effective_max_runs,
    )
    variants = combine_variants(
        variants,
        _coerce_variants(external_variants or ()),
        mode=external_variant_mode,
    )
    plan_id = stable_hash(
        {
            "suite": suite,
            "track": track,
            "search": search,
            "objective": objective,
            "seed": seed,
            "modes": effective_modes,
            "max_runs": effective_max_runs,
            "variants": [variant.variant_hash() for variant in variants],
        }
    )
    return ExperimentPlan(
        plan_id=f"{safe_slug(suite)}-{safe_slug(track)}-{plan_id}",
        suite=suite,
        track=track,
        search=search,
        objective=objective,
        output_root=Path(output_root),
        run_id=run_id,
        cases=cases,
        variants=variants,
        max_runs=effective_max_runs,
        top_k=top_k,
        seed=seed,
        metadata={
            "suite": suite_preset.to_dict(),
            "track": track_preset.to_dict(),
            "requested_modes": list(effective_modes),
            "external_variant_mode": external_variant_mode,
            "external_variant_count": len(tuple(external_variants or ())),
        },
    )


def resolve_suite_cases(
    suite: SuitePreset,
    *,
    seed: int,
    count: int | None = None,
    custom_reference_paths: Mapping[str, Path] | None = None,
) -> tuple[ExperimentCase, ...]:
    if custom_reference_paths:
        return (
            ExperimentCase(
                case_id="custom-images",
                suite=suite.name,
                source="custom_images",
                reference_paths={
                    key: Path(value) for key, value in custom_reference_paths.items()
                },
                metadata={"suite_preset": suite.to_dict()},
            ),
        )

    if suite.source == "builtin_sample":
        return (
            ExperimentCase(
                case_id=suite.name,
                suite=suite.name,
                source="builtin_sample",
                reference_paths=suite.reference_paths,
                known_ambiguity_notes=(
                    "built-in sample image; use synthetic suites for exact ground truth",
                ),
                metadata={"suite_preset": suite.to_dict()},
            ),
        )

    cases: list[ExperimentCase] = []
    suite_count = count
    for suite_name in suite.synthetic_suites:
        for spec in specs_for_suite(
            suite_name, seed=seed + len(cases), count=suite_count
        ):
            definition = _definition_name_from_spec(spec)
            case_id = safe_slug(f"{suite_name}-{getattr(spec, 'shape_id', definition)}")
            cases.append(
                ExperimentCase(
                    case_id=case_id,
                    suite=suite.name,
                    source="synthetic",
                    synthetic_shape_id=str(getattr(spec, "shape_id", case_id)),
                    synthetic_definition=definition,
                    expected_targets=quality_targets_for(spec),
                    known_ambiguity_notes=tuple(
                        getattr(spec, "expected_failure_modes", ())
                    ),
                    metadata={
                        "synthetic_suite": suite_name,
                        "spec": spec.to_dict() if hasattr(spec, "to_dict") else {},
                    },
                )
            )
    return tuple(cases)


def build_variants_for_track(
    track: TrackPreset,
    *,
    search: str,
    modes: Sequence[str],
    seed: int = 0,
    max_runs: int | None = None,
) -> tuple[ExperimentVariant, ...]:
    if search == "grid":
        variants = _grid_variants(track, modes)
    elif search == "random":
        variants = _random_variants(track, modes, seed=seed, max_runs=max_runs or 32)
    elif search == "coordinate":
        variants = _coordinate_variants(track, modes)
    elif search == "successive_halving":
        variants = _successive_halving_variants(track, modes)
    else:
        raise ValueError(f"unknown refinement search strategy: {search}")
    if max_runs is not None:
        variants = variants[: max(1, int(max_runs))]
    return tuple(variants)


def load_variants(path: Path) -> tuple[ExperimentVariant, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return variants_from_payload(payload)


def load_variants_from_files(paths: Sequence[Path]) -> tuple[ExperimentVariant, ...]:
    variants: list[ExperimentVariant] = []
    for path in paths:
        variants.extend(load_variants(path))
    return tuple(variants)


def variants_from_payload(payload: Any) -> tuple[ExperimentVariant, ...]:
    if isinstance(payload, Mapping):
        if "variants" in payload:
            return variants_from_payload(payload["variants"])
        if "variant_id" in payload:
            return (ExperimentVariant.from_dict(payload),)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _coerce_variants(payload)
    raise ValueError("variant payload must contain a variant or variants list")


def combine_variants(
    base_variants: Sequence[ExperimentVariant],
    external_variants: Sequence[ExperimentVariant],
    *,
    mode: str = "append",
) -> tuple[ExperimentVariant, ...]:
    if not external_variants:
        return tuple(base_variants)
    if mode == "append":
        return tuple(_dedupe_variants(tuple(base_variants) + tuple(external_variants)))
    if mode == "replace":
        return tuple(_dedupe_variants(tuple(external_variants)))
    raise ValueError(f"unknown external variant mode: {mode}")


def write_plan(plan: ExperimentPlan, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else plan.output_root / "plan.json"
    return plan.write(target)


def read_plan(path: Path) -> ExperimentPlan:
    return ExperimentPlan.read(path)


def _grid_variants(track: TrackPreset, modes: Sequence[str]) -> list[ExperimentVariant]:
    variants = [_baseline_variant(track, mode) for mode in modes]
    parameters = track.parameters
    if not parameters:
        return variants
    value_lists = [parameter.values or (parameter.default,) for parameter in parameters]
    for mode in modes:
        for index, combination in enumerate(itertools.product(*value_lists)):
            parameter_values = {
                parameter.name: value
                for parameter, value in zip(parameters, combination)
            }
            variants.append(
                _variant_from_values(
                    track,
                    mode,
                    parameter_values,
                    label_prefix=f"{mode}-grid-{index:04d}",
                    stage="grid",
                )
            )
    return _dedupe_variants(variants)


def _random_variants(
    track: TrackPreset,
    modes: Sequence[str],
    *,
    seed: int,
    max_runs: int,
) -> list[ExperimentVariant]:
    rng = random.Random(seed)
    variants = [_baseline_variant(track, mode) for mode in modes]
    attempts = 0
    target_count = max(max_runs, len(variants))
    seen = {variant.variant_hash() for variant in variants}
    while len(variants) < target_count and attempts < target_count * 20:
        attempts += 1
        for mode in modes:
            parameter_values = {}
            for parameter in track.parameters:
                values = parameter.values or parameter.choices
                if values:
                    parameter_values[parameter.name] = rng.choice(tuple(values))
                elif (
                    parameter.min_value is not None and parameter.max_value is not None
                ):
                    parameter_values[parameter.name] = _sample_range(parameter, rng)
            variant = _variant_from_values(
                track,
                mode,
                parameter_values,
                label_prefix=f"{mode}-random-{len(variants):04d}",
                stage="random",
            )
            digest = variant.variant_hash()
            if digest in seen:
                continue
            seen.add(digest)
            variants.append(variant)
            if len(variants) >= target_count:
                break
    return variants


def _coordinate_variants(
    track: TrackPreset, modes: Sequence[str]
) -> list[ExperimentVariant]:
    variants = [_baseline_variant(track, mode) for mode in modes]
    for mode in modes:
        parent = f"{mode}-baseline"
        for parameter in track.parameters:
            for value_index, value in enumerate(
                parameter.values or (parameter.default,)
            ):
                variants.append(
                    _variant_from_values(
                        track,
                        mode,
                        {parameter.name: value},
                        label_prefix=f"{mode}-{safe_slug(parameter.name)}-{value_index:02d}",
                        stage=parameter.group or "coordinate",
                        parent_variant_id=parent,
                    )
                )
    return _dedupe_variants(variants)


def _successive_halving_variants(
    track: TrackPreset,
    modes: Sequence[str],
) -> list[ExperimentVariant]:
    variants = []
    for variant in _grid_variants(track, modes):
        stage = "halving-stage-0" if variant.stage != "baseline" else "baseline"
        variants.append(
            ExperimentVariant(
                variant_id=variant.variant_id,
                label=variant.label,
                mode=variant.mode,
                validation_mode=variant.validation_mode,
                parameters=variant.parameters,
                cli_args=variant.cli_args,
                config_overrides=variant.config_overrides,
                expected_artifacts=variant.expected_artifacts,
                tags=variant.tags + ("successive_halving",),
                parent_variant_id=variant.parent_variant_id,
                stage=stage,
                diagnostic_only=variant.diagnostic_only,
            )
        )
    return variants


def _baseline_variant(track: TrackPreset, mode: str) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=f"{safe_slug(mode)}-baseline",
        label=f"{mode} baseline",
        mode=mode,
        validation_mode=track.default_validation_mode,
        parameters={},
        cli_args=(
            "--reconstruction-mode",
            mode,
            "--validation-mode",
            track.default_validation_mode,
        ),
        tags=("baseline",) + track.tags,
        stage="baseline",
    )


def _variant_from_values(
    track: TrackPreset,
    mode: str,
    values: Mapping[str, Any],
    *,
    label_prefix: str,
    stage: str,
    parent_variant_id: str = "",
) -> ExperimentVariant:
    cli_args: list[str] = [
        "--reconstruction-mode",
        mode,
        "--validation-mode",
        track.default_validation_mode,
    ]
    parameter_map = dict(values)
    by_name = {parameter.name: parameter for parameter in track.parameters}
    for name in sorted(parameter_map):
        parameter = by_name[name]
        cli_args.extend(parameter.cli_tokens_for_value(parameter_map[name]))
    digest = stable_hash(
        {"mode": mode, "parameters": parameter_map, "track": track.name}, length=8
    )
    variant_id = safe_slug(f"{label_prefix}-{digest}")
    return ExperimentVariant(
        variant_id=variant_id,
        label=label_prefix,
        mode=mode,
        validation_mode=track.default_validation_mode,
        parameters=parameter_map,
        cli_args=tuple(cli_args),
        tags=track.tags,
        stage=stage,
        parent_variant_id=parent_variant_id,
    )


def _dedupe_variants(variants: Sequence[ExperimentVariant]) -> list[ExperimentVariant]:
    deduped: list[ExperimentVariant] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    for variant in variants:
        digest = variant.variant_hash()
        if digest in seen_hashes:
            continue
        variant_id = variant.variant_id
        if variant_id in seen_ids:
            variant_id = f"{variant_id}-{digest[:6]}"
            variant = ExperimentVariant(
                variant_id=variant_id,
                label=variant.label,
                mode=variant.mode,
                validation_mode=variant.validation_mode,
                parameters=variant.parameters,
                cli_args=variant.cli_args,
                config_overrides=variant.config_overrides,
                expected_artifacts=variant.expected_artifacts,
                tags=variant.tags,
                parent_variant_id=variant.parent_variant_id,
                stage=variant.stage,
                diagnostic_only=variant.diagnostic_only,
            )
        seen_hashes.add(digest)
        seen_ids.add(variant.variant_id)
        deduped.append(variant)
    return deduped


def _coerce_variants(
    variants: Sequence[ExperimentVariant | Mapping[str, Any]],
) -> tuple[ExperimentVariant, ...]:
    return tuple(
        variant
        if isinstance(variant, ExperimentVariant)
        else ExperimentVariant.from_dict(variant)
        for variant in variants
    )


def _sample_range(parameter: ParameterSpec, rng: random.Random) -> int | float:
    assert parameter.min_value is not None and parameter.max_value is not None
    if parameter.scale == "log":
        import math

        low = math.log(parameter.min_value)
        high = math.log(parameter.max_value)
        value = math.exp(rng.uniform(low, high))
    else:
        value = rng.uniform(parameter.min_value, parameter.max_value)
    if parameter.value_type == "int":
        return int(round(value))
    return float(value)


def _definition_name_from_spec(spec: object) -> str:
    parameters = getattr(spec, "parameters", {})
    for key in (
        "primitive",
        "profile_kind",
        "blockout_kind",
        "mask_kind",
        "degradation",
    ):
        if key in parameters:
            return str(parameters[key])
    return safe_slug(str(getattr(spec, "shape_id", "synthetic")))


def _run_id(suite: str, track: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{safe_slug(suite)}-{safe_slug(track)}"
