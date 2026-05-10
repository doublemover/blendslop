"""Registry for ambitious reconstruction research experiments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .contracts import MoonshotExperiment, MoonshotRequest, MoonshotResult
from . import (
    active_view_planning,
    differentiable_primitives,
    editable_retopology,
    human_constraint_learning,
    implicit_sdf_proxy,
    shape_grammar_search,
)


_REGISTRY: "OrderedDict[str, MoonshotExperiment]" = OrderedDict()


def register(experiment: MoonshotExperiment) -> MoonshotExperiment:
    if experiment.experiment_id in _REGISTRY:
        raise ValueError(f"duplicate moonshot experiment: {experiment.experiment_id}")
    _REGISTRY[experiment.experiment_id] = experiment
    return experiment


def register_builtins() -> None:
    for module in (
        shape_grammar_search,
        active_view_planning,
        differentiable_primitives,
        implicit_sdf_proxy,
        editable_retopology,
        human_constraint_learning,
    ):
        experiment = module.EXPERIMENT
        if experiment.experiment_id not in _REGISTRY:
            register(
                MoonshotExperiment(
                    **{
                        **experiment.to_dict(),
                        "required_inputs": tuple(experiment.required_inputs),
                        "optional_dependencies": tuple(experiment.optional_dependencies),
                        "validation_metrics": tuple(experiment.validation_metrics),
                        "papers": tuple(experiment.papers),
                        "runner": module.run,
                    }
                )
            )


def list_experiments() -> tuple[MoonshotExperiment, ...]:
    register_builtins()
    return tuple(_REGISTRY.values())


def get_experiment(experiment_id: str) -> MoonshotExperiment:
    register_builtins()
    try:
        return _REGISTRY[experiment_id]
    except KeyError as exc:
        raise KeyError(f"unknown moonshot experiment: {experiment_id}") from exc


def run_experiment(request: MoonshotRequest) -> MoonshotResult:
    experiment = get_experiment(request.experiment_id)
    if experiment.runner is None:
        return MoonshotResult(
            experiment_id=request.experiment_id,
            status="research_candidate",
            warnings=("moonshot has no runner registered",),
        )
    return experiment.runner(request)


def experiment_payloads(experiments: Iterable[MoonshotExperiment] | None = None) -> tuple[dict[str, object], ...]:
    return tuple(experiment.to_dict() for experiment in (experiments or list_experiments()))
