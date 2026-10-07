"""Bounded coupled part updates through the existing dimensionless adapters."""
from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
from .resfit_optimizer import (OptimizationBudget, OptimizationBudgetExhausted,
                              OptimizationRecord, OptimizationResult, clone_primitives,
                              ordered_parameter_refs)
from .resfit_parameters import apply_parameter_increment


def coupled_block_optimize(primitives, objective, config, *, budget=None,
                           priority_parts=(), on_progress=None, parameter_filter=None):
    """Trust-region pose/size/shape blocks with complete-objective retention.

    Coordinate descent remains the explicit fallback for nonsmooth structural
    choices. Numeric Jacobian calls share the existing evaluation/deadline budget.
    """
    from scipy.optimize import least_squares
    errors = config.validate()
    if errors:
        raise ValueError("invalid optimizer config: " + "; ".join(errors))
    if not hasattr(objective, "residual_vector"):
        raise TypeError("coupled refinement requires the declared objective residual vector")
    started = time.perf_counter()
    allowance = OptimizationBudget(config.max_objective_evaluations,config.max_elapsed_s,parent=budget)
    working = clone_primitives(primitives)
    initial = allowance.evaluate(objective, working); current = initial
    refs = ordered_parameter_refs(working, priority_parts=priority_parts)
    if parameter_filter is not None:
        refs = [ref for ref in refs if parameter_filter(ref)]
    order = list(dict.fromkeys(ref[0] for ref in refs))
    groups = {i:[ref for ref in refs if ref[0] == i] for i in order}
    history, visits = [], []
    termination = "zero_iterations" if config.iterations == 0 else "max_iterations"
    for iteration in range(config.iterations):
        improved = 0
        for index in order:
            reason = allowance.reason()
            if reason is not None:
                termination = reason
                break
            block = groups[index]
            base = deepcopy(working)
            best_parts, best_result = working, current

            def residual(increment):
                nonlocal best_parts, best_result
                proposal = deepcopy(base)
                for ref, value in zip(block, increment):
                    if value != 0.:
                        visits.append(ref)
                    apply_parameter_increment(proposal, ref, float(value), config.bounds)
                result = allowance.evaluate(objective, proposal)
                vector = objective.residual_vector(proposal, evaluated=result)
                if result.total < best_result.total:
                    best_parts, best_result = proposal, result
                    if on_progress is not None:
                        on_progress(OptimizationResult(tuple(clone_primitives(best_parts)),tuple(history),
                            float(result.total),"scored_checkpoint",allowance.objective_evaluations,
                            time.perf_counter()-started,initial,result,tuple(visits)))
                return vector

            radius = max(config.min_step,config.initial_step)
            try:
                least_squares(residual,np.zeros(len(block)),bounds=(-radius,radius),
                              x_scale="jac",max_nfev=3,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            except OptimizationBudgetExhausted as exc:
                termination = str(exc)
            if best_result.total < current.total:
                improved += 1
                working,current = best_parts,best_result
            if allowance.reason() is not None:
                break
        history.append(OptimizationRecord(iteration,float(current.total),dict(current.terms),improved,
                                          config.initial_step,reason="coupled_part_block"))
        if allowance.reason() is not None:
            termination = allowance.reason()
            break
        if not improved:
            termination = "no_coupled_block_improved"
            break
    return OptimizationResult(tuple(working),tuple(history),float(current.total),termination,
                              allowance.objective_evaluations,time.perf_counter()-started,
                              initial,current,tuple(visits))
