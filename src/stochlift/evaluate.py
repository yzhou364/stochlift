"""WS / RP / EEV, and evaluation of a fixed first-stage decision."""
from __future__ import annotations

import numpy as np

from .lift import ExtensiveForm, extensive_form
from .model import LinearModel, solve


class SolveError(RuntimeError):
    pass


def first_stage_values(model: LinearModel, x: np.ndarray, is_first) -> dict:
    out = {}
    for j, nm in enumerate(model.names):
        if is_first(nm):
            out[nm] = float(round(x[j])) if model.integer[j] else float(x[j])
    return out


def solve_recourse_problem(models, probs, is_first, **opts):
    """Solve the extensive form. Returns (objective, first-stage dict, ExtensiveForm)."""
    ef: ExtensiveForm = extensive_form(models, probs, is_first)
    sol = solve(ef.model, **opts)
    if not sol.ok:
        raise SolveError(f"the stochastic program (extensive form) is {sol.status}"
                         f" [{sol.raw_status}]. If it is infeasible, some scenario has no "
                         "feasible recourse for any first-stage decision: add slack/penalty "
                         "variables to the deterministic model.")
    k = len(ef.first_names)
    x = {}
    for j, nm in enumerate(ef.first_names):
        x[nm] = float(round(sol.x[j])) if ef.model.integer[j] else float(sol.x[j])
    assert k == len(x)
    return sol.objective, x, ef


def _default_value(lb: float, ub: float) -> float:
    return float(min(max(0.0, lb), ub))


def complete_first_stage(x_first: dict, models, is_first):
    """Give a value to first-stage variables that ``x_first`` does not cover.

    Returns the completed decision and the names that were missing. The default
    is 0, moved to the nearest bound when 0 is outside the variable's bounds.
    """
    x = dict(x_first)
    bounds = {}
    for m in models:
        for j, nm in enumerate(m.names):
            if nm not in x and is_first(nm):
                lo, hi = bounds.get(nm, (-np.inf, np.inf))
                bounds[nm] = (max(lo, m.lb[j]), min(hi, m.ub[j]))
    for nm, (lo, hi) in bounds.items():
        x[nm] = _default_value(lo, hi)
    return x, list(bounds)


def evaluate_first_stage(models, x_first: dict, is_first=None, **opts) -> np.ndarray:
    """Cost of each scenario when the first stage is fixed to ``x_first``.

    Infeasible scenarios are returned as ``nan``. When ``is_first`` is given,
    every first-stage column is fixed: one that ``x_first`` does not cover is
    set to its default value rather than left free to adapt to the scenario.
    """
    costs = np.full(len(models), np.nan)
    for s, m in enumerate(models):
        fixed = {}
        for j, nm in enumerate(m.names):
            if nm in x_first:
                fixed[j] = x_first[nm]
            elif is_first is not None and is_first(nm):
                fixed[j] = _default_value(m.lb[j], m.ub[j])
        sol = solve(m, fixed=fixed, **opts)
        if sol.ok:
            costs[s] = sol.objective
        elif sol.status != "infeasible":
            raise SolveError(f"scenario {s} with a fixed first stage is {sol.status}"
                             f" [{sol.raw_status}]")
    return costs


def wait_and_see(models, **opts) -> np.ndarray:
    """Optimal objective of each scenario solved on its own (perfect information)."""
    out = np.empty(len(models))
    for s, m in enumerate(models):
        sol = solve(m, **opts)
        if not sol.ok:
            raise SolveError(f"scenario {s} solved on its own is {sol.status} [{sol.raw_status}]")
        out[s] = sol.objective
    return out


def expected(costs: np.ndarray, probs: np.ndarray) -> float:
    """Probability-weighted mean; ``inf`` if any scenario is infeasible."""
    if np.isnan(costs).any():
        return float("inf")
    return float(np.dot(probs, costs))
