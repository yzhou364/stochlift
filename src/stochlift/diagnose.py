"""Explain why a stochastic program has no solution, in terms of the user's data."""
from __future__ import annotations

import numpy as np

from .evaluate import SolveError
from .lift import extensive_form
from .model import solve

ADVICE = ("Typical fixes: allow shortfalls with slack variables at a penalty cost (unmet demand, "
          "overtime, outsourcing) in the deterministic model, or narrow the distributions "
          "(min/max under scenarios.distributions).")


class ScenarioInfeasible(SolveError):
    """Some scenarios have no solution even with perfect information."""

    def __init__(self, message, scenarios):
        super().__init__(message)
        self.scenarios = scenarios


class NoCommonFirstStage(SolveError):
    """Every scenario is feasible on its own, but no first-stage decision suits them all."""

    def __init__(self, message, scenarios):
        super().__init__(message)
        self.scenarios = scenarios


def unusual_entries(study, s: int, k: int = 3) -> str:
    """The uncertain entries of scenario ``s`` that are furthest from their mean."""
    V = study.scenarios.values
    if V.shape[1] == 0:
        return ""
    mu, sd = study.scenarios.mean(), V.std(axis=0)
    z = np.abs(V[s] - mu) / np.where(sd > 0, sd, 1.0)
    top = np.argsort(-z)[:k]
    names = study.scenarios.names
    return ", ".join(f"{names[j]} = {V[s, j]:.4g} (mean {mu[j]:.4g})" for j in top)


def scenarios_alone(study, statuses) -> ScenarioInfeasible:
    bad = [s for s, st in enumerate(statuses) if st != "optimal"]
    kinds = sorted({statuses[s] for s in bad})
    lines = [f"{len(bad)} of {len(statuses)} scenarios have no optimal solution even when solved on "
             f"their own ({', '.join(kinds)}), so no stochastic program over them can be solved."]
    for s in bad[:3]:
        lines.append(f"  scenario {s}: {unusual_entries(study, s)}")
    if "unbounded" in kinds:
        lines.append("An unbounded scenario means a variable can grow without limit for that data; "
                     "add a bound or a cost to it.")
    if "infeasible" in kinds:
        lines.append("The deterministic model itself has no solution for these data. " + ADVICE)
    return ScenarioInfeasible("\n".join(lines), bad)


def common_first_stage(study, models, max_scenarios: int = 200) -> NoCommonFirstStage:
    """Add scenarios one at a time until the extensive form becomes infeasible, then look
    for a single earlier scenario that conflicts with the one that broke it."""
    S = len(models)
    order = list(range(S)) if S <= max_scenarios else \
        list(np.random.default_rng(0).choice(S, max_scenarios, replace=False))
    opts = {k: v for k, v in study.opts.items() if k in ("solver", "time_limit")}

    def feasible(idx):
        ef = extensive_form([models[i] for i in idx], np.full(len(idx), 1.0 / len(idx)), study.is_first)
        return solve(ef.model, **opts).status != "infeasible"

    kept, culprit = [], None
    for s in order:
        if feasible(kept + [s]):
            kept.append(s)
        else:
            culprit = s
            break
    head = ("Every scenario has a solution on its own, but no single first-stage decision is "
            "feasible in all of them (the recourse is not complete).")
    if culprit is None:
        return NoCommonFirstStage(head + " The conflict could not be isolated.", [])
    partner = next((t for t in kept if not feasible([t, culprit])), None)
    if partner is not None:
        msg = (f"{head}\nScenarios {partner} and {culprit} already need different first-stage decisions:\n"
               f"  scenario {partner}: {unusual_entries(study, partner)}\n"
               f"  scenario {culprit}: {unusual_entries(study, culprit)}\n"
               "Often a constraint on the first stage alone, or an equality that recourse variables "
               "cannot absorb, depends on the uncertain data. " + ADVICE)
        return NoCommonFirstStage(msg, [partner, culprit])
    msg = (f"{head}\nScenario {culprit} ({unusual_entries(study, culprit)}) cannot share a first stage "
           f"with scenarios {kept[:8]}{' ...' if len(kept) > 8 else ''} together. " + ADVICE)
    return NoCommonFirstStage(msg, kept + [culprit])
