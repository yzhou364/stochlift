"""Mean-CVaR objectives.

The risk-adjusted cost of a decision is ``(1 - weight) * E[f] + weight * CVaR_alpha(f)``,
where ``f`` is the scenario cost in minimization form (for a maximization model,
the loss, so the CVaR looks at the worst profits). ``CVaR_alpha`` is the mean of
the worst ``1 - alpha`` share of outcomes (Rockafellar and Uryasev, 2000):

    CVaR_alpha(f) = min_eta  eta + E[(f - eta)^+] / (1 - alpha)

In the extensive form this adds one free variable ``eta`` and one non-negative
variable per scenario, so the program stays a (mixed-integer) linear program.

``alpha = 1`` is the worst case over the scenarios (the limit of CVaR), written
``measure: worst_case`` in a spec: a scenario-based robust objective. It needs
only one extra variable, ``t >= f_s`` for every scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Risk:
    alpha: float = 0.9        # CVaR level: the worst (1 - alpha) share of outcomes; 1 = worst case
    weight: float = 0.0       # 0 = expected cost only, 1 = CVaR only

    def __post_init__(self):
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("risk.alpha must be in [0, 1]")
        if not 0.0 <= float(self.weight) <= 1.0:
            raise ValueError("risk.weight must be in [0, 1]")

    @property
    def active(self) -> bool:
        return self.weight > 0

    @classmethod
    def from_spec(cls, d) -> "Risk":
        if not d:
            return cls()
        if isinstance(d, Risk):
            return d
        d = dict(d)
        measure = str(d.pop("measure", "cvar")).lower().replace("-", "_")
        if measure not in ("cvar", "worst_case"):
            raise ValueError(f"risk.measure must be 'cvar' or 'worst_case', got {measure!r}")
        unknown = set(d) - ({"alpha", "weight"} if measure == "cvar" else {"weight"})
        if unknown:
            allowed = "measure, alpha, weight" if measure == "cvar" else "measure, weight"
            raise ValueError(f"unknown risk fields {sorted(unknown)}; allowed: {allowed}")
        alpha = 1.0 if measure == "worst_case" else float(d.get("alpha", 0.9))
        return cls(alpha=alpha, weight=float(d.get("weight", 1.0)))

    @property
    def worst_case(self) -> bool:
        return self.alpha >= 1.0

    def tail_label(self) -> str:
        return tail_label(self.alpha)

    def describe(self, sense: str = "min") -> str:
        what = "cost" if sense == "min" else "loss (negative value)"
        if not self.active:
            return f"expected {what}"
        cv = ("the worst case over the scenarios" if self.worst_case else
              f"CVaR at {self.alpha:g} (mean of the worst {100 * (1 - self.alpha):g}% of outcomes)")
        if self.weight == 1:
            return f"{cv} of the {what}"
        return f"{1 - self.weight:g} x expected {what} + {self.weight:g} x {cv}"

    def value(self, costs, probs=None) -> float:
        """Risk-adjusted cost; ``inf`` if any scenario is infeasible (``nan``)."""
        c = np.asarray(costs, dtype=float)
        if np.isnan(c).any():
            return float("inf")
        p = np.full(len(c), 1.0 / len(c)) if probs is None else np.asarray(probs, dtype=float)
        mean = float(p @ c)
        if not self.active:
            return mean
        return (1 - self.weight) * mean + self.weight * cvar(c, p, self.alpha)

    def values_equal_weights(self, C: np.ndarray) -> np.ndarray:
        """Risk-adjusted cost of every row of ``C``, each row an equally likely sample."""
        C = np.asarray(C, dtype=float)
        mean = C.mean(axis=1)
        if not self.active:
            return mean
        if self.worst_case:
            return (1 - self.weight) * mean + self.weight * C.max(axis=1)
        n = C.shape[1]
        S = np.sort(C, axis=1)
        k = max(int(np.ceil(self.alpha * n - 1e-9)) - 1, 0)
        eta = S[:, k]
        cv = eta + np.maximum(C - eta[:, None], 0).mean(axis=1) / (1 - self.alpha)
        return (1 - self.weight) * mean + self.weight * cv


def cvar(costs, probs, alpha: float) -> float:
    """Exact CVaR of a discrete distribution (``eta`` = the alpha-quantile is optimal)."""
    c = np.asarray(costs, dtype=float)
    p = np.asarray(probs, dtype=float)
    if alpha >= 1.0:                                      # the limit: the worst possible outcome
        return float(c[p > 0].max())
    order = np.argsort(c, kind="stable")
    c, p = c[order], p[order]
    cum = np.cumsum(p)
    k = int(np.searchsorted(cum, alpha - 1e-12, side="left"))
    eta = c[min(k, len(c) - 1)]
    return float(eta + p @ np.maximum(c - eta, 0.0) / (1 - alpha))


def tail_label(alpha: float) -> str:
    """'CVaR at 0.9', or 'worst case' for alpha = 1."""
    return "worst case" if alpha >= 1.0 else f"CVaR at {alpha:g}"


def tail_meaning(alpha: float, worst: str = "highest costs") -> str:
    return f"the worst scenario" if alpha >= 1.0 else f"mean of the {100 * (1 - alpha):g}% {worst}"
