"""Mean-CVaR objectives.

The risk-adjusted cost of a decision is ``(1 - weight) * E[f] + weight * CVaR_alpha(f)``,
where ``f`` is the scenario cost in minimization form (for a maximization model,
the loss, so the CVaR looks at the worst profits). ``CVaR_alpha`` is the mean of
the worst ``1 - alpha`` share of outcomes (Rockafellar and Uryasev, 2000):

    CVaR_alpha(f) = min_eta  eta + E[(f - eta)^+] / (1 - alpha)

In the extensive form this adds one free variable ``eta`` and one non-negative
variable per scenario, so the program stays a (mixed-integer) linear program.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Risk:
    alpha: float = 0.9        # CVaR level: the worst (1 - alpha) share of outcomes
    weight: float = 0.0       # 0 = expected cost only, 1 = CVaR only

    def __post_init__(self):
        if not 0.0 <= float(self.alpha) < 1.0:
            raise ValueError("risk.alpha must be in [0, 1)")
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
        measure = str(d.pop("measure", "cvar")).lower()
        if measure != "cvar":
            raise ValueError(f"risk.measure must be 'cvar', got {measure!r}")
        unknown = set(d) - {"alpha", "weight"}
        if unknown:
            raise ValueError(f"unknown risk fields {sorted(unknown)}; allowed: measure, alpha, weight")
        return cls(alpha=float(d.get("alpha", 0.9)), weight=float(d.get("weight", 1.0)))

    def describe(self, sense: str = "min") -> str:
        what = "cost" if sense == "min" else "loss (negative value)"
        if not self.active:
            return f"expected {what}"
        cv = f"CVaR at {self.alpha:g} (mean of the worst {100 * (1 - self.alpha):g}% of outcomes)"
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
    order = np.argsort(c, kind="stable")
    c, p = c[order], p[order]
    cum = np.cumsum(p)
    k = int(np.searchsorted(cum, alpha - 1e-12, side="left"))
    eta = c[min(k, len(c) - 1)]
    return float(eta + p @ np.maximum(c - eta, 0.0) / (1 - alpha))
