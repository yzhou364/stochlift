"""Scenario sets: explicit scenarios, or scenarios built from historical data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import datautil as du


@dataclass
class ScenarioSet:
    paths: list            # leaf paths of the uncertain entries
    values: np.ndarray     # (S, d) realisations
    probs: np.ndarray      # (S,) probabilities, sum to 1

    def __post_init__(self):
        self.values = np.atleast_2d(np.asarray(self.values, dtype=float))
        self.probs = np.asarray(self.probs, dtype=float)
        if self.values.shape != (len(self.probs), len(self.paths)):
            raise ValueError("values must have shape (n_scenarios, n_uncertain_entries)")
        if (self.probs < 0).any() or abs(self.probs.sum() - 1.0) > 1e-9:
            raise ValueError("probabilities must be non-negative and sum to 1")

    def __len__(self) -> int:
        return len(self.probs)

    @property
    def names(self) -> list:
        return [du.leaf_name(p) for p in self.paths]

    def mean(self) -> np.ndarray:
        return self.probs @ self.values

    def data(self, base: dict, s: int) -> dict:
        return du.apply(base, self.paths, self.values[s])

    def mean_data(self, base: dict) -> dict:
        return du.apply(base, self.paths, self.mean())


def explicit(base: dict, scenarios) -> ScenarioSet:
    """Scenarios given by hand.

    ``scenarios`` is a list of ``(probability, overrides)`` where ``overrides``
    is a partial, nested copy of the data dictionary, e.g.
    ``(1/3, {"yield": {"wheat": 3.0, "corn": 3.6, "beets": 24}})``.
    """
    paths = None
    rows, probs = [], []
    for p, overrides in scenarios:
        leaves = du.numeric_leaves(overrides)
        these = [path for path, _ in leaves]
        if paths is None:
            paths = these
            for path in paths:
                du.get_leaf(base, path)  # raises KeyError if the entry is not in the data
        elif these != paths:
            raise ValueError("every scenario must override the same entries in the same order")
        rows.append([v for _, v in leaves])
        probs.append(p)
    return ScenarioSet(paths, np.array(rows), np.array(probs))


def match_columns(paths, columns, mapping=None) -> list:
    """Find the history column for every uncertain leaf.

    Order of preference: explicit ``mapping`` (leaf name -> column), a column
    named like the leaf (``demand.c1``), then a column named like the last
    path component (``c1``) when that is unambiguous.
    """
    mapping = dict(mapping or {})
    original = {str(c): c for c in columns}
    columns = [str(c) for c in columns]
    last = [str(p[-1]) for p in paths]
    out = []
    for p in paths:
        name = du.leaf_name(p)
        if name in mapping:
            col = str(mapping[name])
        elif name in columns:
            col = name
        elif str(p[-1]) in columns and last.count(str(p[-1])) == 1:
            col = str(p[-1])
        else:
            raise KeyError(f"no history column for uncertain entry '{name}'. "
                           f"Columns: {columns}. Add it under 'columns:' in the spec.")
        if col not in columns:
            raise KeyError(f"history has no column '{col}' (mapped from '{name}')")
        out.append(original[col])
    return out


def from_history(history: np.ndarray, paths, method: str = "empirical", n=None,
                 seed: int = 0) -> ScenarioSet:
    """Build scenarios from observations (one row per observation).

    - ``empirical``: every observation is one equally likely scenario.
    - ``sample``: ``n`` observations drawn with replacement, equally likely.
    - ``kmeans``: ``n`` cluster centroids, weighted by cluster size. Centroids
      preserve the mean exactly but shrink the tails; compare against
      ``empirical`` on the hold-out set before trusting a small ``n``.
    """
    X = np.asarray(history, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(paths):
        raise ValueError("history must have one column per uncertain entry")
    if np.isnan(X).any():
        raise ValueError("history contains missing values; clean or impute them first")
    N = len(X)
    rng = np.random.default_rng(seed)
    if method == "empirical" or (n is not None and n >= N and method == "kmeans"):
        return ScenarioSet(list(paths), X, np.full(N, 1.0 / N))
    if n is None:
        raise ValueError(f"method '{method}' needs the number of scenarios n")
    if method == "sample":
        idx = rng.integers(0, N, size=n)
        return ScenarioSet(list(paths), X[idx], np.full(n, 1.0 / n))
    if method == "kmeans":
        from scipy.cluster.vq import kmeans2

        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Z = (X - X.mean(axis=0)) / scale
        _, labels = kmeans2(Z, n, minit="++", seed=int(rng.integers(0, 2**31 - 1)))
        keep = [k for k in range(n) if (labels == k).any()]
        values = np.array([X[labels == k].mean(axis=0) for k in keep])
        probs = np.array([(labels == k).mean() for k in keep])
        return ScenarioSet(list(paths), values, probs / probs.sum())
    raise ValueError(f"unknown scenario method '{method}' (use empirical, sample or kmeans)")
