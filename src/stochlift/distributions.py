"""Scenarios sampled from distributions, for when there is no history of observations.

Each uncertain entry gets a marginal distribution. Parameters are relative to the
entry's value in the data (its *nominal* value) unless given in absolute terms, so
one line covers a whole group of entries::

    distributions:
      yield: {dist: normal, cv: 0.15}               # every entry under 'yield'
      yield.beets: {dist: uniform, spread: 0.3}      # a single entry overrides its group

Supported marginals (``v`` is the nominal value):

- ``normal``: ``mean`` (default ``v``), and ``sd`` or ``cv`` (sd = cv * |mean|)
- ``lognormal``: ``mean`` (default ``v``, must be > 0) and ``cv``
- ``uniform``: ``low`` and ``high``, or ``spread`` (``v * (1 - spread)`` to ``v * (1 + spread)``)
- ``triangular``: ``low``, ``mode`` (default ``v``) and ``high``, or ``spread``
- ``discrete``: ``values`` (absolute) or ``factors`` (multiples of ``v``), optional ``probs``

Any marginal accepts ``min`` and ``max`` to clip samples, e.g. ``min: 0`` for demand.
Dependence between entries is a Gaussian copula with one common ``correlation``
(between -1/(d-1) and 1); the default 0 samples the entries independently.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import datautil as du

KNOWN = {"normal", "lognormal", "uniform", "triangular", "discrete"}
_PARAMS = {"dist", "mean", "sd", "cv", "low", "high", "mode", "spread", "values", "factors",
           "probs", "min", "max"}


class Marginal:
    """A one-dimensional distribution with an inverse CDF."""

    def __init__(self, name: str, ppf, mean: float, lo: float = -np.inf, hi: float = np.inf):
        self.name, self._ppf, self.mean, self.lo, self.hi = name, ppf, mean, lo, hi

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return np.clip(self._ppf(u), self.lo, self.hi)


def marginal(params: dict, nominal: float, label: str = "") -> Marginal:
    """Build the marginal of one entry from its parameters and nominal value."""
    if not isinstance(params, dict):
        raise ValueError(f"{label}: distribution must be a mapping such as {{dist: normal, cv: 0.1}}")
    p = dict(params)
    unknown = set(p) - _PARAMS
    if unknown:
        raise ValueError(f"{label}: unknown distribution parameters {sorted(unknown)}; "
                         f"allowed: {sorted(_PARAMS)}")
    kind = str(p.get("dist", "")).lower()
    if kind not in KNOWN:
        raise ValueError(f"{label}: 'dist' must be one of {sorted(KNOWN)}, got {p.get('dist')!r}")
    v = float(nominal)
    lo = float(p.get("min", -np.inf))
    hi = float(p.get("max", np.inf))
    if lo > hi:
        raise ValueError(f"{label}: min > max")

    def need(*names):
        for n in names:
            if n not in p:
                raise ValueError(f"{label}: {kind} distribution needs '{n}'")

    def spread_bounds():
        if "spread" in p:
            s = float(p["spread"])
            if s < 0:
                raise ValueError(f"{label}: spread must be >= 0")
            a, b = v - s * abs(v), v + s * abs(v)
            return a, b
        need("low", "high")
        return float(p["low"]), float(p["high"])

    if kind == "normal":
        mu = float(p.get("mean", v))
        if "sd" in p:
            sd = float(p["sd"])
        elif "cv" in p:
            sd = float(p["cv"]) * abs(mu)
        else:
            raise ValueError(f"{label}: normal distribution needs 'sd' or 'cv'")
        if sd < 0:
            raise ValueError(f"{label}: sd must be >= 0")
        if sd == 0:
            return Marginal(kind, lambda u: np.full_like(u, mu), mu, lo, hi)
        return Marginal(kind, lambda u: stats.norm.ppf(u, loc=mu, scale=sd), mu, lo, hi)

    if kind == "lognormal":
        mu = float(p.get("mean", v))
        need("cv")
        cv = float(p["cv"])
        if mu <= 0 or cv < 0:
            raise ValueError(f"{label}: lognormal needs mean > 0 and cv >= 0 (mean is {mu})")
        s2 = np.log1p(cv ** 2)
        m = np.log(mu) - s2 / 2
        return Marginal(kind, lambda u: stats.lognorm.ppf(u, s=np.sqrt(s2), scale=np.exp(m)), mu, lo, hi)

    if kind == "uniform":
        a, b = spread_bounds()
        if b < a:
            raise ValueError(f"{label}: high < low")
        return Marginal(kind, lambda u: a + (b - a) * u, (a + b) / 2, lo, hi)

    if kind == "triangular":
        a, b = spread_bounds()
        c = float(p.get("mode", v))
        if not a <= c <= b:
            raise ValueError(f"{label}: triangular needs low <= mode <= high ({a}, {c}, {b})")
        if b == a:
            return Marginal(kind, lambda u: np.full_like(u, a), a, lo, hi)
        return Marginal(kind, lambda u: stats.triang.ppf(u, (c - a) / (b - a), loc=a, scale=b - a),
                        (a + b + c) / 3, lo, hi)

    # discrete
    if "values" in p:
        vals = np.asarray(p["values"], dtype=float)
    elif "factors" in p:
        vals = v * np.asarray(p["factors"], dtype=float)
    else:
        raise ValueError(f"{label}: discrete distribution needs 'values' or 'factors'")
    probs = np.asarray(p.get("probs", np.full(len(vals), 1.0 / max(len(vals), 1))), dtype=float)
    if len(vals) == 0 or len(probs) != len(vals) or (probs < 0).any() or abs(probs.sum() - 1) > 1e-9:
        raise ValueError(f"{label}: discrete needs as many probs as values, non-negative, summing to 1")
    order = np.argsort(vals)
    vals, probs = vals[order], probs[order]
    cum = np.cumsum(probs)
    cum[-1] = 1.0
    return Marginal(kind, lambda u: vals[np.minimum(np.searchsorted(cum, u, side="right"), len(vals) - 1)],
                    float(probs @ vals), lo, hi)


def assign(data: dict, distributions: dict):
    """Resolve the ``distributions`` mapping to ``(paths, marginals)``.

    Keys are top-level data keys or dotted entry names; a more specific key
    overrides a less specific one, whatever the order in the mapping.
    """
    if not distributions:
        raise ValueError("scenarios.method 'distribution' needs a 'distributions' mapping")
    chosen: dict = {}            # path -> (depth of the key that set it, params, key)
    order: list = []
    for key, params in distributions.items():
        paths = du.expand_keys(data, [key])
        depth = str(key).count(".")
        for path in paths:
            if path not in chosen:
                order.append(path)
            if path not in chosen or depth >= chosen[path][0]:
                chosen[path] = (depth, params, key)
    marginals = [marginal(chosen[p][1], du.get_leaf(data, p), f"distributions.{du.leaf_name(p)}")
                 for p in order]
    return order, marginals


def _copula_uniforms(n: int, d: int, correlation: float, rng) -> np.ndarray:
    if d == 0:
        return np.empty((n, 0))
    rho = float(correlation)
    if not -1.0 / max(d - 1, 1) - 1e-12 <= rho <= 1.0 or (d == 1 and rho < 0):
        raise ValueError(f"correlation must be between {-1.0 / max(d - 1, 1):.3g} and 1 for {d} entries")
    z0 = rng.standard_normal((n, 1))
    if rho >= 0:
        z = np.sqrt(rho) * z0 + np.sqrt(1 - rho) * rng.standard_normal((n, d))
    else:
        # equicorrelated normals with negative correlation via the eigen-decomposition
        C = np.full((d, d), rho)
        np.fill_diagonal(C, 1.0)
        w, V = np.linalg.eigh(C)
        z = rng.standard_normal((n, d)) @ (V * np.sqrt(np.clip(w, 0, None))).T
    u = stats.norm.cdf(z)
    return np.clip(u, 1e-12, 1 - 1e-12)


class DistributionSampler:
    """Draw realisations of the uncertain entries from the specified distributions."""

    def __init__(self, data: dict, distributions: dict, correlation: float = 0.0):
        self.paths, self.marginals = assign(data, distributions)
        self.correlation = float(correlation or 0.0)
        _copula_uniforms(1, len(self.paths), self.correlation, np.random.default_rng(0))  # validate

    @property
    def names(self) -> list:
        return [du.leaf_name(p) for p in self.paths]

    def mean(self) -> np.ndarray:
        """Mean of each marginal before clipping (clipping can shift the sample mean)."""
        return np.array([m.mean for m in self.marginals])

    def sample(self, n: int, rng) -> np.ndarray:
        if n < 1:
            raise ValueError("the number of scenarios n must be at least 1")
        U = _copula_uniforms(int(n), len(self.paths), self.correlation, rng)
        return np.column_stack([m.ppf(U[:, j]) for j, m in enumerate(self.marginals)]) if self.paths \
            else np.empty((int(n), 0))

    def describe(self) -> list:
        return [f"{nm}: {m.name}" for nm, m in zip(self.names, self.marginals)]


def sample(data: dict, distributions: dict, n: int, seed: int = 0, correlation: float = 0.0):
    """A :class:`ScenarioSet` of ``n`` equally likely draws from ``distributions``.

    >>> sample(DATA, {"yield": {"dist": "normal", "cv": 0.15}}, n=200, correlation=0.8)
    """
    from .scenarios import ScenarioSet

    s = DistributionSampler(data, distributions, correlation)
    values = s.sample(n, np.random.default_rng(seed))
    return ScenarioSet(list(s.paths), values, np.full(int(n), 1.0 / int(n)))
