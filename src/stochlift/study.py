"""The user-facing workflow: lift -> review -> solve -> check -> report."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import datautil as du
from . import scenarios as sc
from .adapters import to_linear_model
from .evaluate import (SolveError, complete_first_stage, evaluate_first_stage, expected, first_stage_values,
                       solve_recourse_problem, wait_and_see)
from .lift import first_stage_names
from .model import LinearModel, solve
from .risk import Risk
from .spec import Spec


@dataclass
class Results:
    sense: str
    ev: float                 # deterministic model at mean data (what the user has today)
    ws: float                 # wait-and-see (perfect information)
    rp: float                 # recourse problem (stochastic solution)
    eev: float                # expected result of using the mean-value solution
    vss: float                # value of the stochastic solution, >= 0
    evpi: float               # expected value of perfect information, >= 0
    x_ev: dict
    x_rp: dict
    n_scenarios: int
    eev_infeasible: int       # scenarios where the mean-value first stage has no recourse
    scenario_costs_ev: np.ndarray = field(repr=False, default=None)
    scenario_costs_rp: np.ndarray = field(repr=False, default=None)
    scenario_costs_ws: np.ndarray = field(repr=False, default=None)
    objective: str = "expected cost"   # what EV, WS, RP and EEV measure
    risk_parts: dict = field(default=None)   # with a risk measure: mean and CVaR of WS, RP, EEV

    @property
    def vss_pct(self) -> float:
        return 100.0 * self.vss / abs(self.rp) if self.rp else float("nan")

    @property
    def evpi_pct(self) -> float:
        return 100.0 * self.evpi / abs(self.rp) if self.rp else float("nan")

    def to_dict(self) -> dict:
        return {"sense": self.sense, "EV": self.ev, "WS": self.ws, "RP": self.rp,
                "EEV": self.eev, "VSS": self.vss, "EVPI": self.evpi,
                "VSS_pct_of_RP": self.vss_pct, "EVPI_pct_of_RP": self.evpi_pct,
                "n_scenarios": self.n_scenarios, "EEV_infeasible_scenarios": self.eev_infeasible,
                "objective": self.objective, "risk_parts": self.risk_parts,
                "first_stage_mean_value_solution": self.x_ev,
                "first_stage_stochastic_solution": self.x_rp}


class Study:
    """A deterministic model plus an uncertainty specification."""

    def __init__(self, build_model: Callable, data: dict, spec, scenarios=None, history=None,
                 mip_gap: float = 1e-6, time_limit: Optional[float] = None, solver: str = "highs",
                 n_jobs: Optional[int] = None):
        self.build_model = build_model
        self.data = data
        self.spec = Spec.load(spec)
        from .model import SOLVERS

        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        self.opts = {"mip_gap": mip_gap, "time_limit": time_limit, "solver": solver}
        self.n_jobs = n_jobs          # parallel scenario solves; None = up to 8 threads
        self.risk = Risk.from_spec(self.spec.risk)
        self.frontier: Optional[list] = None
        self._cache: dict = {}
        self.train = self.test = None
        self.history_columns = None
        self.results: Optional[Results] = None
        self.checks: list = []
        self.oos: Optional[dict] = None
        self.stability_table: Optional[list] = None
        self.gap: Optional[dict] = None
        self.lift_notes: list = []
        self.sampler = None           # set when scenarios are drawn from distributions

        method = self.spec.scenarios.get("method", "empirical")
        if scenarios is not None:
            if not isinstance(scenarios, sc.ScenarioSet):
                scenarios = sc.explicit(data, scenarios)
            self.scenarios = scenarios
            self.paths = list(scenarios.paths)
            if not self.spec.uncertain:
                self.spec.uncertain = list(scenarios.names)
            self.spec.scenarios = {"method": "explicit", "n": len(scenarios)}
        elif method == "distribution":
            if history is not None:
                raise ValueError("scenarios.method 'distribution' does not use a history; "
                                 "remove history=... or choose empirical, sample or kmeans")
            self._init_distribution()
        else:
            if history is None:
                raise ValueError("no source of scenarios: pass scenarios=..., a history=..., or set "
                                 "scenarios.method to 'distribution' in the spec")
            if not self.spec.uncertain:
                raise ValueError("the spec lists no uncertain data keys")
            self.paths = du.expand_keys(data, self.spec.uncertain)
            X = self._load_history(history)
            n_test = int(round(len(X) * float(self.spec.holdout)))
            self.train, self.test = (X[:len(X) - n_test], X[len(X) - n_test:]) if n_test else (X, None)
            s = self.spec.scenarios
            self.scenarios = sc.from_history(self.train, self.paths, method=s.get("method", "empirical"),
                                             n=s.get("n"), seed=int(s.get("seed", 0)))

    def _init_distribution(self):
        from .distributions import DistributionSampler

        s = self.spec.scenarios
        self.sampler = DistributionSampler(self.data, s.get("distributions"), s.get("correlation", 0.0))
        if self.spec.uncertain:
            missing = [du.leaf_name(p) for p in du.expand_keys(self.data, self.spec.uncertain)
                       if p not in self.sampler.paths]
            if missing:
                raise ValueError(f"uncertain entries {missing[:6]} have no distribution under "
                                 "scenarios.distributions")
        else:
            self.spec.uncertain = [str(k) for k in s["distributions"]]
        self.paths = list(self.sampler.paths)
        seed = int(s.get("seed", 0))
        n = int(s.get("n", 100))
        self.scenarios = sc.ScenarioSet(self.paths, self.sampler.sample(n, np.random.default_rng(seed)),
                                        np.full(n, 1.0 / n))
        # an independent sample plays the role of the hold-out period
        n_test = int(s.get("n_test", 500))
        self.test = self.sampler.sample(n_test, np.random.default_rng(seed + 1)) if n_test else None

    def _draw(self, n: int, rng) -> np.ndarray:
        """``n`` equally likely realisations from the reference distribution."""
        if self.sampler is not None:
            return self.sampler.sample(n, rng)
        pool, w = self._sample_pool()
        return pool[rng.choice(len(pool), size=n, replace=True, p=w)]

    # ------------------------------------------------------------------ building
    def _load_history(self, history) -> np.ndarray:
        import pandas as pd

        df = pd.read_csv(history) if isinstance(history, (str, os.PathLike)) else history
        cols = sc.match_columns(self.paths, df.columns, self.spec.columns)
        self.history_columns = cols
        X = df[cols].to_numpy(dtype=float)
        if np.isnan(X).any():
            bad = int(np.isnan(X).any(axis=1).sum())
            raise ValueError(f"history has missing values in {bad} row(s); clean or impute them first")
        return X

    def _observations(self, obs) -> np.ndarray:
        """Observations as an (n, d) array in the order of the uncertain entries."""
        if hasattr(obs, "columns"):
            cols = self.history_columns or sc.match_columns(self.paths, obs.columns, self.spec.columns)
            obs = obs[cols]
        X = np.atleast_2d(np.asarray(obs, dtype=float))
        if X.shape[1] != len(self.paths):
            raise ValueError(f"observations need {len(self.paths)} columns (one per uncertain entry: "
                             f"{self.scenarios.names[:4]}...), got {X.shape[1]}")
        if np.isnan(X).any():
            raise ValueError("observations contain missing values")
        return X

    def is_first(self, name: str) -> bool:
        return self.spec.is_first_stage(name)

    def model_for(self, values) -> LinearModel:
        """The user's model built with the uncertain entries set to ``values`` (cached)."""
        values = np.asarray(values, dtype=float).ravel()
        if len(values) != len(self.paths):
            raise ValueError(f"expected {len(self.paths)} values (one per uncertain entry), got {len(values)}")
        key = values.tobytes()
        if key not in self._cache:
            data = du.apply(self.data, self.paths, values)
            if len(self._cache) >= _CACHE_SIZE:      # sampled scenarios are rarely rebuilt: drop the oldest
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = to_linear_model(self.build_model(data))
        return self._cache[key]

    def models(self, values=None) -> list:
        values = self.scenarios.values if values is None else values
        return [self.model_for(v) for v in values]

    def mean_model(self) -> LinearModel:
        return self.model_for(self.scenarios.mean())

    # ------------------------------------------------------------------- probing
    def probe(self, keys=None, rel: float = 0.1) -> dict:
        """Where does each data key enter the model?

        Every numeric entry under a key is shifted by ``rel`` and the model is
        rebuilt; the result counts the coefficients that changed, split into
        objective, constraint matrix, right-hand sides and variable bounds.
        With ``keys=None`` all numeric top-level keys are probed.
        """
        if keys is None:
            keys = [k for k, v in self.data.items() if du.numeric_leaves(v)]
        paths = []
        for key in keys:
            try:
                paths += du.expand_keys(self.data, [key])
            except KeyError:
                pass
        out = probe_paths(self.build_model, self.data, paths, self.is_first, rel, per_leaf=False)
        for key in keys:
            out.setdefault(str(key), {"error": "not a numeric data entry"})
        return out

    def probe_uncertain(self, rel: float = 0.1) -> dict:
        """The same, but changing only the entries declared uncertain."""
        return probe_paths(self.build_model, self.data, self.paths, self.is_first, rel)

    # --------------------------------------------------------------------- solve
    def solve(self) -> Results:
        S = self.scenarios
        models = self.models()
        sign = models[0].sign
        mean = self.mean_model()

        ev_sol = solve(mean, **self.opts)
        if not ev_sol.ok:
            raise SolveError(f"the deterministic model at mean data is {ev_sol.status}")
        if not first_stage_names(models + [mean], self.is_first):
            raise ValueError(f"first_stage patterns {self.spec.first_stage} match no variable. "
                             f"Variables look like: {mean.names[:8]}")
        x_ev = first_stage_values(mean, ev_sol.x, self.is_first)
        # a first-stage variable can be missing from the mean-value model (for instance when
        # its coefficient is zero there); it then has no mean-value decision and is set to a default
        x_ev, missing = complete_first_stage(x_ev, models, self.is_first)

        from . import diagnose

        ws, statuses = wait_and_see(models, n_jobs=self.n_jobs, **self.opts)
        if any(st != "optimal" for st in statuses):
            raise diagnose.scenarios_alone(self, statuses)
        try:
            z_rp, x_rp, ef = solve_recourse_problem(models, S.probs, self.is_first, risk=self.risk,
                                                    n_jobs=self.n_jobs, **self.opts)
        except SolveError as e:
            if e.status == "infeasible":
                raise diagnose.common_first_stage(self, models) from e
            raise
        self.lift_notes = list(ef.notes)
        if missing:
            self.lift_notes.append(
                f"first-stage variables {missing[:5]} do not appear in the mean-value model; the "
                "mean-value decision sets them to 0 (or the nearest bound)")
        costs_ev = evaluate_first_stage(models, x_ev, self.is_first, n_jobs=self.n_jobs, **self.opts)
        costs_rp = evaluate_first_stage(models, x_rp, self.is_first, n_jobs=self.n_jobs, **self.opts)
        z_ws = self.risk.value(ws, S.probs)
        z_eev = self.risk.value(costs_ev, S.probs)
        parts = None
        if self.risk.active:
            from .risk import cvar

            def split(costs):
                if np.isnan(costs).any():
                    return {"mean": float("inf") * sign, "cvar": float("inf") * sign}
                return {"mean": sign * expected(costs, S.probs),
                        "cvar": sign * cvar(costs, S.probs, self.risk.alpha)}

            parts = {"WS": split(ws), "RP": split(costs_rp), "EEV": split(costs_ev)}

        self.results = Results(
            sense="min" if sign > 0 else "max", ev=sign * ev_sol.objective, ws=sign * z_ws,
            rp=sign * z_rp, eev=sign * z_eev, vss=_clamp(z_eev - z_rp, z_rp), evpi=_clamp(z_rp - z_ws, z_rp),
            x_ev=x_ev, x_rp=x_rp, n_scenarios=len(S),
            eev_infeasible=int(np.isnan(costs_ev).sum()),
            scenario_costs_ev=sign * costs_ev, scenario_costs_rp=sign * costs_rp,
            scenario_costs_ws=sign * ws,
            objective=self.risk.describe("min" if sign > 0 else "max"), risk_parts=parts)
        self._internal = {"z_rp": z_rp, "z_ws": z_ws, "z_eev": z_eev, "z_ev": ev_sol.objective,
                          "costs_rp": costs_rp, "sign": sign}
        return self.results

    def check(self) -> list:
        from .checks import run_checks

        if self.results is None:
            self.solve()
        self.checks = run_checks(self)
        return self.checks

    # ------------------------------------------------------------- out of sample
    def out_of_sample(self, observations=None, n_boot: int = 10000, seed: int = 0) -> dict:
        """Apply both first-stage decisions to observations that were not used to build scenarios.

        These are the held-out rows of the history, an independent sample when the
        scenarios come from distributions, or ``observations`` when given.
        """
        if self.results is None:
            self.solve()
        X = self.test if observations is None else self._observations(observations)
        if X is None or len(X) == 0:
            raise ValueError("no hold-out observations: set 'holdout' in the spec or pass observations")
        models = self.models(X)
        sign = models[0].sign
        c_ev = evaluate_first_stage(models, self.results.x_ev, self.is_first, n_jobs=self.n_jobs, **self.opts)
        c_rp = evaluate_first_stage(models, self.results.x_rp, self.is_first, n_jobs=self.n_jobs, **self.opts)
        both = ~np.isnan(c_ev) & ~np.isnan(c_rp)
        d = (c_ev - c_rp)[both]           # > 0: the stochastic solution is better
        rng = np.random.default_rng(seed)
        if len(d) > 1:
            idx = rng.integers(0, len(d), size=(n_boot, len(d)))
            boot = d[idx].mean(axis=1)
            ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        else:
            idx, ci = None, [float("nan"), float("nan")]
        risk = {}
        if self.risk.active and both.any():
            a, b = c_ev[both], c_rp[both]
            gain = float(self.risk.value(a) - self.risk.value(b))
            rci = [float("nan"), float("nan")]
            if idx is not None:
                rb = self.risk.values_equal_weights(a[idx]) - self.risk.values_equal_weights(b[idx])
                rci = [float(np.quantile(rb, 0.025)), float(np.quantile(rb, 0.975))]
            risk = {"risk_ev": float(sign * self.risk.value(a)), "risk_rp": float(sign * self.risk.value(b)),
                    "risk_gain": gain, "risk_gain_ci95": rci}
        self.oos = {
            "n": int(len(X)), "n_feasible_ev": int((~np.isnan(c_ev)).sum()),
            "n_feasible_rp": int((~np.isnan(c_rp)).sum()), "n_compared": int(both.sum()), "n_boot": int(n_boot),
            "mean_ev": float(sign * c_ev[both].mean()) if both.any() else float("nan"),
            "mean_rp": float(sign * c_rp[both].mean()) if both.any() else float("nan"),
            "mean_gain": float(d.mean()) if len(d) else float("nan"), "gain_ci95": ci,
            "share_rp_better": float((d > 1e-9).mean()) if len(d) else float("nan"),
            "share_ev_better": float((d < -1e-9).mean()) if len(d) else float("nan"),
            "costs_ev": sign * c_ev, "costs_rp": sign * c_rp,
            "source": ("given observations" if observations is not None else
                       "independent samples from the distributions" if self.sampler is not None
                       else "held-out observations"),
            **risk,
        }
        return self.oos

    # ----------------------------------------------------------------- stability
    def _sample_pool(self):
        if self.train is not None:
            return self.train, np.full(len(self.train), 1.0 / len(self.train))
        return self.scenarios.values, self.scenarios.probs

    def stability(self, sizes=(5, 10, 20, 50), reps: int = 10, seed: int = 0) -> list:
        """Re-solve with ``reps`` random scenario samples of each size.

        Reports the spread of the in-sample optimum and, when a hold-out set
        exists, the hold-out result of each sampled solution.
        """
        rng = np.random.default_rng(seed)
        test_models = self.models(self.test) if self.test is not None else None
        rows = []
        for n in sizes:
            for r in range(reps):
                models = self.models(self._draw(n, rng))
                z, x, _ = solve_recourse_problem(models, np.full(n, 1.0 / n), self.is_first,
                                                 risk=self.risk, n_jobs=self.n_jobs, **self.opts)
                sign = models[0].sign
                row = {"n": int(n), "rep": r, "in_sample": sign * z}
                if test_models is not None:
                    c = evaluate_first_stage(test_models, x, self.is_first, n_jobs=self.n_jobs, **self.opts)
                    ok = c[~np.isnan(c)]
                    row["holdout"] = float(sign * self.risk.value(ok)) if len(ok) else float("nan")
                    row["holdout_infeasible"] = int(np.isnan(c).sum())
                rows.append(row)
        self.stability_table = rows
        return rows

    def saa_gap(self, n: int = 50, batches: int = 20, seed: int = 0) -> dict:
        """Optimality-gap estimate for the stochastic solution (Mak, Morton and Wood, 1999).

        The reference distribution is the specified distributions, the empirical
        distribution of the training history, or the scenario set, in that order.
        Each batch draws ``n`` scenarios and measures how much worse the
        solution is than the batch optimum; the mean over batches estimates
        an upper bound on the true gap.
        """
        from scipy import stats

        if self.results is None:
            self.solve()
        rng = np.random.default_rng(seed)
        gaps = []
        for _ in range(batches):
            models = self.models(self._draw(n, rng))
            z, _, _ = solve_recourse_problem(models, np.full(n, 1.0 / n), self.is_first,
                                             risk=self.risk, n_jobs=self.n_jobs, **self.opts)
            c = evaluate_first_stage(models, self.results.x_rp, self.is_first, n_jobs=self.n_jobs, **self.opts)
            gaps.append(float(self.risk.value(c) - z))
        g = np.array(gaps)
        mean, sd = float(g.mean()), float(g.std(ddof=1)) if batches > 1 else 0.0
        upper = mean + stats.t.ppf(0.95, batches - 1) * sd / np.sqrt(batches) if batches > 1 else mean
        self.gap = {"n": n, "batches": batches, "mean_gap": mean, "upper95": float(upper),
                    "pct_of_RP": 100.0 * float(upper) / abs(self.results.rp) if self.results.rp else float("nan")}
        return self.gap

    # ---------------------------------------------------------------------- risk
    def risk_frontier(self, weights=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0), alpha: float = None) -> list:
        """Mean-CVaR trade-off: solve with each CVaR weight and report the expected
        cost and the CVaR of the resulting decision on the scenario set (and out of
        sample when there is a hold-out set)."""
        from .risk import cvar

        alpha = self.risk.alpha if alpha is None else float(alpha)
        models = self.models()
        probs = self.scenarios.probs
        test_models = self.models(self.test) if self.test is not None else None
        sign = models[0].sign
        rows = []
        for w in weights:
            r = Risk(alpha=alpha, weight=float(w))
            _, x, _ = solve_recourse_problem(models, probs, self.is_first, risk=r, n_jobs=self.n_jobs, **self.opts)
            c = evaluate_first_stage(models, x, self.is_first, n_jobs=self.n_jobs, **self.opts)
            row = {"weight": float(w), "alpha": alpha, "mean": sign * expected(c, probs),
                   "cvar": sign * cvar(c, probs, alpha) if not np.isnan(c).any() else sign * float("inf"),
                   "first_stage": x}
            if test_models is not None:
                t = evaluate_first_stage(test_models, x, self.is_first, n_jobs=self.n_jobs, **self.opts)
                t = t[~np.isnan(t)]
                row["holdout_mean"] = float(sign * t.mean()) if len(t) else float("nan")
                row["holdout_cvar"] = float(sign * cvar(t, np.full(len(t), 1 / len(t)), alpha))                     if len(t) else float("nan")
            rows.append(row)
        self.frontier = rows
        return rows

    # -------------------------------------------------------------------- export
    def to_mpisppy(self) -> dict:
        """Scenario names and a ``scenario_creator`` for mpi-sppy (see :mod:`stochlift.export`)."""
        from .export import to_mpisppy

        return to_mpisppy(self)

    # -------------------------------------------------------------------- output
    def figures(self, outdir, style: str = "nature") -> list:
        """Write every available figure, a multi-panel overview, ``captions.md`` and
        ``source_data/`` to ``outdir``; returns the PDF paths. See :mod:`stochlift.plots`."""
        from .plots import make_figures

        if self.results is None:
            self.solve()
        os.makedirs(outdir, exist_ok=True)
        return make_figures(self, outdir, style=style)

    def review(self, show: bool = True) -> str:
        from .report import review_text

        text = review_text(self)
        if show:
            print(text)
        return text

    def report(self, outdir, figures: bool = True, style: str = "nature") -> str:
        """Write summary.md, results.json, a LaTeX table and, with ``figures``, every figure in
        ``style`` ("nature": final print size for journals; "presentation": slide size)."""
        from .report import write_report

        if self.results is None:
            self.solve()
        if not self.checks:
            self.check()
        return write_report(self, outdir, figures=figures, style=style)


_CACHE_SIZE = 4096


def _clamp(value: float, scale: float) -> float:
    """VSS and EVPI are non-negative in exact arithmetic; round-off on either side of zero is zero."""
    return 0.0 if abs(value) < 1e-9 * max(1.0, abs(scale)) else float(value)


def probe_paths(build_model, data, paths, is_first, rel: float = 0.1, per_leaf: bool = True,
                max_leaf_probes: int = 60) -> dict:
    """Perturb the given entries, grouped by top-level key, and diff the rebuilt model.

    With ``per_leaf`` (and at most ``max_leaf_probes`` entries) each entry is also
    perturbed on its own, and entries that change nothing are listed under ``dead``.
    """
    base = to_linear_model(build_model(copy.deepcopy(data)))
    groups: dict = {}
    for p in paths:
        groups.setdefault(str(p[0]), []).append(p)

    def perturbed(ps):
        vals = [du.get_leaf(data, p) for p in ps]
        new = [float(v) * (1 + rel) if v != 0 else rel for v in vals]
        return to_linear_model(build_model(du.apply(data, ps, new)))

    def changed(d) -> bool:
        return bool(d.get("structure_changed") or d["objective"] + d["matrix"] + d["rhs"] + d["bounds"]
                    or d["offset"])

    out = {}
    single = per_leaf and len(paths) <= max_leaf_probes
    for key, ps in groups.items():
        try:
            d = _diff(base, perturbed(ps), is_first)
        except Exception as e:  # the builder may reject perturbed data (e.g. an index set)
            out[key] = {"error": f"the model cannot be rebuilt when it changes ({type(e).__name__}: {e})"}
            continue
        d["dead"] = []
        if not changed(d):
            d["dead"] = [du.leaf_name(p) for p in ps]
        elif single and len(ps) > 1 and not d.get("structure_changed"):
            for p in ps:
                try:
                    if not changed(_diff(base, perturbed([p]), is_first)):
                        d["dead"].append(du.leaf_name(p))
                except Exception:
                    d["dead"].append(du.leaf_name(p))
        out[key] = d
    return out


def _diff(base: LinearModel, pert: LinearModel, is_first) -> dict:
    if base.names != pert.names or base.row_names != pert.row_names:
        return {"structure_changed": True}
    obj = np.flatnonzero(~np.isclose(base.c, pert.c, rtol=1e-12, atol=0))
    D = (base.A - pert.A).tocoo()
    nz = np.abs(D.data) > 0
    rhs = np.flatnonzero(~(np.isclose(base.row_lb, pert.row_lb, rtol=1e-12, atol=0, equal_nan=True)
                           & np.isclose(base.row_ub, pert.row_ub, rtol=1e-12, atol=0, equal_nan=True)))
    bnd = np.flatnonzero(~(np.isclose(base.lb, pert.lb, rtol=1e-12, atol=0, equal_nan=True)
                           & np.isclose(base.ub, pert.ub, rtol=1e-12, atol=0, equal_nan=True)))
    rows = sorted(set(D.row[nz].tolist()) | set(rhs.tolist()))
    cols = sorted(set(D.col[nz].tolist()) | set(obj.tolist()) | set(bnd.tolist()))
    first = [base.names[j] for j in cols if is_first(base.names[j])]
    return {"objective": int(len(obj)), "matrix": int(nz.sum()), "rhs": int(len(rhs)),
            "bounds": int(len(bnd)),
            "offset": bool(abs(base.offset - pert.offset) > 1e-12 * max(1.0, abs(base.offset))),
            "example_rows": [base.row_names[i] for i in rows[:4]],
            "example_vars": [base.names[j] for j in cols[:4]],
            "touches_first_stage_vars": len(first),
            "structure_changed": False}


def lift(build_model: Callable, data: dict, *, spec=None, history=None, scenarios=None,
         first_stage=None, llm=None, **kwargs) -> Study:
    """Turn a deterministic model into a two-stage stochastic study.

    ``build_model(data)`` must return a PuLP / Pyomo / gurobipy / OR-Tools /
    highspy model built from the ``data`` dictionary.

    The uncertainty specification comes from, in order: ``spec`` (a
    :class:`Spec`, dict, YAML text or path), the ``first_stage`` shortcut
    together with explicit ``scenarios``, or a proposal from ``llm``.
    """
    if spec is None and first_stage is not None:
        spec = Spec(first_stage=first_stage)
    if spec is None:
        if llm is None:
            raise ValueError("no uncertainty specification: pass spec=..., or first_stage=... "
                             "with explicit scenarios, or llm=... to have one proposed")
        from .llm import propose_spec

        spec = propose_spec(build_model, data, history=history, llm=llm)
    return Study(build_model, data, spec, scenarios=scenarios, history=history, **kwargs)
