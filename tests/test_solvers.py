"""Solver backends and parallel scenario solves give the same answers."""
import numpy as np
import pytest
import scipy.sparse as sp

import stochlift as sl
from stochlift.model import LinearModel, solve
from conftest import farmer_scenarios


def random_model(seed, integer=False):
    rng = np.random.default_rng(seed)
    n, m = 12, 8
    A = sp.random(m, n, density=0.5, random_state=seed, data_rvs=lambda k: rng.integers(-3, 6, k)).tocsr()
    x0 = rng.uniform(0, 4, n)                                 # a feasible point
    act = A @ x0
    lo = np.where(rng.random(m) < 0.5, act - rng.uniform(0, 3, m), -np.inf)
    hi = np.where(rng.random(m) < 0.7, act + rng.uniform(0, 3, m), np.inf)
    eq = rng.random(m) < 0.15
    lo[eq] = hi[eq] = act[eq]
    if integer:
        x0 = np.round(x0)
        act = A @ x0
        lo, hi = np.minimum(lo, act), np.maximum(hi, act)
    return LinearModel(names=[f"x{j}" for j in range(n)], c=rng.normal(size=n), offset=rng.normal(),
                       lb=np.zeros(n), ub=np.full(n, 6.0), integer=np.full(n, integer) & (np.arange(n) % 2 == 0),
                       A=A, row_lb=lo, row_ub=hi)


@pytest.mark.parametrize("seed", range(12))
def test_gurobi_matches_highs(seed):
    pytest.importorskip("gurobipy")
    for integer in (False, True):
        lm = random_model(seed, integer)
        a, b = solve(lm), solve(lm, solver="gurobi")
        assert a.status == b.status == "optimal"
        assert a.objective == pytest.approx(b.objective, abs=1e-6 * max(1, abs(a.objective)))
        fixed = {0: 1.0, 3: 2.0}
        assert solve(lm, fixed=fixed).objective == pytest.approx(
            solve(lm, fixed=fixed, solver="gurobi").objective, abs=1e-6 * max(1, abs(a.objective)))


def test_gurobi_statuses():
    pytest.importorskip("gurobipy")
    infeasible = LinearModel(["x"], [1.0], 0.0, [0.0], [1.0], [False], sp.csr_matrix([[1.0]]), [2.0], [np.inf])
    unbounded = LinearModel(["x"], [-1.0], 0.0, [0.0], [np.inf], [False], sp.csr_matrix((0, 1)), [], [])
    assert solve(infeasible, solver="gurobi").status == "infeasible"
    assert solve(unbounded, solver="gurobi").status == "unbounded"
    with pytest.raises(ValueError, match="solver must be one of"):
        solve(infeasible, solver="cplex")


def test_farmer_with_gurobi(farmer):
    pytest.importorskip("gurobipy")
    study = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"],
                    scenarios=farmer_scenarios(farmer.DATA), solver="gurobi")
    r = study.solve()
    assert -r.rp == pytest.approx(108390, abs=0.5) and r.vss == pytest.approx(1150, abs=0.5)
    assert all(c.status == "pass" for c in study.check())


def test_parallel_scenario_solves_change_nothing(farmer):
    spec = {"first_stage": ["acres_*"], "scenarios": {
        "method": "distribution", "n": 24, "n_test": 30, "seed": 4,
        "distributions": {"yield": {"dist": "normal", "cv": 0.2, "min": 0}}}}
    out = []
    for jobs in (1, 4):
        study = sl.lift(farmer.build_model, farmer.DATA, spec=spec, n_jobs=jobs)
        r = study.solve()
        o = study.out_of_sample()
        out.append((r.rp, r.ws, r.eev, tuple(r.scenario_costs_ws), tuple(o["costs_rp"])))
    assert out[0] == out[1]
