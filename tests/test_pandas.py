"""Data held in pandas Series and DataFrames, as many real models keep it."""
import numpy as np
import pandas as pd
import pytest

import stochlift as sl
from stochlift import datautil as du
from conftest import pulp_var

SITES, CUSTOMERS = ["s1", "s2"], ["a", "b", "c"]


def make_data():
    return {
        "capacity_cost": pd.Series([4.0, 5.0], index=SITES),
        "demand": pd.Series([30, 20, 25], index=CUSTOMERS),           # integer dtype on purpose
        "ship": pd.DataFrame([[1.0, 2.0, 3.0], [3.0, 1.5, 1.0]], index=SITES, columns=CUSTOMERS),
        "penalty": 40.0,
    }


def build(data):
    import pulp

    m = pulp.LpProblem("cap", pulp.LpMinimize)
    cap = {i: pulp_var(m, f"cap_{i}", 0) for i in SITES}
    x = {(i, j): pulp_var(m, f"x_{i}_{j}", 0) for i in SITES for j in CUSTOMERS}
    short = {j: pulp_var(m, f"short_{j}", 0) for j in CUSTOMERS}
    m += (pulp.lpSum(data["capacity_cost"][i] * cap[i] for i in SITES)
          + pulp.lpSum(data["ship"].loc[i, j] * x[i, j] for i in SITES for j in CUSTOMERS)
          + data["penalty"] * pulp.lpSum(short.values()))
    for j in CUSTOMERS:
        m += pulp.lpSum(x[i, j] for i in SITES) + short[j] >= float(data["demand"][j]), f"meet_{j}"
    for i in SITES:
        m += pulp.lpSum(x[i, j] for j in CUSTOMERS) <= cap[i], f"cap_{i}"
    return m


def test_pandas_leaves_and_writes():
    data = make_data()
    leaves = dict(du.numeric_leaves(data))
    assert leaves[("demand", "b")] == 20.0 and leaves[("ship", "s2", "c")] == 1.0
    assert du.expand_keys(data, ["ship.s1"]) == [("ship", "s1", c) for c in CUSTOMERS]
    new = du.apply(data, [("demand", "a"), ("ship", "s2", "a")], [31.5, 2.5])
    assert new["demand"]["a"] == 31.5 and new["ship"].loc["s2", "a"] == 2.5
    assert data["demand"]["a"] == 30 and data["ship"].loc["s2", "a"] == 3.0    # original untouched
    arr = du.apply({"d": np.array([1, 2, 3])}, [("d", 1)], [2.5])               # integer NumPy array
    assert arr["d"].tolist() == [1.0, 2.5, 3.0]
    nan = {"d": pd.Series([1.0, np.nan], index=["x", "y"])}
    assert du.numeric_leaves(nan) == [(("d", "x"), 1.0)]


def test_explicit_and_history_scenarios_with_pandas(tmp_path):
    pytest.importorskip("pulp")
    data = make_data()
    scen = [(0.5, {"demand": {"a": 20, "b": 10, "c": 15}}), (0.5, {"demand": {"a": 40, "b": 30, "c": 35}})]
    r = sl.lift(build, data, first_stage=["cap_*"], scenarios=scen).solve()
    assert r.ws <= r.rp + 1e-6 <= r.eev + 2e-6

    rng = np.random.default_rng(0)
    hist = pd.DataFrame(rng.poisson([30, 20, 25], size=(60, 3)), columns=CUSTOMERS)
    path = tmp_path / "h.csv"
    hist.to_csv(path, index=False)
    study = sl.lift(build, data, spec={"first_stage": ["cap_*"], "uncertain": ["demand"], "holdout": 0.25},
                    history=str(path))
    study.solve()
    assert all(c.status == "pass" for c in study.check())
    assert study.out_of_sample()["n"] == 15


def test_distributions_over_a_dataframe():
    pytest.importorskip("pulp")
    spec = {"first_stage": ["cap_*"], "uncertain": ["demand", "ship"],
            "scenarios": {"method": "distribution", "n": 15, "n_test": 0, "distributions": {
                "demand": {"dist": "lognormal", "cv": 0.3},
                "ship": {"dist": "uniform", "spread": 0.2}}}}
    study = sl.lift(build, make_data(), spec=spec)
    assert len(study.paths) == 3 + 6
    study.solve()
    assert all(c.status != "fail" for c in study.check())
