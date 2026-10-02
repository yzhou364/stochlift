"""Independent checks of the lift against results that do not come from StochLift."""
import numpy as np
import pytest

import stochlift as sl
from conftest import pulp_var


def newsvendor(data):
    pulp = pytest.importorskip("pulp")
    m = pulp.LpProblem("newsvendor", pulp.LpMinimize)
    q = pulp_var(m, "order", 0)
    sold = pulp_var(m, "sold", 0)
    left = pulp_var(m, "left", 0)
    m += data["cost"] * q - data["price"] * sold - data["salvage"] * left
    m += sold <= data["demand"], "demand"
    m += sold + left == q, "balance"
    return m


def test_newsvendor_matches_the_critical_ratio():
    """Closed form: order the smallest demand d with P(D <= d) >= (price - cost) / (price - salvage)."""
    rng = np.random.default_rng(3)
    demand = np.sort(rng.integers(20, 200, size=25).astype(float))
    probs = rng.dirichlet(np.ones(25))
    data = {"cost": 6.0, "price": 10.0, "salvage": 1.0, "demand": 100.0}
    ratio = (10.0 - 6.0) / (10.0 - 1.0)
    q_star = demand[np.searchsorted(np.cumsum(probs), ratio)]
    expected_cost = float(sum(p * (6 * q_star - 10 * min(q_star, d) - 1 * max(q_star - d, 0))
                              for p, d in zip(probs, demand)))
    study = sl.lift(newsvendor, data, first_stage=["order"],
                    scenarios=[(p, {"demand": d}) for p, d in zip(probs, demand)])
    r = study.solve()
    assert r.x_rp["order"] == pytest.approx(q_star)
    assert r.rp == pytest.approx(expected_cost)
    assert r.ws == pytest.approx(float(-4.0 * probs @ demand))      # perfect information: order = demand
    assert all(c.status == "pass" for c in study.check())


def test_integer_first_stage_against_enumeration(facility):
    """A binary first stage: the extensive form must agree with brute-force enumeration."""
    import itertools

    from stochlift.evaluate import evaluate_first_stage

    rng = np.random.default_rng(0)
    base = np.array(list(facility.DATA["demand"].values()), dtype=float)
    scen = [(1 / 6, {"demand": dict(zip(facility.CUSTOMERS, base * rng.uniform(0.6, 1.6, size=8)))})
            for _ in range(6)]
    study = sl.lift(facility.build_model, facility.DATA, first_stage=["open[*]"], scenarios=scen)
    r = study.solve()
    models = study.models()
    best = min(float(np.mean(evaluate_first_stage(models, {f"open[{s}]": float(v) for s, v in
                                                          zip(facility.SITES, combo)})))
               for combo in itertools.product([0, 1], repeat=5))
    assert r.rp == pytest.approx(best, rel=1e-6)
    assert all(c.status == "pass" for c in study.check())


def test_first_stage_only_rows_are_not_duplicated(farmer):
    from conftest import farmer_scenarios
    from stochlift.lift import extensive_form

    study = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"],
                    scenarios=farmer_scenarios(farmer.DATA))
    ef = extensive_form(study.models(), study.scenarios.probs, study.is_first)
    assert ef.model.n == 3 + 3 * 6                 # 3 shared acres, 6 recourse variables per scenario
    assert ef.model.m == 1 + 3 * 4                 # the land row once, 4 recourse rows per scenario
    assert sum(name.startswith("land") for name in ef.model.row_names) == 1


def test_spec_round_trip_and_patterns(tmp_path):
    spec = sl.Spec(first_stage=["open[*]", "x_?"], uncertain="demand", holdout=0.25,
                   scenarios={"method": "kmeans", "n": 10})
    assert [spec.is_first_stage(n) for n in ("open[S1]", "open", "x_1", "x_12", "ship[S1,C1]")] == \
        [True, False, True, False, False]
    spec.save(tmp_path / "s.yaml")
    again = sl.Spec.load(tmp_path / "s.yaml")
    assert again.to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match="unknown spec fields"):
        sl.Spec.from_dict({"first_stage": ["a"], "stages": 2})
