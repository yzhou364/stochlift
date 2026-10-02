"""Mean-CVaR objectives, checked against values that do not come from the extensive form."""
import numpy as np
import pytest

import stochlift as sl
from stochlift.risk import Risk, cvar
from conftest import farmer_scenarios, pulp_var


def test_cvar_by_hand():
    c, p = np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 0.25)
    assert cvar(c, p, 0.0) == pytest.approx(2.5)                 # CVaR_0 is the mean
    assert cvar(c, p, 0.5) == pytest.approx(3.5)                 # mean of the worst half
    assert cvar(c, p, 0.75) == pytest.approx(4.0)
    assert cvar(c, p, 0.6) == pytest.approx((4 * 0.25 + 3 * 0.15) / 0.4)   # splits an atom
    # unordered input, unequal probabilities: brute force over eta
    rng = np.random.default_rng(0)
    c, p = rng.normal(size=40), rng.dirichlet(np.ones(40))
    for a in (0.1, 0.5, 0.93):
        brute = min(e + p @ np.maximum(c - e, 0) / (1 - a) for e in np.concatenate([c, np.linspace(-4, 4, 801)]))
        assert cvar(c, p, a) == pytest.approx(brute, abs=1e-12)


def test_equal_weight_rows_match_the_scalar_version():
    rng = np.random.default_rng(1)
    C = rng.lognormal(size=(7, 23))
    for r in (Risk(0.9, 0.5), Risk(0.3, 1.0), Risk(0.0, 0.0)):
        assert r.values_equal_weights(C) == pytest.approx([r.value(row) for row in C])


def test_risk_validation():
    with pytest.raises(ValueError, match="alpha"):
        Risk(alpha=1.0)
    with pytest.raises(ValueError, match="weight"):
        Risk(weight=1.5)
    with pytest.raises(ValueError, match="measure"):
        Risk.from_spec({"measure": "variance"})
    with pytest.raises(ValueError, match="unknown risk fields"):
        sl.Spec(first_stage=["x"], risk={"alpha": 0.9, "beta": 1})
    assert Risk.from_spec({"alpha": 0.8}) == Risk(0.8, 1.0)


def newsvendor(d):
    import pulp

    m = pulp.LpProblem("newsvendor", pulp.LpMinimize)
    q, sold, left = pulp_var(m, "order", 0), pulp_var(m, "sold", 0), pulp_var(m, "left", 0)
    m += 1.0 * q - 3.0 * sold - 0.2 * left + 5.0          # constant: the objective offset enters CVaR rows
    m += sold <= d["demand"], "demand"
    m += sold + left == q, "balance"
    return m


@pytest.mark.parametrize("alpha, weight", [(0.8, 0.5), (0.9, 1.0), (0.5, 0.25)])
def test_newsvendor_matches_brute_force(alpha, weight):
    pytest.importorskip("pulp")
    demand = [20.0, 35.0, 50.0, 80.0, 120.0]
    probs = [0.1, 0.3, 0.3, 0.2, 0.1]
    scen = [(p, {"demand": d}) for p, d in zip(probs, demand)]
    study = sl.lift(newsvendor, {"demand": 50.0}, spec={"first_stage": ["order"],
                    "risk": {"alpha": alpha, "weight": weight}}, scenarios=scen)
    r = study.solve()
    risk = Risk(alpha, weight)

    def cost(q, d):
        s = min(q, d)
        return q - 3 * s - 0.2 * (q - s) + 5

    # the objective is piecewise linear in q with kinks at the demands: enumerate them
    best = min(risk.value([cost(q, d) for d in demand], probs) for q in [0.0] + demand)
    assert r.rp == pytest.approx(best, abs=1e-6)
    assert risk.value([cost(r.x_rp["order"], d) for d in demand], probs) == pytest.approx(best, abs=1e-6)
    assert all(c.status != "fail" for c in study.check())


def test_special_cases_reduce_to_the_expectation(farmer):
    base = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"],
                   scenarios=farmer_scenarios(farmer.DATA)).solve()
    for risk in ({"alpha": 0.9, "weight": 0.0}, {"alpha": 0.0, "weight": 1.0}):
        r = sl.lift(farmer.build_model, farmer.DATA, spec={"first_stage": ["acres_*"], "risk": risk},
                    scenarios=farmer_scenarios(farmer.DATA)).solve()
        assert r.rp == pytest.approx(base.rp, abs=1e-6) and r.vss == pytest.approx(base.vss, abs=1e-6)


def test_risk_averse_farmer(farmer, tmp_path):
    spec = {"first_stage": ["acres_*"], "risk": {"alpha": 0.8, "weight": 0.6},
            "scenarios": {"method": "distribution", "n": 40, "n_test": 60, "seed": 2, "correlation": 0.8,
                          "distributions": {"yield": {"dist": "normal", "cv": 0.2, "min": 0}}}}
    study = sl.lift(farmer.build_model, farmer.DATA, spec=spec)
    r = study.solve()
    assert r.ws <= r.rp + 1e-6 <= r.eev + 2e-6                  # still holds for the risk measure
    assert r.risk_parts["RP"]["cvar"] >= r.risk_parts["RP"]["mean"]
    assert "CVaR at 0.8" in r.objective
    assert all(c.status == "pass" for c in study.check() if c.name != "lift_note")

    o = study.out_of_sample()
    assert {"risk_gain", "risk_gain_ci95", "risk_ev", "risk_rp"} <= set(o)
    assert o["risk_gain_ci95"][0] <= o["risk_gain"] <= o["risk_gain_ci95"][1]
    assert o["risk_gain"] == pytest.approx(o["risk_ev"] - o["risk_rp"])

    rows = study.risk_frontier(weights=(0.0, 0.3, 0.6, 1.0))
    means, cvars = [f["mean"] for f in rows], [f["cvar"] for f in rows]
    assert all(b >= a - 1e-6 for a, b in zip(means, means[1:]))     # paying more on average ...
    assert all(b <= a + 1e-6 for a, b in zip(cvars, cvars[1:]))     # ... for a better tail
    assert "holdout_cvar" in rows[0]

    study.report(tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert "Risk-averse objective" in text and "Mean-risk trade-off" in text
    assert (tmp_path / "fig_risk_frontier.pdf").exists()
    assert "risk:" in (tmp_path / "uncertainty.yaml").read_text()
