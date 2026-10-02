"""Regression tests for bugs found by an independent review of v0.1."""
import numpy as np
import pandas as pd
import pytest

import stochlift as sl
from stochlift.llm import parse_spec, validate
from conftest import pulp_var


def capacity_model(d):
    pulp = pytest.importorskip("pulp")
    m = pulp.LpProblem("p", pulp.LpMinimize)
    x = pulp_var(m, "cap", 0, 10)
    y = pulp_var(m, "buy", 0)
    m += x + 5 * y
    m += x + y >= d["demand"], "meet"
    return m


def test_first_stage_variable_missing_from_the_mean_model():
    """x has coefficient 0 at the mean, so Pyomo drops it there; it must still be fixed in EEV."""
    pyo = pytest.importorskip("pyomo.environ")

    def build(d):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.x2 = pyo.Var(bounds=(0, 10))
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.o = pyo.Objective(expr=m.y + m.x2)
        m.c = pyo.Constraint(expr=m.y >= d["a"] * m.x + 1)
        return m

    study = sl.lift(build, {"a": 0.0}, first_stage=["x", "x2"],
                    scenarios=[(0.5, {"a": 1.0}), (0.5, {"a": -1.0})])
    r = study.solve()
    assert r.vss >= 0 and r.evpi >= 0 and r.eev >= r.rp - 1e-9
    assert set(r.x_ev) == {"x", "x2"} and r.x_ev["x"] == 0.0
    checks = {c.name: c for c in study.check()}
    assert checks["bound_ordering"].status == "pass"
    assert any(c.name == "lift_note" and "mean-value model" in c.detail for c in study.checks)


def test_report_when_every_holdout_gain_is_equal(tmp_path):
    h = pd.DataFrame({"demand": np.random.default_rng(0).uniform(20, 40, 40)})
    study = sl.lift(capacity_model, {"demand": 30.0}, history=h,
                    spec={"first_stage": ["cap"], "uncertain": ["demand"], "holdout": 0.25})
    r = study.solve()
    assert r.vss == 0.0
    study.out_of_sample()
    text = open(study.report(tmp_path)).read()
    assert "**No.**" in text and (tmp_path / "fig_gain.pdf").exists()


def test_missing_values_are_rejected_everywhere():
    h = pd.DataFrame({"demand": np.random.default_rng(0).uniform(20, 40, 40)})
    h.loc[35, "demand"] = np.nan                     # in the hold-out part
    spec = {"first_stage": ["cap"], "uncertain": ["demand"], "holdout": 0.25}
    with pytest.raises(ValueError, match="missing values"):
        sl.lift(capacity_model, {"demand": 30.0}, history=h, spec=spec)
    study = sl.lift(capacity_model, {"demand": 30.0}, history=h.dropna(), spec=spec)
    with pytest.raises(ValueError, match="missing values"):
        study.out_of_sample(observations=[[25.0], [np.nan]])


def test_observations_must_match_the_uncertain_entries(facility, facility_history):
    hist = pd.read_csv(facility_history)
    study = sl.lift(facility.build_model, facility.DATA, history=hist.head(40),
                    spec={"first_stage": ["open[*]"], "uncertain": ["demand"]})
    new = hist.iloc[100:120]
    right = study.out_of_sample(observations=new[facility.CUSTOMERS].to_numpy())["mean_rp"]
    assert study.out_of_sample(observations=new)["mean_rp"] == pytest.approx(right)   # picks columns by name
    with pytest.raises(ValueError, match="columns"):
        study.out_of_sample(observations=new[facility.CUSTOMERS[:3]].to_numpy())
    with pytest.raises(ValueError, match="one per uncertain entry"):
        study.model_for([1.0, 2.0])


def test_probe_touches_only_the_uncertain_entries():
    pulp = pytest.importorskip("pulp")

    def build(d):
        m = pulp.LpProblem("p", pulp.LpMinimize)
        x = pulp_var(m, "cap", 0, 100)
        ys = [pulp_var(m, f"buy_{t}", 0) for t in range(d["cfg"]["periods"])]
        m += x + 5 * pulp.lpSum(ys)
        for t, y in enumerate(ys):
            m += x + y >= d["cfg"]["demand"], f"meet_{t}"
        return m

    data = {"cfg": {"periods": 3, "demand": 30.0, "unused": 1.0}}
    scen = lambda key: [(0.5, {"cfg": {key: 20.0}}), (0.5, {"cfg": {key: 40.0}})]   # noqa: E731
    ok = sl.lift(build, data, first_stage=["cap"], scenarios=scen("demand"))
    assert {c.name: c for c in ok.check()}["uncertainty_reaches_model"].status == "pass"
    dead = sl.lift(build, data, first_stage=["cap"], scenarios=scen("unused"))
    assert {c.name: c for c in dead.check()}["uncertainty_reaches_model"].status == "fail"
    assert validate(build, data, sl.Spec(first_stage=["cap"], uncertain=["cfg.demand"])) == []
    assert any("do not enter the model" in p
               for p in validate(build, data, sl.Spec(first_stage=["cap"], uncertain=["cfg.unused"])))


def test_history_with_integer_column_labels():
    pulp = pytest.importorskip("pulp")

    def build(d):
        m = pulp.LpProblem("p", pulp.LpMinimize)
        x = pulp_var(m, "cap", 0, 100)
        m += x + 5 * pulp.lpSum(pulp_var(m, f"buy_{t}", 0) for t in range(3))
        for t in range(3):
            m += x + m.variablesDict()[f"buy_{t}"] >= d["demand"][t], f"meet_{t}"
        return m

    hist = pd.DataFrame(np.random.default_rng(1).uniform(20, 40, size=(12, 3)))
    study = sl.lift(build, {"demand": [30.0, 30.0, 30.0]}, history=hist,
                    spec={"first_stage": ["cap"], "uncertain": ["demand"]})
    assert len(study.scenarios) == 12 and study.solve().vss >= 0


def test_gurobi_unsupported_objectives_are_refused():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(params={"OutputFlag": 0})
    except gp.GurobiError:
        pytest.skip("no Gurobi license")
    m = gp.Model(env=env)
    x = m.addVar(ub=4, name="x")
    m.addConstr(x >= 1)
    m.setPWLObj(x, [0, 2, 4], [0, 10, 12])
    with pytest.raises(sl.UnsupportedModel):
        sl.to_linear_model(m)


def test_highspy_partly_named_model():
    highspy = pytest.importorskip("highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addVariable(lb=0, ub=4, name="cap")
    h.addVariable(lb=0, ub=4)
    h.addVariable(lb=0, ub=4)
    lm = sl.to_linear_model(h)
    assert lm.names[0] == "cap" and len(set(lm.names)) == 3 and all(lm.names)


def test_parse_spec_with_other_fenced_blocks_and_real_errors():
    reply = "```python\nx = 1\n```\nSpec:\n```yaml\nfirst_stage: ['open[*]']\nuncertain: [demand]\nholdout:\n```"
    spec = parse_spec(reply)
    assert spec.first_stage == ["open[*]"] and spec.holdout == 0.0
    with pytest.raises(ValueError, match="unknown spec fields"):
        parse_spec("```yaml\nfirst_stage: [a]\nstages: 2\n```")


def test_zero_dimensional_array_in_data():
    h = pd.DataFrame({"demand": np.random.default_rng(0).uniform(20, 40, 12)})
    study = sl.lift(capacity_model, {"c": np.array(1.0), "demand": 30.0}, history=h,
                    spec={"first_stage": ["cap"], "uncertain": ["demand"]})
    assert study.solve().n_scenarios == 12


def test_vss_and_evpi_are_never_negative():
    rng = np.random.default_rng(7)
    for _ in range(25):
        d = rng.uniform(5, 50, size=4)
        p = rng.dirichlet(np.ones(4))
        r = sl.lift(capacity_model, {"demand": 30.0}, first_stage=["cap"],
                    scenarios=[(pi, {"demand": di}) for pi, di in zip(p, d)]).solve()
        assert r.vss >= 0 and r.evpi >= 0
