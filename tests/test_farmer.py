"""The farmer problem must reproduce the values published by Birge and Louveaux."""
import copy

import pytest

import stochlift as sl
from conftest import pulp_var, farmer_scenarios


def _study(farmer, **kw):
    return sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"],
                   scenarios=farmer_scenarios(farmer.DATA), **kw)


def test_textbook_values(farmer):
    r = _study(farmer).solve()
    assert r.sense == "min"
    assert -r.rp == pytest.approx(108390, abs=0.5)
    assert -r.ws == pytest.approx(115405.56, abs=0.5)
    assert -r.eev == pytest.approx(107240, abs=0.5)
    assert -r.ev == pytest.approx(118600, abs=0.5)
    assert r.vss == pytest.approx(1150, abs=0.5)
    assert r.evpi == pytest.approx(7015.56, abs=0.5)
    assert r.x_rp == pytest.approx({"acres_wheat": 170, "acres_corn": 80, "acres_beets": 250})
    assert r.x_ev == pytest.approx({"acres_wheat": 120, "acres_corn": 80, "acres_beets": 300})


def test_all_checks_pass(farmer):
    checks = _study(farmer).check()
    assert {c.name for c in checks} >= {"builder_deterministic", "uncertainty_reaches_model", "stage_split",
                                        "single_scenario_reduction", "bound_ordering",
                                        "decomposition_consistency", "mean_value_recourse"}
    assert all(c.status == "pass" for c in checks), [str(c) for c in checks]


def test_maximization_gives_the_same_values(farmer):
    """Writing the model as profit maximization must not change VSS or EVPI."""
    import pulp

    def build_profit(data):
        m = farmer.build_model(data)
        m.setObjective(-m.objective)
        m.sense = pulp.LpMaximize
        return m

    r = sl.lift(build_profit, farmer.DATA, first_stage=["acres_*"],
                scenarios=farmer_scenarios(farmer.DATA)).solve()
    assert r.sense == "max"
    assert r.rp == pytest.approx(108390, abs=0.5) and r.ws == pytest.approx(115405.56, abs=0.5)
    assert r.vss == pytest.approx(1150, abs=0.5) and r.evpi == pytest.approx(7015.56, abs=0.5)


def test_report_files(farmer, tmp_path):
    study = _study(farmer)
    path = study.report(tmp_path)
    text = open(path).read()
    assert "Yes, on the scenario set" in text and "108,390" in text
    for name in ("results.json", "table_values.tex", "uncertainty.yaml", "fig_value.pdf",
                 "fig_first_stage.pdf", "fig_scenarios.png"):
        assert (tmp_path / name).exists(), name


def test_wrong_specs_are_caught(farmer):
    scen = farmer_scenarios(farmer.DATA)
    # a data key that never enters the model cannot be uncertain
    data = copy.deepcopy(farmer.DATA)
    data["unused"] = {"a": 1.0}
    dead = sl.lift(farmer.build_model, data, first_stage=["acres_*"],
                   scenarios=[(0.5, {"unused": {"a": 1.0}}), (0.5, {"unused": {"a": 2.0}})])
    by_name = {c.name: c for c in dead.check()}
    assert by_name["uncertainty_reaches_model"].status == "fail"
    # every variable first-stage: not a two-stage model
    everything = sl.lift(farmer.build_model, farmer.DATA, first_stage=["*"], scenarios=scen)
    assert {c.name: c for c in everything.check()}["stage_split"].status == "fail"
    # a pattern that matches nothing
    with pytest.raises(ValueError, match="match no variable"):
        sl.lift(farmer.build_model, farmer.DATA, first_stage=["hectares_*"], scenarios=scen).solve()


def test_no_recourse_means_infinite_eev():
    """Without a shortage variable the mean-value order cannot serve a high-demand scenario."""
    pulp = pytest.importorskip("pulp")

    def build(data):
        m = pulp.LpProblem("order", pulp.LpMinimize)
        order = pulp_var(m, "order", 0)
        extra = pulp_var(m, "sold", 0)
        m += 2 * order - 3 * extra
        m += order >= data["demand"], "must_cover"
        m += extra <= data["demand"], "sales"
        m += extra <= order, "stock"
        return m

    study = sl.lift(build, {"demand": 10.0}, first_stage=["order"],
                    scenarios=[(0.5, {"demand": 5.0}), (0.5, {"demand": 15.0})])
    r = study.solve()
    assert r.eev_infeasible == 1 and r.eev == float("inf") and r.x_rp["order"] == pytest.approx(15)
    checks = {c.name: c for c in study.check()}
    assert checks["mean_value_recourse"].status == "warn"
    assert checks["first_stage_rows_stable"].status == "warn"   # 'must_cover' is first-stage only
    from stochlift.report import verdict
    assert verdict(study).startswith("**Yes.**")
