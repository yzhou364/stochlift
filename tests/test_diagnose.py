"""When the stochastic program has no solution, the error says which scenarios and why."""
import pytest

import stochlift as sl
from stochlift.diagnose import NoCommonFirstStage, ScenarioInfeasible
from conftest import pulp_var


def capacity(d):
    """Build capacity x (first stage), serve demand with production y <= x; no shortfall allowed."""
    import pulp

    m = pulp.LpProblem("cap", pulp.LpMinimize)
    x, y = pulp_var(m, "cap", 0, 100), pulp_var(m, "prod", 0)
    m += 2 * x + y
    m += y <= x, "capacity"
    m += y >= d["demand"], "meet"
    return m


def test_scenarios_infeasible_on_their_own_are_named():
    pytest.importorskip("pulp")
    scen = [(0.25, {"demand": v}) for v in (40.0, 60.0, 130.0, 80.0)]
    study = sl.lift(capacity, {"demand": 50.0}, first_stage=["cap"], scenarios=scen)
    with pytest.raises(ScenarioInfeasible) as e:
        study.solve()
    assert e.value.scenarios == [2]
    text = str(e.value)
    assert "1 of 4 scenarios" in text and "demand = 130" in text and "slack variables" in text


def exact(d):
    """A first-stage order that must equal the demand: no recourse can absorb a difference."""
    import pulp

    m = pulp.LpProblem("exact", pulp.LpMinimize)
    x, y = pulp_var(m, "order", 0), pulp_var(m, "extra", 0, 5)
    m += x + 3 * y
    m += x + y == d["demand"], "balance"
    return m


def test_no_common_first_stage_names_a_conflicting_pair():
    pytest.importorskip("pulp")
    scen = [(0.25, {"demand": v}) for v in (50.0, 52.0, 70.0, 51.0)]
    study = sl.lift(exact, {"demand": 50.0}, first_stage=["order"], scenarios=scen)
    with pytest.raises(NoCommonFirstStage) as e:
        study.solve()
    assert e.value.scenarios == [0, 2]               # 50 and 70 differ by more than the recourse range 5
    assert "Scenarios 0 and 2" in str(e.value) and "demand = 70" in str(e.value)


def test_unbounded_scenario_is_explained():
    pytest.importorskip("pulp")
    import pulp

    def build(d):
        m = pulp.LpProblem("u", pulp.LpMaximize)
        x, y = pulp_var(m, "x", 0, 10), pulp_var(m, "y", 0)     # y has no upper bound
        m += x + d["price"] * y
        m += x + y >= 1, "minimum"
        return m

    study = sl.lift(build, {"price": 0.0}, first_stage=["x"],
                    scenarios=[(0.5, {"price": -1.0}), (0.5, {"price": 1.0})])
    with pytest.raises(ScenarioInfeasible, match="unbounded") as e:
        study.solve()
    assert e.value.scenarios == [1] and "add a bound" in str(e.value)
