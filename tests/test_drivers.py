"""How the value of the stochastic solution depends on how much, and which, data are uncertain."""
import numpy as np
import pytest

import stochlift as sl
from conftest import farmer_scenarios


@pytest.fixture(scope="module")
def study(farmer):
    s = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"], scenarios=farmer_scenarios(farmer.DATA))
    s.solve()
    return s


def test_sweep_endpoints(study):
    rows = study.uncertainty_sweep(scales=(0.0, 1.0, 2.0))
    zero, one, two = rows
    assert zero["VSS"] == 0.0 and zero["EVPI"] == pytest.approx(0.0, abs=1e-6)   # no uncertainty, no value
    assert one["VSS"] == pytest.approx(study.results.vss, abs=1e-6)               # as specified: the study itself
    assert one["EVPI"] == pytest.approx(study.results.evpi, abs=1e-6)
    assert two["EVPI"] >= one["EVPI"] - 1e-6


def test_scaled_scenarios_match_a_direct_lift(farmer, study):
    """Scale 1.5 equals lifting the farmer with yields 1 +/- 0.3 instead of 1 +/- 0.2."""
    row = study.uncertainty_sweep(scales=(1.5,))[0]
    scen = [(1 / 3, {"yield": {c: f * y for c, y in farmer.DATA["yield"].items()}}) for f in (1.3, 1.0, 0.7)]
    direct = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"], scenarios=scen).solve()
    assert row["RP"] == pytest.approx(direct.rp, abs=1e-6) and row["VSS"] == pytest.approx(direct.vss, abs=1e-6)


def test_value_by_parameter(study):
    rows = study.value_by_parameter()
    groups = [r["group"] for r in rows]
    assert groups[-1].startswith("all") and set(groups[:-1]) == {"yield.wheat", "yield.corn", "yield.beets"}
    assert rows[-1]["VSS"] == pytest.approx(study.results.vss, abs=1e-6)
    evpi = [r["EVPI"] for r in rows[:-1]]
    assert evpi == sorted(evpi, reverse=True)
    # one group at a time can never exceed perfect information about everything
    assert max(evpi) <= rows[-1]["EVPI"] + 1e-6
    by_key = study.value_by_parameter(level="key")
    assert [r["group"] for r in by_key] == ["yield", "all (as specified)"]
    assert by_key[0]["VSS"] == pytest.approx(study.results.vss, abs=1e-6)


def test_infeasible_scales_are_reported_not_raised(farmer):
    """Scaling past what the data allow (negative yields) gives nan, not an exception."""
    import pulp  # noqa: F401
    from conftest import pulp_var

    def build(d):
        import pulp

        m = pulp.LpProblem("cap", pulp.LpMinimize)
        x, y = pulp_var(m, "cap", 0, 100), pulp_var(m, "prod", 0)
        m += 2 * x + y
        m += y <= x, "capacity"
        m += y >= d["demand"], "meet"
        return m

    s = sl.lift(build, {"demand": 50.0}, first_stage=["cap"],
                scenarios=[(0.5, {"demand": 40.0}), (0.5, {"demand": 90.0})])
    s.solve()
    rows = s.uncertainty_sweep(scales=(1.0, 3.0))           # 3x: demand 140 > capacity bound 100
    assert rows[0]["VSS"] == s.results.vss == float("inf")   # mean-value decision has no recourse at 90
    assert np.isnan(rows[1]["VSS"]) and "infeasible" in rows[1]["status"]
