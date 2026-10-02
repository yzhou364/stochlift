"""Scenarios sampled from distributions when there is no history."""
import numpy as np
import pytest

import stochlift as sl
from stochlift.distributions import DistributionSampler, marginal
from stochlift.llm import validate

U = (np.arange(200_000) + 0.5) / 200_000          # a fine grid of probabilities


def _moments(params, nominal):
    x = marginal(params, nominal).ppf(U)
    return x.mean(), x.std(), x.min(), x.max()


def test_marginals_match_their_parameters():
    m, s, _, _ = _moments({"dist": "normal", "cv": 0.2}, 50.0)
    assert m == pytest.approx(50.0, rel=1e-3) and s == pytest.approx(10.0, rel=1e-2)
    m, s, lo, _ = _moments({"dist": "lognormal", "cv": 0.5}, 40.0)
    assert m == pytest.approx(40.0, rel=1e-2) and s == pytest.approx(20.0, rel=3e-2) and lo > 0
    m, _, lo, hi = _moments({"dist": "uniform", "spread": 0.25}, 8.0)
    assert m == pytest.approx(8.0, rel=1e-6) and lo == pytest.approx(6.0, abs=1e-3) and hi == pytest.approx(10.0, abs=1e-3)
    m, _, lo, hi = _moments({"dist": "triangular", "low": 0, "mode": 3, "high": 6}, 1.0)
    assert m == pytest.approx(3.0, rel=1e-4) and lo >= 0 and hi <= 6
    x = marginal({"dist": "discrete", "factors": [0.8, 1.0, 1.2], "probs": [0.2, 0.5, 0.3]}, 10.0).ppf(U)
    assert sorted(set(np.round(x, 9))) == [8.0, 10.0, 12.0]
    assert [np.mean(np.isclose(x, v)) for v in (8, 10, 12)] == pytest.approx([0.2, 0.5, 0.3], abs=1e-4)
    _, _, lo, _ = _moments({"dist": "normal", "cv": 1.0, "min": 0}, 5.0)
    assert lo == 0.0


@pytest.mark.parametrize("params, message", [
    ({"dist": "gamma"}, "'dist' must be one of"),
    ({"dist": "normal"}, "needs 'sd' or 'cv'"),
    ({"dist": "normal", "cv": 0.1, "scale": 2}, "unknown distribution parameters"),
    ({"dist": "uniform"}, "needs 'low'"),
    ({"dist": "triangular", "low": 0, "high": 1, "mode": 2}, "low <= mode <= high"),
    ({"dist": "discrete", "values": [1, 2], "probs": [0.5]}, "as many probs as values"),
    ({"dist": "lognormal", "mean": -1, "cv": 0.1}, "mean > 0"),
])
def test_bad_parameters_are_rejected(params, message):
    with pytest.raises(ValueError, match=message):
        marginal(params, 1.0)


def test_specific_keys_override_groups_in_any_order():
    data = {"d": {"a": 10.0, "b": 20.0}}
    for dists in ({"d": {"dist": "normal", "cv": 0.1}, "d.b": {"dist": "uniform", "spread": 0.5}},
                  {"d.b": {"dist": "uniform", "spread": 0.5}, "d": {"dist": "normal", "cv": 0.1}}):
        s = DistributionSampler(data, dists)
        assert s.names == (["d.a", "d.b"] if list(dists)[0] == "d" else ["d.b", "d.a"])
        kinds = dict(zip(s.names, (m.name for m in s.marginals)))
        assert kinds == {"d.a": "normal", "d.b": "uniform"}
    with pytest.raises(KeyError, match="matches no numeric entry"):
        DistributionSampler(data, {"nope": {"dist": "normal", "cv": 0.1}})


def test_common_correlation():
    data = {"d": [1.0, 1.0, 1.0, 1.0]}
    for rho in (0.0, 0.7, -0.3):
        s = DistributionSampler(data, {"d": {"dist": "normal", "sd": 1.0}}, correlation=rho)
        X = s.sample(40_000, np.random.default_rng(1))
        C = np.corrcoef(X.T)
        assert C[np.triu_indices(4, 1)] == pytest.approx(rho, abs=0.02)
    X = DistributionSampler(data, {"d": {"dist": "uniform", "spread": 0.2}}, correlation=1.0).sample(
        50, np.random.default_rng(0))
    assert np.allclose(X, X[:, :1])                  # perfectly co-moving
    with pytest.raises(ValueError, match="correlation must be between"):
        DistributionSampler(data, {"d": {"dist": "normal", "sd": 1}}, correlation=-0.5)


def test_sampling_is_reproducible():
    data = {"d": {"a": 5.0}}
    a = sl.sample(data, {"d": {"dist": "lognormal", "cv": 0.3}}, n=20, seed=3)
    b = sl.sample(data, {"d": {"dist": "lognormal", "cv": 0.3}}, n=20, seed=3)
    assert np.array_equal(a.values, b.values) and a.probs.sum() == pytest.approx(1.0)


FARMER_SPEC = {
    "first_stage": ["acres_*"],
    "scenarios": {"method": "distribution", "n": 30, "seed": 0, "n_test": 40, "correlation": 0.8,
                  "distributions": {"yield": {"dist": "normal", "cv": 0.15, "min": 0}}},
}


def test_farmer_from_distributions(farmer, tmp_path):
    study = sl.lift(farmer.build_model, farmer.DATA, spec=FARMER_SPEC)
    assert study.spec.uncertain == ["yield"] and len(study.scenarios) == 30 and len(study.test) == 40
    r = study.solve()
    assert r.ws <= r.rp + 1e-6 <= r.eev + 2e-6          # min form: net cost
    assert all(c.status != "fail" for c in study.check())

    # the spec route and the sl.sample route give the same scenarios and the same answer
    S = sl.sample(farmer.DATA, FARMER_SPEC["scenarios"]["distributions"], n=30, seed=0, correlation=0.8)
    assert np.array_equal(S.values, study.scenarios.values)
    r2 = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"], scenarios=S).solve()
    assert r2.rp == pytest.approx(r.rp, rel=1e-9)

    o = study.out_of_sample()
    assert o["n"] == 40 and o["source"] == "independent samples from the distributions"
    rows = study.stability(sizes=(5, 10), reps=2)
    assert len(rows) == 4 and "holdout" in rows[0]
    gap = study.saa_gap(n=10, batches=3)
    assert gap["mean_gap"] >= -1e-6
    study.review(show=False)
    study.report(tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert "independent samples from the distributions" in text
    assert "method: distribution" in (tmp_path / "uncertainty.yaml").read_text()


def test_distribution_spec_errors(farmer):
    with pytest.raises(ValueError, match="does not use a history"):
        sl.lift(farmer.build_model, farmer.DATA, spec=FARMER_SPEC, history="x.csv")
    bad = {**FARMER_SPEC, "uncertain": ["yield", "sell_price"]}
    with pytest.raises(ValueError, match="have no distribution"):
        sl.lift(farmer.build_model, farmer.DATA, spec=bad)
    with pytest.raises(ValueError, match="no source of scenarios"):
        sl.lift(farmer.build_model, farmer.DATA, spec={"first_stage": ["acres_*"], "uncertain": ["yield"]})

    ok = sl.Spec.from_dict(FARMER_SPEC)
    assert validate(farmer.build_model, farmer.DATA, ok) == []
    broken = sl.Spec.from_dict({**FARMER_SPEC, "scenarios": {
        "method": "distribution", "distributions": {"yield": {"dist": "normal"}}}})
    assert any("needs 'sd' or 'cv'" in p for p in validate(farmer.build_model, farmer.DATA, broken))
