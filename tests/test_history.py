"""Scenarios from a history, hold-out evaluation, stability, and the written report."""
import numpy as np
import pandas as pd
import pytest

import stochlift as sl
from stochlift.scenarios import from_history, match_columns


def test_kmeans_keeps_the_mean_and_weights():
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(200, 4))
    S = from_history(X, [("d", i) for i in range(4)], method="kmeans", n=12, seed=5)
    assert len(S) <= 12 and S.probs.sum() == pytest.approx(1.0)
    assert S.mean() == pytest.approx(X.mean(axis=0))
    E = from_history(X, [("d", i) for i in range(4)], method="empirical")
    assert len(E) == 200
    with pytest.raises(ValueError, match="missing values"):
        from_history(np.array([[1.0, np.nan]]), [("a",), ("b",)])


def test_column_matching():
    paths = [("demand", "C1"), ("demand", "C2")]
    assert match_columns(paths, ["week", "C1", "C2"]) == ["C1", "C2"]
    assert match_columns(paths, ["demand.C1", "demand.C2"]) == ["demand.C1", "demand.C2"]
    assert match_columns(paths, ["a", "b"], {"demand.C1": "a", "demand.C2": "b"}) == ["a", "b"]
    with pytest.raises(KeyError, match="no history column"):
        match_columns(paths, ["x", "y"])


def test_history_workflow(facility, facility_history, tmp_path):
    hist = pd.read_csv(facility_history).head(90)
    spec = {"first_stage": ["open[*]"], "uncertain": ["demand"], "holdout": 1 / 3,
            "scenarios": {"method": "kmeans", "n": 12, "seed": 0}}
    study = sl.lift(facility.build_model, facility.DATA, spec=spec, history=hist)
    assert len(study.train) == 60 and len(study.test) == 30 and len(study.scenarios) <= 12
    r = study.solve()
    assert r.ws <= r.rp + 1e-6 <= r.eev + 2e-6
    assert all(c.status == "pass" for c in study.check())

    o = study.out_of_sample()
    assert o["n"] == 30 and o["n_compared"] == 30
    assert o["gain_ci95"][0] <= o["mean_gain"] <= o["gain_ci95"][1]
    assert o["mean_gain"] == pytest.approx(o["mean_ev"] - o["mean_rp"])

    rows = study.stability(sizes=(4, 8), reps=3)
    assert len(rows) == 6 and {"n", "in_sample", "holdout"} <= set(rows[0])
    gap = study.saa_gap(n=10, batches=4)
    assert gap["mean_gap"] >= -1e-6 and gap["upper95"] >= gap["mean_gap"]

    study.report(tmp_path)
    for name in ("summary.md", "fig_out_of_sample.pdf", "fig_gain.pdf", "fig_stability.pdf"):
        assert (tmp_path / name).exists(), name
    assert "Out-of-sample test" in (tmp_path / "summary.md").read_text()


def test_holdout_rows_never_build_scenarios(facility, facility_history):
    hist = pd.read_csv(facility_history).head(40)
    spec = {"first_stage": ["open[*]"], "uncertain": ["demand"], "holdout": 0.25}
    study = sl.lift(facility.build_model, facility.DATA, spec=spec, history=hist)
    cols = list(facility.CUSTOMERS)
    assert np.array_equal(study.scenarios.values, hist[cols].to_numpy()[:30])
    assert np.array_equal(study.test, hist[cols].to_numpy()[30:])
