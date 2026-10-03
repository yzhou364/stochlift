"""The OR-Tools and gurobipy examples run, pass the checks and match known answers."""
import os

import numpy as np
import pytest

import stochlift as sl
from conftest import ROOT, _load


def test_newsvendor_ortools_matches_the_critical_ratio():
    pytest.importorskip("ortools")
    folder = os.path.join(ROOT, "examples", "newsvendor_ortools")
    mod = _load(os.path.join(folder, "model.py"), "newsvendor_ortools_model")
    spec = sl.Spec.load(os.path.join(folder, "uncertainty.yaml"))
    spec.scenarios.update(n=101, n_test=0)
    study = sl.lift(mod.build_model, mod.DATA, spec=spec)
    r = study.solve()
    d = np.sort(study.scenarios.values[:, 0])
    ratio = (mod.DATA["price"] - mod.DATA["cost"]) / (mod.DATA["price"] - mod.DATA["salvage"])
    k = int(np.ceil(ratio * len(d))) - 1
    assert d[k] - 1e-6 <= r.x_rp["order"] <= d[min(k + 1, len(d) - 1)] + 1e-6
    assert all(c.status == "pass" for c in study.check())


def test_capacity_expansion_gurobi():
    pytest.importorskip("gurobipy")
    folder = os.path.join(ROOT, "examples", "capacity_expansion_gurobi")
    mod = _load(os.path.join(folder, "model.py"), "capacity_gurobi_model")
    spec = sl.Spec.load(os.path.join(folder, "uncertainty.yaml"))
    spec.scenarios.update(n=30, n_test=0)
    study = sl.lift(mod.build_model, mod.DATA, spec=spec)
    r = study.solve()
    assert r.ws <= r.rp + 1e-6 * abs(r.rp) and r.rp <= r.eev + 1e-6 * abs(r.rp)
    assert all(c.status == "pass" for c in study.check())
    assert r.x_rp["cap[peak]"] > r.x_ev["cap[peak]"]          # hedging against high load needs peakers
