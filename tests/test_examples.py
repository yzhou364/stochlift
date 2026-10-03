"""The OR-Tools and gurobipy examples run, pass the checks and match known answers."""
import os

import pytest

import stochlift as sl
from conftest import ROOT, _load


def test_newsvendor_ortools_matches_the_critical_ratio():
    # in a fresh process: on Linux some OR-Tools builds cannot load once highspy is loaded
    pytest.importorskip("ortools")
    import subprocess
    import sys

    folder = os.path.join(ROOT, "examples", "newsvendor_ortools")
    script = f"""
import sys, numpy as np
sys.path.insert(0, {folder!r})
from model import DATA, build_model          # OR-Tools first, as in a user's script
import stochlift as sl
spec = sl.Spec.load({os.path.join(folder, "uncertainty.yaml")!r})
spec.scenarios.update(n=101, n_test=0)
study = sl.lift(build_model, DATA, spec=spec)
r = study.solve()
d = np.sort(study.scenarios.values[:, 0])
ratio = (DATA["price"] - DATA["cost"]) / (DATA["price"] - DATA["salvage"])
k = int(np.ceil(ratio * len(d))) - 1
assert d[k] - 1e-6 <= r.x_rp["order"] <= d[min(k + 1, len(d) - 1)] + 1e-6, (r.x_rp, d[k])
assert all(c.status == "pass" for c in study.check())
print("ok")
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                         env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert run.returncode == 0 and "ok" in run.stdout, run.stdout[-2000:] + run.stderr[-2000:]


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
