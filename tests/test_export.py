"""Export to mpi-sppy: its own extensive-form solver must reproduce StochLift's RP."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import stochlift as sl
from stochlift.export import to_pyomo
from stochlift.model import solve
from conftest import farmer_scenarios

pytest.importorskip("mpisppy")
from stochlift.export import solve_ef_with_mpisppy  # noqa: E402


def test_pyomo_rebuild_matches_the_matrix(farmer):
    import pyomo.environ as pyo

    lm = sl.to_linear_model(farmer.build_model(farmer.DATA))
    back = sl.to_linear_model(to_pyomo(lm))
    assert solve(back).objective == pytest.approx(solve(lm).objective, abs=1e-6)
    assert back.n == lm.n and back.m == lm.m
    assert pyo.SolverFactory("appsi_highs").available(exception_flag=False)


def test_farmer_matches_mpisppy(farmer):
    study = sl.lift(farmer.build_model, farmer.DATA, first_stage=["acres_*"],
                    scenarios=farmer_scenarios(farmer.DATA))
    r = study.solve()
    obj, x = solve_ef_with_mpisppy(study)
    assert obj == pytest.approx(r.rp, abs=1e-6) and -obj == pytest.approx(108390, abs=0.5)
    assert x == pytest.approx(r.x_rp, abs=1e-6)


def test_binary_first_stage_matches_mpisppy(facility, facility_history):
    hist = pd.read_csv(facility_history).head(80)
    study = sl.lift(facility.build_model, facility.DATA, history=hist,
                    spec={"first_stage": ["open[*]"], "uncertain": ["demand"],
                          "scenarios": {"method": "kmeans", "n": 10, "seed": 1}})
    r = study.solve()
    obj, x = solve_ef_with_mpisppy(study)
    assert obj == pytest.approx(r.rp, rel=1e-6)
    assert x == pytest.approx(r.x_rp, abs=1e-6)


def test_risk_is_not_exported(farmer):
    study = sl.lift(farmer.build_model, farmer.DATA, scenarios=farmer_scenarios(farmer.DATA),
                    spec={"first_stage": ["acres_*"], "risk": {"alpha": 0.9, "weight": 0.5}})
    with pytest.raises(NotImplementedError, match="CVaR"):
        study.to_mpisppy()


def test_export_module_runs_in_generic_cylinders(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    farmer_dir = os.path.join(root, "examples", "farmer")
    spec = tmp_path / "u.yaml"
    spec.write_text(open(os.path.join(farmer_dir, "uncertainty_distribution.yaml")).read()
                    .replace("n: 100", "n: 6").replace("n_test: 500", "n_test: 0"))
    from stochlift.cli import main

    out = tmp_path / "farmer_scen.py"
    assert main(["export", os.path.join(farmer_dir, "model.py"), "--spec", str(spec), "-o", str(out)]) == 0
    run = subprocess.run([sys.executable, "-m", "mpisppy.generic_cylinders", "--module-name", "farmer_scen",
                          "--num-scens", "6", "--EF", "--EF-solver-name", "appsi_highs"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    study = sl.lift(*sl_load(farmer_dir), spec=str(spec))
    rp = study.solve().rp
    found = [float(tok.replace(",", "")) for line in (run.stdout + run.stderr).splitlines()
             if "EF objective" in line for tok in line.split() if _is_number(tok)]
    assert found and min(abs(v - rp) for v in found) <= 1e-4 * abs(rp)


def sl_load(folder):
    from stochlift.cli import load_model

    return load_model(os.path.join(folder, "model.py"))


def _is_number(tok):
    try:
        float(tok.replace(",", ""))
        return True
    except ValueError:
        return False
