"""The same small model in every supported library must give the same LinearModel answer.

    max 3x + 2y - z + 7   s.t.  x + y <= 5.5,  x - z >= -1,  0 <= x <= 4,  y >= 0 integer,  -3 <= z <= 3
    optimum 24.5 at x = 3.5, y = 2, z = -3
"""
import pytest

import stochlift as sl
from conftest import pulp_var

EXPECTED = 24.5


def _check(native):
    lm = sl.to_linear_model(native)
    assert lm.sense == "max" and lm.n == 3 and lm.m == 2 and lm.integer.sum() == 1
    sol = sl.solve(lm)
    assert sol.ok
    assert lm.sign * sol.objective == pytest.approx(EXPECTED)
    return lm


def test_pulp():
    pulp = pytest.importorskip("pulp")
    p = pulp.LpProblem("t", pulp.LpMaximize)
    x = pulp_var(p, "x", 0, 4)
    y = pulp_var(p, "y", 0, None, cat="Integer")
    z = pulp_var(p, "z", -3, 3)
    p += 3 * x + 2 * y - z + 7
    p += x + y <= 5.5, "cap"
    p += x - z >= -1, "link"
    _check(p)


def test_pyomo():
    pyo = pytest.importorskip("pyomo.environ")
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.y = pyo.Var(domain=pyo.NonNegativeIntegers)
    m.z = pyo.Var(bounds=(-3, 3))
    m.obj = pyo.Objective(expr=3 * m.x + 2 * m.y - m.z + 7, sense=pyo.maximize)
    m.cap = pyo.Constraint(expr=m.x + m.y <= 5.5)
    m.link = pyo.Constraint(expr=m.x - m.z >= -1)
    _check(m)


def test_pyomo_rejects_nonlinear():
    pyo = pytest.importorskip("pyomo.environ")
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    m.obj = pyo.Objective(expr=m.x * m.x)
    with pytest.raises(sl.UnsupportedModel):
        sl.to_linear_model(m)


def test_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(params={"OutputFlag": 0})
    except gp.GurobiError:
        pytest.skip("no Gurobi license")
    m = gp.Model(env=env)
    x = m.addVar(lb=0, ub=4, name="x")
    y = m.addVar(lb=0, vtype=gp.GRB.INTEGER, name="y")
    z = m.addVar(lb=-3, ub=3, name="z")
    m.setObjective(3 * x + 2 * y - z + 7, gp.GRB.MAXIMIZE)
    m.addConstr(x + y <= 5.5, name="cap")
    m.addConstr(x - z >= -1, name="link")
    _check(m)


ORTOOLS_SCRIPT = """
from ortools.linear_solver import pywraplp
import stochlift as sl
from stochlift.model import _backend

s = pywraplp.Solver.CreateSolver("GLOP")
x = s.NumVar(0, 4, "x"); y = s.IntVar(0, s.infinity(), "y"); z = s.NumVar(-3, 3, "z")
s.Maximize(3 * x + 2 * y - z + 7)
s.Add(x + y <= 5.5); s.Add(x - z >= -1)
lm = sl.to_linear_model(s)
assert lm.sense == "max" and lm.n == 3 and lm.m == 2 and lm.integer.sum() == 1
sol = sl.solve(lm)
assert sol.ok and abs(lm.sign * sol.objective - 24.5) < 1e-7, sol
print("ok", _backend())
"""


def test_ortools():
    """Run in a fresh process: some OR-Tools builds cannot share a process with highspy,
    in which case StochLift falls back to SciPy's bundled HiGHS."""
    import subprocess
    import sys

    pytest.importorskip("ortools")
    out = subprocess.run([sys.executable, "-c", ORTOOLS_SCRIPT], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.startswith("ok")


def test_scipy_backend_matches_highspy(monkeypatch):
    pulp = pytest.importorskip("pulp")
    import stochlift.model as model

    p = pulp.LpProblem("t", pulp.LpMaximize)
    x = pulp_var(p, "x", 0, 4)
    y = pulp_var(p, "y", 0, None, cat="Integer")
    z = pulp_var(p, "z", -3, 3)
    p += 3 * x + 2 * y - z + 7
    p += x + y <= 5.5, "cap"
    p += x - z >= -1, "link"
    lm = sl.to_linear_model(p)
    monkeypatch.setattr(model, "_BACKEND", "scipy")
    sol = sl.solve(lm)
    assert sol.ok and lm.sign * sol.objective == pytest.approx(EXPECTED)
    assert sl.solve(lm, fixed={lm.index()["x"]: 9.0}).status == "infeasible"
    infeasible = sl.solve(sl.LinearModel(names=["a"], c=[1.0], offset=0.0, lb=[0.0], ub=[1.0], integer=[False],
                                         A=[[1.0]], row_lb=[2.0], row_ub=[float("inf")]))
    assert infeasible.status == "infeasible"


def test_highspy_and_file(tmp_path):
    highspy = pytest.importorskip("highspy")
    lp = tmp_path / "m.lp"
    lp.write_text("Maximize\n obj: 3 x + 2 y - z + 7\nSubject To\n cap: x + y <= 5.5\n link: x - z >= -1\n"
                  "Bounds\n 0 <= x <= 4\n -3 <= z <= 3\nGeneral\n y\nEnd\n")
    _check(str(lp))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(lp))
    _check(h)


def test_unknown_type():
    with pytest.raises(sl.UnsupportedModel):
        sl.to_linear_model(object())
