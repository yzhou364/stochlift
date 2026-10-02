"""Hand a lifted study to mpi-sppy for decomposition (progressive hedging, Benders, ...).

StochLift solves the extensive form directly. For models too large for that, the
same scenarios can be passed to `mpi-sppy <https://github.com/Pyomo/mpi-sppy>`_::

    ex = study.to_mpisppy()
    from mpisppy.opt.ef import ExtensiveForm
    ef = ExtensiveForm({"solver": "appsi_highs"}, ex["all_scenario_names"],
                       ex["scenario_creator"])

or, from the command line, ``stochlift export model.py`` writes a module for
mpi-sppy's ``generic_cylinders``.

Each scenario model is rebuilt in Pyomo from the matrix StochLift read from the
user's model, so any supported modeling library can be exported. The objective
is in minimization form (negated for a maximization model).
"""
from __future__ import annotations

import numpy as np

from .lift import first_stage_names


def to_pyomo(model, name: str = "scenario"):
    """A Pyomo ``ConcreteModel`` equivalent to a :class:`~stochlift.model.LinearModel`.

    Variables are ``m.x[j]`` in column order (``m.names[j]`` holds the original
    name); the objective is ``m.obj`` (minimize).
    """
    import pyomo.environ as pyo
    from pyomo.core.expr.numeric_expr import LinearExpression

    m = pyo.ConcreteModel(name)
    m.names = list(model.names)
    m.x = pyo.Var(range(model.n))
    for j in range(model.n):
        v = m.x[j]
        v.domain = pyo.Integers if model.integer[j] else pyo.Reals
        v.setlb(None if not np.isfinite(model.lb[j]) else float(model.lb[j]))
        v.setub(None if not np.isfinite(model.ub[j]) else float(model.ub[j]))
    A = model.A.tocsr()
    m.rows = pyo.ConstraintList()
    for i in range(model.m):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        expr = LinearExpression(constant=0.0, linear_coefs=A.data[lo:hi].tolist(),
                                linear_vars=[m.x[j] for j in A.indices[lo:hi]])
        lb = float(model.row_lb[i]) if np.isfinite(model.row_lb[i]) else None
        ub = float(model.row_ub[i]) if np.isfinite(model.row_ub[i]) else None
        if lb is None and ub is None:
            continue
        m.rows.add((lb, expr, ub))
    nz = np.flatnonzero(model.c)
    m.obj = pyo.Objective(expr=LinearExpression(constant=float(model.offset), linear_coefs=model.c[nz].tolist(),
                                                linear_vars=[m.x[j] for j in nz]), sense=pyo.minimize)
    return m


def to_mpisppy(study) -> dict:
    """``all_scenario_names``, ``scenario_creator`` and ``scenario_creator_kwargs`` for mpi-sppy."""
    if study.risk.active:
        raise NotImplementedError("export of a mean-CVaR objective is not supported; mpi-sppy has its "
                                  "own --cvar option for the risk-neutral export")
    from mpisppy.utils import sputils
    from pyomo.core.expr.numeric_expr import LinearExpression

    models = study.models()
    probs = study.scenarios.probs
    first = first_stage_names(models, study.is_first)
    names = [f"scen{s}" for s in range(len(models))]
    for s, lm in enumerate(models):
        missing = set(first) - set(lm.names)
        if missing:
            raise ValueError(f"scenario {s} has no first-stage variables {sorted(missing)[:5]}; "
                             "mpi-sppy needs the same first-stage variables in every scenario")

    def scenario_creator(scenario_name, **kwargs):
        s = int(str(scenario_name).replace("scen", ""))
        lm = models[s]
        m = to_pyomo(lm, scenario_name)
        idx = lm.index()
        cols = [idx[nm] for nm in first]
        m.first_stage_cost = LinearExpression(constant=0.0, linear_coefs=[float(lm.c[j]) for j in cols],
                                              linear_vars=[m.x[j] for j in cols])
        sputils.attach_root_node(m, m.first_stage_cost, [m.x[j] for j in cols])
        m._mpisppy_probability = float(probs[s])
        return m

    return {"all_scenario_names": names, "scenario_creator": scenario_creator,
            "scenario_creator_kwargs": {}, "first_stage": first}


def solve_ef_with_mpisppy(study, solver: str = "appsi_highs") -> tuple:
    """Solve the extensive form with mpi-sppy. Returns (objective in the user's sense,
    first-stage values by name). Used to cross-check StochLift's own extensive form."""
    from mpisppy.opt.ef import ExtensiveForm
    import pyomo.environ as pyo

    ex = to_mpisppy(study)
    ef = ExtensiveForm({"solver": solver}, ex["all_scenario_names"], ex["scenario_creator"],
                       scenario_creator_kwargs=ex["scenario_creator_kwargs"], suppress_warnings=True)
    res = ef.solve_extensive_form(tee=False)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"mpi-sppy extensive form: {res.solver.termination_condition}")
    sign = study.models()[0].sign
    scen = next(iter(ef.scenarios()))[1]
    x = {nm: float(pyo.value(v)) for nm, v in zip(ex["first_stage"], scen._mpisppy_node_list[0].nonant_vardata_list)}
    return sign * float(pyo.value(ef.ef.EF_Obj)), x


MODULE = '''"""mpi-sppy scenario module written by `stochlift export`.

Run, for example (the extensive form needs no MPI; decomposition needs mpi4py and an MPI
installation such as MS-MPI or Open MPI, see the mpi-sppy documentation):
    python -m mpisppy.generic_cylinders --module-name {module} --num-scens {n} --EF --EF-solver-name appsi_highs
    mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders --module-name {module} --num-scens {n} \\
        --solver-name appsi_highs --max-iterations 50 --default-rho 1 --lagrangian --xhatshuffle
The objective is in minimization form{negated}.
"""
import stochlift as sl
from stochlift.cli import load_model

_build, _data = load_model({model!r}, {data!r})
_study = sl.lift(_build, _data, spec={spec!r}, history={history!r})
_export = _study.to_mpisppy()


def scenario_creator(scenario_name, **kwargs):
    return _export["scenario_creator"](scenario_name)


def scenario_names_creator(num_scens, start=None):
    names = _export["all_scenario_names"]
    start = start or 0
    return names[start:start + num_scens]


def inparser_adder(cfg):
    cfg.num_scens_required()


def kw_creator(cfg):
    return {{}}


def scenario_denouement(rank, scenario_name, scenario):
    pass
'''


def write_module(path: str, model: str, spec: str, data=None, history=None, n: int = 0, sign: float = 1.0) -> str:
    import os

    module = os.path.splitext(os.path.basename(path))[0]
    text = MODULE.format(module=module, n=n, model=os.path.abspath(model), spec=os.path.abspath(spec),
                         data=os.path.abspath(data) if data and os.path.exists(data) else data,
                         history=os.path.abspath(history) if history else None,
                         negated=" (negated: the original model maximizes)" if sign < 0 else "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
