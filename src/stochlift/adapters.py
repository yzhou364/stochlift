"""Convert models from common Python modeling libraries to :class:`LinearModel`.

Models are read from the library objects directly rather than through an MPS
file, because MPS drops the objective sense and the objective constant in
several writers.

Supported: PuLP, Pyomo, gurobipy, OR-Tools ``pywraplp``, ``highspy.Highs``,
a path to an ``.mps`` / ``.lp`` file, or a :class:`LinearModel`.
Only linear and mixed-integer linear models are supported.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

from .model import INF, LinearModel, _clean_inf


class UnsupportedModel(TypeError):
    pass


def to_linear_model(obj) -> LinearModel:
    if isinstance(obj, LinearModel):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return _from_file(os.fspath(obj))
    mod = type(obj).__module__ or ""
    if mod.startswith("pulp"):
        return _from_pulp(obj)
    if mod.startswith("pyomo"):
        return _from_pyomo(obj)
    if mod.startswith("gurobipy"):
        return _from_gurobi(obj)
    if mod.startswith("ortools"):
        return _from_ortools(obj)
    if mod.startswith("highspy"):
        return _from_highs(obj)
    raise UnsupportedModel(
        f"cannot read a model of type {type(obj).__module__}.{type(obj).__name__}; "
        "supported: PuLP, Pyomo, gurobipy, OR-Tools pywraplp, highspy, .mps/.lp path")


def _assemble(names, c, offset, lb, ub, integer, rows, cols, vals, row_lb, row_ub,
              row_names, maximize) -> LinearModel:
    n, m = len(names), len(row_lb)
    A = sp.coo_matrix((np.asarray(vals, dtype=float), (rows, cols)), shape=(m, n)).tocsr()
    A.sum_duplicates()
    c = np.asarray(c, dtype=float)
    sign = -1.0 if maximize else 1.0
    return LinearModel(names=list(names), c=sign * c, offset=sign * float(offset),
                       lb=_clean_inf(lb), ub=_clean_inf(ub), integer=integer, A=A,
                       row_lb=_clean_inf(row_lb), row_ub=_clean_inf(row_ub),
                       row_names=list(row_names), sign=sign)


# --------------------------------------------------------------------------- PuLP
def _from_pulp(prob) -> LinearModel:
    import pulp

    variables = prob.variables()
    names = [v.name for v in variables]
    # look variables up by name: PuLP >= 4 returns a new wrapper object on every access
    idx = {v.name: j for j, v in enumerate(variables)}
    n = len(variables)
    lb = [(-INF if v.lowBound is None else v.lowBound) for v in variables]
    ub = [(INF if v.upBound is None else v.upBound) for v in variables]
    integer = [v.cat in ("Integer", "Binary") for v in variables]
    c = np.zeros(n)
    offset = 0.0
    if prob.objective is not None:
        for v, coef in prob.objective.items():
            c[idx[v.name]] += coef
        offset = prob.objective.constant or 0.0
    rows, cols, vals, rl, ru, rn = [], [], [], [], [], []
    cons = prob.constraints
    try:                                   # PuLP >= 3.3: constraints() returns a list
        named = [(con.name, con) for con in cons()] if callable(cons) else list(cons.items())
    except TypeError:
        named = list(cons.items())
    for i, (name, con) in enumerate(named):
        for v, coef in con.items():
            rows.append(i)
            cols.append(idx[v.name])
            vals.append(coef)
        rhs = -con.constant
        if con.sense == pulp.LpConstraintLE:
            rl.append(-INF); ru.append(rhs)
        elif con.sense == pulp.LpConstraintGE:
            rl.append(rhs); ru.append(INF)
        else:
            rl.append(rhs); ru.append(rhs)
        rn.append(name)
    return _assemble(names, c, offset, lb, ub, integer, rows, cols, vals, rl, ru, rn,
                     maximize=(prob.sense == pulp.LpMaximize))


# -------------------------------------------------------------------------- Pyomo
def _from_pyomo(model) -> LinearModel:
    from pyomo.core import Constraint, Objective, maximize, value
    from pyomo.repn import generate_standard_repn

    objs = list(model.component_data_objects(Objective, active=True))
    if len(objs) != 1:
        raise UnsupportedModel(f"expected exactly one active objective, found {len(objs)}")
    obj = objs[0]

    var_index: dict = {}
    variables: list = []

    def col(v):
        k = id(v)
        if k not in var_index:
            var_index[k] = len(variables)
            variables.append(v)
        return var_index[k]

    def linear(expr, what):
        repn = generate_standard_repn(expr, compute_values=True)
        if not repn.is_linear():
            raise UnsupportedModel(f"{what} is not linear; StochLift supports (MI)LP only")
        const = float(repn.constant) if repn.constant is not None else 0.0
        return [(col(v), float(a)) for v, a in zip(repn.linear_vars, repn.linear_coefs)], const

    cons = []
    for con in model.component_data_objects(Constraint, active=True):
        terms, const = linear(con.body, f"constraint {con.name}")
        lo = -INF if con.lower is None else float(value(con.lower)) - const
        hi = INF if con.upper is None else float(value(con.upper)) - const
        cons.append((con.name, terms, lo, hi))
    obj_terms, offset = linear(obj.expr, "objective")

    n = len(variables)
    c = np.zeros(n)
    for j, a in obj_terms:
        c[j] += a
    rows, cols, vals, rl, ru, rn = [], [], [], [], [], []
    for i, (name, terms, lo, hi) in enumerate(cons):
        for j, a in terms:
            rows.append(i); cols.append(j); vals.append(a)
        rl.append(lo); ru.append(hi); rn.append(name)
    names = [v.name for v in variables]
    lb = [(-INF if v.lb is None else float(v.lb)) for v in variables]
    ub = [(INF if v.ub is None else float(v.ub)) for v in variables]
    integer = [bool(v.is_integer() or v.is_binary()) for v in variables]
    return _assemble(names, c, offset, lb, ub, integer, rows, cols, vals, rl, ru, rn,
                     maximize=(obj.sense == maximize))


# ------------------------------------------------------------------------- Gurobi
def _from_gurobi(model) -> LinearModel:
    model.update()
    if model.IsQP or model.IsQCP or model.NumSOS or model.NumGenConstrs:
        raise UnsupportedModel("quadratic, SOS and general constraints are not supported")
    if getattr(model, "NumPWLObjVars", 0) or getattr(model, "NumObj", 1) > 1:
        raise UnsupportedModel("piecewise-linear and multiple objectives are not supported")
    vs, cs = model.getVars(), model.getConstrs()
    A = model.getA().tocsr()
    c = np.array(model.getAttr("Obj", vs), dtype=float)
    lb = np.array(model.getAttr("LB", vs), dtype=float)
    ub = np.array(model.getAttr("UB", vs), dtype=float)
    vtype = model.getAttr("VType", vs)
    if any(t in ("S", "N") for t in vtype):
        raise UnsupportedModel("semi-continuous and semi-integer variables are not supported")
    integer = [t in ("B", "I") for t in vtype]
    sense = model.getAttr("Sense", cs)
    rhs = np.array(model.getAttr("RHS", cs), dtype=float)
    rl = np.where(np.array([s in (">", "=") for s in sense]), rhs, -INF) if len(cs) else np.array([])
    ru = np.where(np.array([s in ("<", "=") for s in sense]), rhs, INF) if len(cs) else np.array([])
    sign = -1.0 if model.ModelSense < 0 else 1.0
    return LinearModel(names=list(model.getAttr("VarName", vs)), c=sign * c,
                       offset=sign * float(model.ObjCon), lb=_clean_inf(lb), ub=_clean_inf(ub),
                       integer=integer, A=A, row_lb=rl, row_ub=ru,
                       row_names=list(model.getAttr("ConstrName", cs)), sign=sign)


# ----------------------------------------------------------------------- OR-Tools
def _from_ortools(solver) -> LinearModel:
    if not hasattr(solver, "variables") or not hasattr(solver, "Objective"):
        raise UnsupportedModel("only ortools.linear_solver.pywraplp.Solver is supported")
    vs, cs = solver.variables(), solver.constraints()
    obj = solver.Objective()
    names = [v.name() for v in vs]
    c = [obj.GetCoefficient(v) for v in vs]
    rows, cols, vals, rl, ru, rn = [], [], [], [], [], []
    for i, con in enumerate(cs):
        for j, v in enumerate(vs):
            a = con.GetCoefficient(v)
            if a != 0.0:
                rows.append(i); cols.append(j); vals.append(a)
        rl.append(con.lb()); ru.append(con.ub()); rn.append(con.name() or f"r{i}")
    return _assemble(names, c, obj.offset(), [v.lb() for v in vs], [v.ub() for v in vs],
                     [bool(v.integer()) for v in vs], rows, cols, vals, rl, ru, rn,
                     maximize=obj.maximization())


# -------------------------------------------------------------------------- HiGHS
def _fill_names(names, count, prefix) -> list:
    """Give unnamed columns/rows a unique default name."""
    names = (names + [""] * count)[:count]
    used = {nm for nm in names if nm}
    out = []
    for i, nm in enumerate(names):
        if not nm:
            nm = f"{prefix}{i}"
            while nm in used:
                nm += "_"
            used.add(nm)
        out.append(nm)
    return out


def _from_highs(h) -> LinearModel:
    import highspy

    m = h.getModel()
    if m.hessian_.dim_ > 0:
        raise UnsupportedModel("quadratic objectives are not supported")
    lp = m.lp_
    n, nr = lp.num_col_, lp.num_row_
    data = (np.array(lp.a_matrix_.value_, dtype=float), np.array(lp.a_matrix_.index_),
            np.array(lp.a_matrix_.start_))
    if lp.a_matrix_.format_ == highspy.MatrixFormat.kColwise:
        A = sp.csc_matrix(data, shape=(nr, n)).tocsr()
    else:
        A = sp.csr_matrix(data, shape=(nr, n))
    integ = list(lp.integrality_)
    integer = [t != highspy.HighsVarType.kContinuous for t in integ] if integ else [False] * n
    names = _fill_names(list(lp.col_names_), n, "x")
    rnames = _fill_names(list(lp.row_names_), nr, "r")
    sign = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    return LinearModel(names=names, c=sign * np.array(lp.col_cost_, dtype=float),
                       offset=sign * float(lp.offset_), lb=_clean_inf(lp.col_lower_),
                       ub=_clean_inf(lp.col_upper_), integer=integer, A=A,
                       row_lb=_clean_inf(lp.row_lower_), row_ub=_clean_inf(lp.row_upper_),
                       row_names=rnames, sign=sign)


def _from_file(path: str) -> LinearModel:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(path)
    if status == highspy.HighsStatus.kError:
        raise UnsupportedModel(f"HiGHS could not read {path}")
    return _from_highs(h)
