"""Solver-neutral linear model and the HiGHS solve wrapper.

Every modeling library is converted to :class:`LinearModel`, which always
stores a *minimization* problem. ``sign`` remembers the user's sense:
``user_objective = sign * internal_objective``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

INF = float("inf")


@dataclass
class LinearModel:
    names: list                 # column names
    c: np.ndarray               # objective coefficients (minimization form)
    offset: float               # objective constant (minimization form)
    lb: np.ndarray
    ub: np.ndarray
    integer: np.ndarray         # bool mask
    A: sp.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    row_names: list = field(default_factory=list)
    sign: float = 1.0           # +1: user minimizes, -1: user maximizes

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.lb = np.asarray(self.lb, dtype=float)
        self.ub = np.asarray(self.ub, dtype=float)
        self.integer = np.asarray(self.integer, dtype=bool)
        self.row_lb = np.asarray(self.row_lb, dtype=float)
        self.row_ub = np.asarray(self.row_ub, dtype=float)
        self.A = sp.csr_matrix(self.A, dtype=float)
        n = len(self.names)
        if self.A.shape != (len(self.row_lb), n):
            raise ValueError(f"matrix shape {self.A.shape} does not match "
                             f"{len(self.row_lb)} rows x {n} columns")
        if len(set(self.names)) != n:
            raise ValueError("variable names must be unique")
        if not self.row_names:
            self.row_names = [f"r{i}" for i in range(len(self.row_lb))]

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return len(self.row_lb)

    @property
    def sense(self) -> str:
        return "min" if self.sign > 0 else "max"

    def index(self) -> dict:
        return {name: j for j, name in enumerate(self.names)}

    def same_as(self, other: "LinearModel", tol: float = 0.0) -> bool:
        """Exact structural and numerical equality (used by the checks)."""
        if self.names != other.names or self.row_names != other.row_names:
            return False
        if self.sign != other.sign:
            return False
        pairs = [(self.c, other.c), (self.lb, other.lb), (self.ub, other.ub),
                 (self.row_lb, other.row_lb), (self.row_ub, other.row_ub)]
        for a, b in pairs:
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=tol, equal_nan=True):
                return False
        if not np.array_equal(self.integer, other.integer):
            return False
        if abs(self.offset - other.offset) > tol:
            return False
        d = (self.A - other.A)
        return d.nnz == 0 or abs(d).max() <= tol


@dataclass
class Solution:
    status: str                       # optimal | infeasible | unbounded | other
    objective: float                  # internal (minimization) objective
    x: Optional[np.ndarray] = None
    raw_status: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "optimal"


def _clean_inf(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a[a >= 1e20] = INF
    a[a <= -1e20] = -INF
    return a


def _backend() -> str:
    """``highspy`` when it can be imported, otherwise SciPy's bundled HiGHS.

    Some OR-Tools builds ship their own copy of HiGHS and cannot share a
    process with ``highspy``; SciPy's private copy is unaffected.
    """
    global _BACKEND
    if _BACKEND is None:
        try:
            import highspy  # noqa: F401

            _BACKEND = "highspy"
        except ImportError:
            _BACKEND = "scipy"
    return _BACKEND


_BACKEND = None


def solve(model: LinearModel, fixed: Optional[dict] = None, mip_gap: float = 1e-6,
          time_limit: Optional[float] = None, threads: int = 1) -> Solution:
    """Solve ``model`` with HiGHS. ``fixed`` maps column index -> value."""
    lb, ub = model.lb.copy(), model.ub.copy()
    if fixed:
        for j, v in fixed.items():
            # tolerate round-off when a fixed value sits a hair outside a bound
            if v < lb[j] and lb[j] - v <= 1e-7 * max(1.0, abs(lb[j])):
                v = lb[j]
            if v > ub[j] and v - ub[j] <= 1e-7 * max(1.0, abs(ub[j])):
                v = ub[j]
            if v < lb[j] or v > ub[j]:
                return Solution("infeasible", INF, None, "fixed value outside bounds")
            lb[j] = ub[j] = v
    if _backend() == "scipy":
        return _solve_scipy(model, lb, ub, mip_gap, time_limit)
    return _solve_highspy(model, lb, ub, mip_gap, time_limit, threads)


def _solve_scipy(model, lb, ub, mip_gap, time_limit) -> Solution:
    from scipy.optimize import Bounds, LinearConstraint, milp

    options = {"mip_rel_gap": float(mip_gap), "disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    cons = LinearConstraint(model.A, model.row_lb, model.row_ub) if model.m else None
    res = milp(c=model.c, integrality=model.integer.astype(int), bounds=Bounds(lb, ub),
               constraints=cons, options=options)
    if res.status == 0:
        return Solution("optimal", float(res.fun) + float(model.offset), np.asarray(res.x, dtype=float),
                        "scipy: optimal")
    if res.status == 2:
        return Solution("infeasible", INF, None, "scipy: infeasible")
    if res.status == 3:
        return Solution("unbounded", -INF, None, "scipy: unbounded")
    return Solution("other", float("nan"), None, f"scipy: {res.message}")


def _solve_highspy(model, lb, ub, mip_gap, time_limit, threads) -> Solution:
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_ = model.n
    lp.num_row_ = model.m
    lp.col_cost_ = model.c
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = model.row_lb
    lp.row_upper_ = model.row_ub
    lp.offset_ = float(model.offset)
    A = model.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    if model.integer.any():
        lp.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                           for i in model.integer]

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.setOptionValue("mip_abs_gap", 0.0)
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp)
    h.run()
    st = h.getModelStatus()
    name = h.modelStatusToString(st)
    S = highspy.HighsModelStatus
    if st == S.kOptimal:
        x = np.array(h.getSolution().col_value, dtype=float)
        return Solution("optimal", float(h.getInfo().objective_function_value), x, name)
    if st == S.kInfeasible:
        return Solution("infeasible", INF, None, name)
    if st in (S.kUnbounded, S.kUnboundedOrInfeasible):
        if st == S.kUnboundedOrInfeasible:
            return Solution("other", float("nan"), None, name)
        return Solution("unbounded", -INF, None, name)
    return Solution("other", float("nan"), None, name)
