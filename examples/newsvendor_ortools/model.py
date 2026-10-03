"""The newsvendor problem as an ordinary deterministic OR-Tools model.

Order before demand is known, sell what demand allows, salvage the rest.
With demand known the answer is trivial (order exactly the demand); under
uncertainty the optimal order is the critical-ratio quantile of demand:
F(q*) = (price - cost) / (price - salvage).
"""
from ortools.linear_solver import pywraplp

DATA = {"cost": 6.0, "price": 10.0, "salvage": 2.0, "demand": 100.0}


def build_model(data):
    s = pywraplp.Solver.CreateSolver("GLOP")
    order = s.NumVar(0, s.infinity(), "order")
    sold = s.NumVar(0, s.infinity(), "sold")
    left = s.NumVar(0, s.infinity(), "left")
    s.Add(sold <= data["demand"], "demand")
    s.Add(sold + left == order, "balance")
    s.Maximize(data["price"] * sold + data["salvage"] * left - data["cost"] * order)
    return s
