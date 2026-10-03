"""Generation capacity expansion as an ordinary deterministic gurobipy model.

Choose how much capacity of each technology to build (annualized cost per MW),
then dispatch it to meet the load in three blocks of the load-duration curve
(fuel cost per MWh). Load that cannot be served costs a high penalty.
Synthetic data in the spirit of the classic textbook example.
"""
import gurobipy as gp
from gurobipy import GRB

TECH = ["base", "mid", "peak"]
BLOCKS = ["night", "day", "peak_hours"]

DATA = {
    "invest": {"base": 120_000.0, "mid": 70_000.0, "peak": 35_000.0},     # $ per MW-year
    "fuel": {"base": 15.0, "mid": 45.0, "peak": 110.0},                   # $ per MWh
    "hours": {"night": 3_500.0, "day": 4_500.0, "peak_hours": 760.0},     # hours per year
    "load": {"night": 600.0, "day": 900.0, "peak_hours": 1_250.0},        # MW
    "unserved": 3_000.0,                                                  # $ per MWh not served
}


def build_model(data):
    m = gp.Model("capacity")
    m.Params.OutputFlag = 0
    cap = m.addVars(TECH, lb=0, name="cap")
    gen = m.addVars(TECH, BLOCKS, lb=0, name="gen")
    short = m.addVars(BLOCKS, lb=0, name="short")
    m.setObjective(
        gp.quicksum(data["invest"][t] * cap[t] for t in TECH)
        + gp.quicksum(data["hours"][b] * (gp.quicksum(data["fuel"][t] * gen[t, b] for t in TECH)
                                          + data["unserved"] * short[b]) for b in BLOCKS),
        GRB.MINIMIZE)
    m.addConstrs((gp.quicksum(gen[t, b] for t in TECH) + short[b] >= data["load"][b] for b in BLOCKS),
                 name="meet")
    m.addConstrs((gen[t, b] <= cap[t] for t in TECH for b in BLOCKS), name="avail")
    m.update()
    return m
