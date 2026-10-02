"""Capacitated facility location for one week, as an ordinary deterministic Pyomo model."""
import pyomo.environ as pyo

SITES = ["S1", "S2", "S3", "S4", "S5"]
CUSTOMERS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# site and customer coordinates (synthetic)
_SITE_XY = {"S1": (2, 8), "S2": (8, 8), "S3": (5, 5), "S4": (2, 2), "S5": (8, 2)}
_CUST_XY = {"C1": (1, 9), "C2": (4, 8), "C3": (7, 9), "C4": (9, 6),
            "C5": (6, 4), "C6": (3, 4), "C7": (1, 1), "C8": (8, 1)}

DATA = {
    "fixed_cost": {"S1": 1500, "S2": 1400, "S3": 1900, "S4": 1300, "S5": 1450},   # per week
    "capacity": {"S1": 110, "S2": 100, "S3": 150, "S4": 95, "S5": 105},           # units per week
    "ship_cost": {(i, j): round(1.0 + 0.9 * (abs(_SITE_XY[i][0] - _CUST_XY[j][0])
                                           + abs(_SITE_XY[i][1] - _CUST_XY[j][1])), 2)
                  for i in SITES for j in CUSTOMERS},                             # per unit
    "outsource_cost": 80.0,                                                       # per unit short
    "demand": {"C1": 38, "C2": 42, "C3": 35, "C4": 40, "C5": 45, "C6": 36, "C7": 33, "C8": 41},
}


def build_model(data):
    m = pyo.ConcreteModel()
    m.open = pyo.Var(SITES, domain=pyo.Binary)
    m.ship = pyo.Var(SITES, CUSTOMERS, domain=pyo.NonNegativeReals)
    m.short = pyo.Var(CUSTOMERS, domain=pyo.NonNegativeReals)

    m.cost = pyo.Objective(
        expr=sum(data["fixed_cost"][i] * m.open[i] for i in SITES)
        + sum(data["ship_cost"][i, j] * m.ship[i, j] for i in SITES for j in CUSTOMERS)
        + data["outsource_cost"] * sum(m.short[j] for j in CUSTOMERS),
        sense=pyo.minimize)

    m.meet = pyo.Constraint(CUSTOMERS, rule=lambda m, j:
                            sum(m.ship[i, j] for i in SITES) + m.short[j] >= data["demand"][j])
    m.cap = pyo.Constraint(SITES, rule=lambda m, i:
                           sum(m.ship[i, j] for j in CUSTOMERS) <= data["capacity"][i] * m.open[i])
    return m
