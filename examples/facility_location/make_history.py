"""Generate the synthetic weekly demand history used by this example.

The data is made up. Demand per customer is lognormal around its base level,
with one market-wide factor shared by all customers and an individual factor.
"""
import numpy as np
import pandas as pd

from model import CUSTOMERS, DATA

rng = np.random.default_rng(20260930)
weeks = 260
common = rng.normal(0.0, 0.16, size=(weeks, 1))          # market-wide swings
own = rng.normal(0.0, 0.22, size=(weeks, len(CUSTOMERS)))
base = np.array([DATA["demand"][c] for c in CUSTOMERS])
demand = base * np.exp(common + own - 0.5 * (0.16**2 + 0.22**2))
df = pd.DataFrame(np.round(demand, 1), columns=CUSTOMERS)
df.insert(0, "week", np.arange(1, weeks + 1))
df.to_csv("demand_history.csv", index=False)
print(df.describe().round(1))
