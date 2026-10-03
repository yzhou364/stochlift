"""Compare StochLift's order with the textbook critical-ratio answer on the same scenarios."""
import numpy as np

import stochlift as sl
from model import DATA, build_model

study = sl.lift(build_model, DATA, spec="uncertainty.yaml")
r = study.solve()
d = np.sort(study.scenarios.values[:, 0])
ratio = (DATA["price"] - DATA["cost"]) / (DATA["price"] - DATA["salvage"])
k = int(np.ceil(ratio * len(d))) - 1                     # smallest q with F(q) >= ratio
print(f"critical ratio {ratio:.2f}: quantile of the scenario demands = {d[k]:.3f}, "
      f"StochLift order = {r.x_rp['order']:.3f}, mean-value order = {r.x_ev['order']:.3f}")
lo, hi = d[k], d[min(k + 1, len(d) - 1)]                 # at an exact ratio any order in [lo, hi] is optimal
assert lo - 1e-6 <= r.x_rp["order"] <= hi + 1e-6
