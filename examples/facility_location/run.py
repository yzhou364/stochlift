"""Facility location with a demand history: scenarios from data, tested on a hold-out period.

    python run.py                      # outsourcing at 80 per unit
    python run.py --outsource-cost 45  # cheap outsourcing: is the stochastic model still worth it?
"""
import argparse
import copy

import stochlift as sl
from model import DATA, build_model

parser = argparse.ArgumentParser()
parser.add_argument("--outsource-cost", type=float, default=DATA["outsource_cost"])
parser.add_argument("--out", default="report")
args = parser.parse_args()

data = copy.deepcopy(DATA)
data["outsource_cost"] = args.outsource_cost

study = sl.lift(build_model, data, spec="uncertainty.yaml", history="demand_history.csv")
study.review()
study.solve()
for check in study.check():
    print(check)
study.out_of_sample()
study.stability(sizes=(5, 10, 20, 40, 80), reps=10)
study.saa_gap(n=50, batches=20)
print(study.report(args.out))
