"""Farmer problem: three equally likely yield scenarios (20% above, at, 20% below average)."""
import stochlift as sl
from model import DATA, build_model

scenarios = [(1 / 3, {"yield": {crop: f * y for crop, y in DATA["yield"].items()}})
             for f in (1.2, 1.0, 0.8)]

study = sl.lift(build_model, DATA, first_stage=["acres_*"], scenarios=scenarios)
study.review()
results = study.solve()
for check in study.check():
    print(check)
print(study.report("report"))

# Published values (Birge and Louveaux): profit 108,390 (RP), 115,406 (WS), 107,240 (EEV).
assert round(-results.rp) == 108390 and round(-results.ws) == 115406 and round(-results.eev) == 107240
print(f"RP {results.rp:,.0f}  WS {results.ws:,.0f}  EEV {results.eev:,.0f}  "
      f"VSS {results.vss:,.0f}  EVPI {results.evpi:,.0f}")
