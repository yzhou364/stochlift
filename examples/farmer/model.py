"""The farmer problem (Birge and Louveaux, Introduction to Stochastic Programming, ch. 1).

This file is an ordinary *deterministic* PuLP model. It knows nothing about
scenarios: StochLift calls ``build_model`` once per scenario with different data.
"""
import pulp

CROPS = ["wheat", "corn", "beets"]

DATA = {
    "land": 500,                                                  # acres
    "plant_cost": {"wheat": 150, "corn": 230, "beets": 260},      # $/acre
    "yield": {"wheat": 2.5, "corn": 3.0, "beets": 20.0},          # tons/acre (average year)
    "feed_need": {"wheat": 200, "corn": 240},                     # tons needed for cattle
    "buy_price": {"wheat": 238, "corn": 210},                     # $/ton
    "sell_price": {"wheat": 170, "corn": 150, "beets": 36},       # $/ton
    "beet_quota": 6000,                                           # tons sold at full price
    "beet_excess_price": 10,                                      # $/ton above the quota
}


def build_model(data):
    m = pulp.LpProblem("farmer", pulp.LpMinimize)
    acres = pulp.LpVariable.dicts("acres", CROPS, lowBound=0)
    buy = pulp.LpVariable.dicts("buy", ["wheat", "corn"], lowBound=0)
    sell = pulp.LpVariable.dicts("sell", CROPS, lowBound=0)
    sell_excess = pulp.LpVariable("sell_beets_excess", lowBound=0)

    m += (pulp.lpSum(data["plant_cost"][c] * acres[c] for c in CROPS)
          + pulp.lpSum(data["buy_price"][c] * buy[c] for c in buy)
          - pulp.lpSum(data["sell_price"][c] * sell[c] for c in CROPS)
          - data["beet_excess_price"] * sell_excess), "net_cost"

    m += pulp.lpSum(acres[c] for c in CROPS) <= data["land"], "land"
    for c in ["wheat", "corn"]:
        m += data["yield"][c] * acres[c] + buy[c] - sell[c] >= data["feed_need"][c], f"feed_{c}"
    m += sell["beets"] + sell_excess <= data["yield"]["beets"] * acres["beets"], "beet_harvest"
    m += sell["beets"] <= data["beet_quota"], "beet_quota"
    return m
