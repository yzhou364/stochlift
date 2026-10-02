# StochLift report

Objective sense: **min** (lower is better). Scenarios: **3**.

## Is modeling the uncertainty worth it?

**Yes, on the scenario set.** On the scenario set the stochastic decision is better by 1,150.00 per decision in expectation (1.06% of RP); this is the value of the stochastic solution (VSS). There is no hold-out data, so this has not been tested on observations outside the scenario set.

Better forecasts are worth at most 7,015.56 (6.47% of RP): the expected value of perfect information (EVPI).

| Quantity | Meaning | Expected cost |
| --- | --- | ---: |
| EV | deterministic model at mean data (its own, optimistic estimate) | -118,600 |
| WS | wait-and-see: each scenario solved with perfect information | -115,406 |
| RP | stochastic (recourse) solution | -108,390 |
| EEV | the mean-value decision evaluated over the scenarios | -107,240 |
| VSS | EEV vs RP | 1,150.00 |
| EVPI | RP vs WS | 7,015.56 |

## First-stage decision

| Variable | Mean-value model | Stochastic model |
| --- | ---: | ---: |
| acres_wheat | 120 | 170 |
| acres_beets | 300 | 250 |
| acres_corn | 80 | 80 |

2 of 3 first-stage variables differ.

## Checks

- **PASS** `builder_deterministic`: two builds from the same data give the same model
- **PASS** `uncertainty_reaches_model`: coefficients affected: yield -> 3 matrix
- **PASS** `stage_split`: 3 first-stage and 6 recourse variables
- **PASS** `first_stage_rows_stable`: constraints on first-stage variables only are the same in every scenario
- **PASS** `single_scenario_reduction`: one-scenario lift = -118600, deterministic model = -118600
- **PASS** `bound_ordering`: WS <= RP <= EEV: WS = -115406, RP = -108390, EEV = -107240
- **PASS** `decomposition_consistency`: extensive form = -108390; re-solving each scenario with the first stage fixed gives -108390
- **PASS** `mean_value_recourse`: the mean-value first stage is feasible in every scenario

The checks verify that the lift is mathematically consistent with the deterministic model. They cannot verify that the stage assignment matches how decisions are made in practice; that is confirmed by reviewing `uncertainty.yaml`.

## Figures

- `fig_value.pdf`
- `fig_first_stage.pdf`
- `fig_scenarios.pdf`
