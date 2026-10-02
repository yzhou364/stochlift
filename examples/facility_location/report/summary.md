# StochLift report

Objective sense: **min** (lower is better). Scenarios: **30**.

## Is modeling the uncertainty worth it?

**Not clearly.** On the scenario set the stochastic decision is better by 669.24 per decision in expectation (10.53% of RP); this is the value of the stochastic solution (VSS). On 78 held-out observations the mean gain is 282.97 (95% interval -4.20 to 595.55). The interval contains zero, so the gain is not distinguishable from noise on unseen data.

Better forecasts are worth at most 672.09 (10.57% of RP): the expected value of perfect information (EVPI).

| Quantity | Meaning | Expected cost |
| --- | --- | ---: |
| EV | deterministic model at mean data (its own, optimistic estimate) | 5,479.48 |
| WS | wait-and-see: each scenario solved with perfect information | 5,685.98 |
| RP | stochastic (recourse) solution | 6,358.07 |
| EEV | the mean-value decision evaluated over the scenarios | 7,027.31 |
| VSS | EEV vs RP | 669.24 |
| EVPI | RP vs WS | 672.09 |

## First-stage decision

| Variable | Mean-value model | Stochastic model |
| --- | ---: | ---: |
| open[S3] | 0 | 1 |
| open[S4] | 1 | 0 |
| open[S1] | 1 | 1 |
| open[S2] | 0 | 0 |
| open[S5] | 1 | 1 |

2 of 5 first-stage variables differ.

## Out-of-sample test

Both decisions were applied to 78 held-out observations that were not used to build the scenarios.

- Mean cost: mean-value decision 6,477.10, stochastic decision 6,194.13.
- Mean gain of the stochastic decision: 282.97 per observation, 95% bootstrap interval [-4.20, 595.55].
- The stochastic decision was better in 36% of the observations and worse in 64%.
- Feasible observations: mean-value 78/78, stochastic 78/78.

## Solution quality

Optimality gap of the stochastic solution (Mak-Morton-Wood estimate, 20 batches of 50 scenarios): mean 4.55, 95% upper bound 8.19 (0.13% of RP).

## Checks

- **PASS** `builder_deterministic`: two builds from the same data give the same model
- **PASS** `uncertainty_reaches_model`: coefficients affected: demand -> 8 rhs
- **PASS** `stage_split`: 5 first-stage and 48 recourse variables
- **PASS** `first_stage_rows_stable`: constraints on first-stage variables only are the same in every scenario
- **PASS** `single_scenario_reduction`: one-scenario lift = 5479.48, deterministic model = 5479.48
- **PASS** `bound_ordering`: WS <= RP <= EEV: WS = 5685.98, RP = 6358.07, EEV = 7027.31
- **PASS** `decomposition_consistency`: extensive form = 6358.07; re-solving each scenario with the first stage fixed gives 6358.07
- **PASS** `mean_value_recourse`: the mean-value first stage is feasible in every scenario

The checks verify that the lift is mathematically consistent with the deterministic model. They cannot verify that the stage assignment matches how decisions are made in practice; that is confirmed by reviewing `uncertainty.yaml`.

## Figures

- `fig_value.pdf`
- `fig_first_stage.pdf`
- `fig_scenarios.pdf`
- `fig_out_of_sample.pdf`
- `fig_gain.pdf`
- `fig_stability.pdf`
