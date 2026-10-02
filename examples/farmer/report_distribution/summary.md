# StochLift report

Objective sense: **min** (lower is better). Scenarios: **100**.

## Is modeling the uncertainty worth it?

**Yes.** On the scenario set the stochastic decision is better by 939.73 per decision in expectation (0.84% of RP); this is the value of the stochastic solution (VSS). On 500 independent samples from the distributions the mean gain is 806.69 (95% interval 448.64 to 1,175.42).

Better forecasts are worth at most 6,505.89 (5.83% of RP): the expected value of perfect information (EVPI).

| Quantity | Meaning | Expected cost |
| --- | --- | ---: |
| EV | deterministic model at mean data (its own, optimistic estimate) | -120,305 |
| WS | wait-and-see: each scenario solved with perfect information | -118,167 |
| RP | stochastic (recourse) solution | -111,661 |
| EEV | the mean-value decision evaluated over the scenarios | -110,721 |
| VSS | EEV vs RP | 939.73 |
| EVPI | RP vs WS | 6,505.89 |

## First-stage decision

| Variable | Mean-value model | Stochastic model |
| --- | ---: | ---: |
| acres_wheat | 122.5 | 136.5 |
| acres_corn | 79.8 | 84.59 |
| acres_beets | 297.7 | 279 |

3 of 3 first-stage variables differ.

## Out-of-sample test

Both decisions were applied to 500 independent samples from the distributions that were not used to build the scenarios.

- Mean cost: mean-value decision -108,130, stochastic decision -108,937.
- Mean gain of the stochastic decision: 806.69 per observation, 95% bootstrap interval [448.64, 1,175.42].
- The stochastic decision was better in 40% of the observations and worse in 60%.
- Feasible observations: mean-value 500/500, stochastic 500/500.

## Solution quality

Optimality gap of the stochastic solution (Mak-Morton-Wood estimate, 10 batches of 50 scenarios): mean 75.53, 95% upper bound 135.65 (0.12% of RP).

## Checks

- **PASS** `builder_deterministic`: two builds from the same data give the same model
- **PASS** `uncertainty_reaches_model`: coefficients affected: yield -> 3 matrix
- **PASS** `stage_split`: 3 first-stage and 6 recourse variables
- **PASS** `first_stage_rows_stable`: constraints on first-stage variables only are the same in every scenario
- **PASS** `single_scenario_reduction`: one-scenario lift = -120305, deterministic model = -120305
- **PASS** `bound_ordering`: WS <= RP <= EEV: WS = -118167, RP = -111661, EEV = -110721
- **PASS** `decomposition_consistency`: extensive form = -111661; re-solving each scenario with the first stage fixed gives -111661
- **PASS** `mean_value_recourse`: the mean-value first stage is feasible in every scenario

The checks verify that the lift is mathematically consistent with the deterministic model. They cannot verify that the stage assignment matches how decisions are made in practice; that is confirmed by reviewing `uncertainty.yaml`.

## Figures

- `fig_value.pdf`
- `fig_first_stage.pdf`
- `fig_scenarios.pdf`
- `fig_out_of_sample.pdf`
- `fig_gain.pdf`
- `fig_stability.pdf`
