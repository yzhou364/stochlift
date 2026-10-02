# StochLift report

Objective sense: **min** (lower is better). Scenarios: **100**.

Risk-averse objective: **0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes)**. EV, WS, RP and EEV below are values of this objective, so VSS and EVPI are risk-adjusted too.

## Is modeling the uncertainty worth it?

**Yes.** On the scenario set the stochastic decision is better by 380.97 per decision in the objective (0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes)) (0.48% of RP); this is the value of the stochastic solution (VSS). On 500 independent samples from the distributions the same objective improves by 244.77 (95% interval 121.45 to 346.42); the mean gain is -141.74.

Better forecasts are worth at most 6,784.25 (8.48% of RP): the expected value of perfect information (EVPI).

| Quantity | Meaning | Objective |
| --- | --- | ---: |
| EV | deterministic model at mean data (its own, optimistic estimate) | -120,305 |
| WS | wait-and-see: each scenario solved with perfect information | -86,816 |
| RP | stochastic (recourse) solution | -80,032 |
| EEV | the mean-value decision evaluated over the scenarios | -79,651 |
| VSS | EEV vs RP | 380.97 |
| EVPI | RP vs WS | 6,784.25 |

| Decision | Expected cost | CVaR at 0.9 |
| --- | ---: | ---: |
| stochastic (risk-averse) | -110,586 | -49,478 |
| mean-value | -110,721 | -48,581 |
| perfect information | -118,167 | -55,466 |

## Mean-risk trade-off

Each row solves the stochastic program with a different weight on CVaR at 0.9.

| CVaR weight | Expected cost | CVaR | Out-of-sample mean | Out-of-sample CVaR |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -111,661 | -46,800 | -108,937 | -40,210 |
| 0.25 | -111,397 | -48,001 | -108,817 | -41,193 |
| 0.5 | -110,586 | -49,478 | -107,989 | -42,155 |
| 0.75 | -109,415 | -50,095 | -106,798 | -42,555 |
| 0.9 | -104,036 | -51,048 | -101,387 | -43,102 |
| 1 | -96,459 | -51,776 | -94,083 | -43,287 |

## First-stage decision

| Variable | Mean-value model | Stochastic model |
| --- | ---: | ---: |
| acres_wheat | 122.5 | 108.3 |
| acres_corn | 79.8 | 94.02 |
| acres_beets | 297.7 | 297.7 |

3 of 3 first-stage variables differ.

## Out-of-sample test

Both decisions were applied to 500 independent samples from the distributions that were not used to build the scenarios.

- Mean cost: mean-value decision -108,130, stochastic decision -107,989.
- Mean gain of the stochastic decision: -141.74 per observation, 95% bootstrap interval [-207.52, -75.53].
- The stochastic decision was better in 40% of the observations and worse in 60%.
- Feasible observations: mean-value 500/500, stochastic 500/500.
- Risk-adjusted objective: mean-value decision -74,827, stochastic decision -75,072; gain 244.77, 95% bootstrap interval [121.45, 346.42].

## Checks

- **PASS** `builder_deterministic`: two builds from the same data give the same model
- **PASS** `uncertainty_reaches_model`: coefficients affected: yield -> 3 matrix
- **PASS** `stage_split`: 3 first-stage and 6 recourse variables
- **PASS** `first_stage_rows_stable`: constraints on first-stage variables only are the same in every scenario
- **PASS** `single_scenario_reduction`: one-scenario lift = -120305, deterministic model = -120305
- **PASS** `bound_ordering`: WS <= RP <= EEV: WS = -86816.2, RP = -80032, EEV = -79651
- **PASS** `decomposition_consistency`: extensive form = -80032; re-solving each scenario with the first stage fixed gives -80032
- **PASS** `mean_value_recourse`: the mean-value first stage is feasible in every scenario

The checks verify that the lift is mathematically consistent with the deterministic model. They cannot verify that the stage assignment matches how decisions are made in practice; that is confirmed by reviewing `uncertainty.yaml`.

## Figures

- `fig_value.pdf`
- `fig_first_stage.pdf`
- `fig_scenarios.pdf`
- `fig_out_of_sample.pdf`
- `fig_gain.pdf`
- `fig_risk_frontier.pdf`
