# StochLift report

Objective sense: **min** (lower is better). Scenarios: **200**.

Risk-averse objective: **0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes)**. EV, WS, RP and EEV below are values of this objective, so VSS and EVPI are risk-adjusted too.

## Is modeling the uncertainty worth it?

**Yes.** On the scenario set the stochastic decision is better by 426,809,926 per decision in the objective (0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes)) (143.87% of RP); this is the value of the stochastic solution (VSS). On 1000 independent samples from the distributions the same objective improves by 420,999,692 (95% interval 385,107,191 to 457,061,968); the mean gain is 132,433,677.

Better forecasts are worth at most 20,589,033 (6.94% of RP): the expected value of perfect information (EVPI).

| Quantity | Meaning | Objective |
| --- | --- | ---: |
| EV | deterministic model at mean data (its own, optimistic estimate) | 247,069,250 |
| WS | wait-and-see: each scenario solved with perfect information | 276,084,857 |
| RP | stochastic (recourse) solution | 296,673,890 |
| EEV | the mean-value decision evaluated over the scenarios | 723,483,815 |
| VSS | EEV vs RP | 426,809,926 |
| EVPI | RP vs WS | 20,589,033 |

| Decision | Expected cost | CVaR at 0.9 |
| --- | ---: | ---: |
| stochastic (risk-averse) | 275,084,522 | 318,263,258 |
| mean-value | 430,123,472 | 1,016,844,159 |
| perfect information | 247,069,250 | 305,100,464 |

## Mean-risk trade-off

Each row solves the stochastic program with a different weight on CVaR at 0.9.

| CVaR weight | Expected cost | CVaR | Out-of-sample mean | Out-of-sample CVaR |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 270,641,097 | 342,756,795 | 271,727,406 | 369,286,995 |
| 0.25 | 273,506,078 | 320,558,071 | 273,148,155 | 331,698,969 |
| 0.5 | 275,084,522 | 318,263,258 | 274,727,062 | 328,566,167 |
| 0.75 | 279,519,689 | 315,463,104 | 279,551,142 | 325,874,506 |
| 1 | 280,432,603 | 315,316,059 | 280,503,936 | 325,748,084 |

## First-stage decision

| Variable | Mean-value model | Stochastic model |
| --- | ---: | ---: |
| cap[base] | 895.7 | 1,115 |
| cap[mid] | 359.4 | 187.9 |
| cap[peak] | 0 | 429.7 |

3 of 3 first-stage variables differ.

## Out-of-sample test

Both decisions were applied to 1000 independent samples from the distributions that were not used to build the scenarios.

- Mean cost: mean-value decision 407,160,739, stochastic decision 274,727,062.
- Mean gain of the stochastic decision: 132,433,677 per observation, 95% bootstrap interval [117,759,093, 147,818,919].
- The stochastic decision was better in 45% of the observations and worse in 55%.
- Feasible observations: mean-value 1000/1000, stochastic 1000/1000.
- Risk-adjusted objective: mean-value decision 722,646,307, stochastic decision 301,646,614; gain 420,999,692, 95% bootstrap interval [385,107,191, 457,061,968].

## Checks

- **PASS** `builder_deterministic`: two builds from the same data give the same model
- **PASS** `uncertainty_reaches_model`: coefficients affected: load -> 3 rhs
- **PASS** `stage_split`: 3 first-stage and 12 recourse variables
- **PASS** `first_stage_rows_stable`: constraints on first-stage variables only are the same in every scenario
- **PASS** `single_scenario_reduction`: one-scenario lift = 2.47069e+08, deterministic model = 2.47069e+08
- **PASS** `bound_ordering`: WS <= RP <= EEV: WS = 2.76085e+08, RP = 2.96674e+08, EEV = 7.23484e+08
- **PASS** `decomposition_consistency`: extensive form = 2.96674e+08; re-solving each scenario with the first stage fixed gives 2.96674e+08
- **PASS** `mean_value_recourse`: the mean-value first stage is feasible in every scenario

The checks verify that the lift is mathematically consistent with the deterministic model. They cannot verify that the stage assignment matches how decisions are made in practice; that is confirmed by reviewing `uncertainty.yaml`.

## Figures

![Overview](fig_overview.png)

Legends for every figure are in `captions.md`, and the numbers behind each figure in `source_data/`. Files: `fig_overview`, `fig_value`, `fig_first_stage`, `fig_scenarios`, `fig_out_of_sample`, `fig_gain`, `fig_stability`, `fig_risk_frontier` (PDF, SVG and PNG).
