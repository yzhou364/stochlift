# Figure legends

Draft legends for each figure, generated from the results. Edit them to describe your model.

**Overview | Lifting the deterministic model to a two-stage stochastic program.** **a,** value of modelling the uncertainty: Expected outcome of the stochastic decision (RP, orange square), of the mean-value decision evaluated over the scenarios (EEV, blue circle) and with perfect information (WS, green diamond); the open circle is the deterministic model's own estimate at the mean data (EV = 247M). Brackets: value of the stochastic solution, VSS = 427M (143.87% of RP), and expected value of perfect information, EVPI = 20.6M (6.94% of RP). 200 scenarios. Objective: 0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes). **b,** first-stage decision: Value of each first-stage variable in the mean-value model (blue circles) and in the stochastic model (orange squares); 3 of 3 variables differ. **c,** scenarios: Values of the 3 uncertain entries across the 200 scenarios (sampled from the specified distributions), relative to their probability-weighted means. Shaded: probability-weighted density; bar: 25th–75th percentiles; line: 5th–95th percentiles; white dot: median. **d,** paired out-of-sample comparison: Gain of the stochastic decision over the mean-value decision in each of 1000 independent samples from the distributions (positive: the stochastic decision is better). Black dot and bar: mean gain 132M and its 95% bootstrap interval (118M to 148M; 10,000 resamples, treating scenarios as independent). The stochastic decision is better in 45% and worse in 55% of the scenarios. On the risk-adjusted objective the gain is 421M (95% interval 385M to 457M). **e,** stability: Optimal value of the stochastic program on 5 randomly drawn scenario sets of each size (blue circles) and its result out of sample (orange squares). Line: mean over the sets; band: range. **f,** mean–risk trade-off: Expected cost against CVaR at 0.9 of the decisions that minimize (1 − w) × expectation + w × CVaR for the weights w shown, evaluated on the scenario set (filled squares). Open squares: the same decisions out of sample. The blue circle is the mean-value decision.  
`fig_overview`

**Fig. 1 | Value of modelling the uncertainty.** Expected outcome of the stochastic decision (RP, orange square), of the mean-value decision evaluated over the scenarios (EEV, blue circle) and with perfect information (WS, green diamond); the open circle is the deterministic model's own estimate at the mean data (EV = 247M). Brackets: value of the stochastic solution, VSS = 427M (143.87% of RP), and expected value of perfect information, EVPI = 20.6M (6.94% of RP). 200 scenarios. Objective: 0.5 x expected cost + 0.5 x CVaR at 0.9 (mean of the worst 10% of outcomes).  
`fig_value` · source data: `source_data/fig_value.csv`

**Fig. 2 | First-stage decision.** Value of each first-stage variable in the mean-value model (blue circles) and in the stochastic model (orange squares); 3 of 3 variables differ.  
`fig_first_stage` · source data: `source_data/fig_first_stage.csv`

**Fig. 3 | Scenarios.** Values of the 3 uncertain entries across the 200 scenarios (sampled from the specified distributions), relative to their probability-weighted means. Shaded: probability-weighted density; bar: 25th–75th percentiles; line: 5th–95th percentiles; white dot: median.  
`fig_scenarios` · source data: `source_data/fig_scenarios.csv`

**Fig. 4 | Out-of-sample outcomes.** Realized cost of each decision on 1000 independent samples from the distributions that were not used to make it. Box: 25th–75th percentiles with the median; whiskers: 5th–95th percentiles; marker: mean. Risk-adjusted objective: mean-value decision 723M, stochastic decision 302M.  
`fig_out_of_sample` · source data: `source_data/fig_out_of_sample.csv`

**Fig. 5 | Paired out-of-sample comparison.** Gain of the stochastic decision over the mean-value decision in each of 1000 independent samples from the distributions (positive: the stochastic decision is better). Black dot and bar: mean gain 132M and its 95% bootstrap interval (118M to 148M; 10,000 resamples, treating scenarios as independent). The stochastic decision is better in 45% and worse in 55% of the scenarios. On the risk-adjusted objective the gain is 421M (95% interval 385M to 457M).  
`fig_gain` · source data: `source_data/fig_gain.csv`

**Fig. 6 | Stability.** Optimal value of the stochastic program on 5 randomly drawn scenario sets of each size (blue circles) and its result out of sample (orange squares). Line: mean over the sets; band: range.  
`fig_stability` · source data: `source_data/fig_stability.csv`

**Fig. 7 | Mean–risk trade-off.** Expected cost against CVaR at 0.9 of the decisions that minimize (1 − w) × expectation + w × CVaR for the weights w shown, evaluated on the scenario set (filled squares). Open squares: the same decisions out of sample. The blue circle is the mean-value decision.  
`fig_risk_frontier` · source data: `source_data/fig_risk_frontier.csv`
