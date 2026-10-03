# Figure legends

Draft legends for each figure, generated from the results. Edit them to describe your model.

**Overview | Lifting the deterministic model to a two-stage stochastic program.** **a,** value of modelling the uncertainty: Expected outcome of the stochastic decision (RP, orange square), of the mean-value decision evaluated over the scenarios (EEV, blue circle) and with perfect information (WS, green diamond); the open circle is the deterministic model's own estimate at the mean data (EV = 398.1). Brackets: value of the stochastic solution, VSS = 0.9 (0.30% of RP), and expected value of perfect information, EVPI = 92.4 (30.23% of RP). 400 scenarios. **b,** first-stage decision: Value of each first-stage variable in the mean-value model (blue circles) and in the stochastic model (orange squares); 1 of 1 variables differ. **c,** scenarios: Values of the 1 uncertain entries across the 400 scenarios (sampled from the specified distributions), relative to their probability-weighted means. Shaded: probability-weighted density; bar: 25th–75th percentiles; line: 5th–95th percentiles; white dot: median. **d,** paired out-of-sample comparison: Gain of the stochastic decision over the mean-value decision in each of 2000 independent samples from the distributions (positive: the stochastic decision is better). Black dot and bar: mean gain 0.952 and its 95% bootstrap interval (0.201 to 1.7; 10,000 resamples, treating scenarios as independent). The stochastic decision is better in 53% and worse in 47% of the scenarios.  
`fig_overview`

**Fig. 1 | Value of modelling the uncertainty.** Expected outcome of the stochastic decision (RP, orange square), of the mean-value decision evaluated over the scenarios (EEV, blue circle) and with perfect information (WS, green diamond); the open circle is the deterministic model's own estimate at the mean data (EV = 398.1). Brackets: value of the stochastic solution, VSS = 0.9 (0.30% of RP), and expected value of perfect information, EVPI = 92.4 (30.23% of RP). 400 scenarios.  
`fig_value` · source data: `source_data/fig_value.csv`

**Fig. 2 | First-stage decision.** Value of each first-stage variable in the mean-value model (blue circles) and in the stochastic model (orange squares); 1 of 1 variables differ.  
`fig_first_stage` · source data: `source_data/fig_first_stage.csv`

**Fig. 3 | Scenarios.** Values of the 1 uncertain entries across the 400 scenarios (sampled from the specified distributions), relative to their probability-weighted means. Shaded: probability-weighted density; bar: 25th–75th percentiles; line: 5th–95th percentiles; white dot: median.  
`fig_scenarios` · source data: `source_data/fig_scenarios.csv`

**Fig. 4 | Out-of-sample outcomes.** Realized value of each decision on 2000 independent samples from the distributions that were not used to make it. Box: 25th–75th percentiles with the median; whiskers: 5th–95th percentiles; marker: mean.  
`fig_out_of_sample` · source data: `source_data/fig_out_of_sample.csv`

**Fig. 5 | Paired out-of-sample comparison.** Gain of the stochastic decision over the mean-value decision in each of 2000 independent samples from the distributions (positive: the stochastic decision is better). Black dot and bar: mean gain 0.952 and its 95% bootstrap interval (0.201 to 1.7; 10,000 resamples, treating scenarios as independent). The stochastic decision is better in 53% and worse in 47% of the scenarios.  
`fig_gain` · source data: `source_data/fig_gain.csv`
