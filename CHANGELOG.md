# Changelog

## 0.2.0 (unreleased)

- Scenarios from distributions when there is no history (`scenarios.method: distribution`):
  normal, lognormal, uniform, triangular and discrete marginals relative to the values in the
  data, optional clipping, and a common correlation through a Gaussian copula. Out-of-sample
  tests, stability runs and the optimality-gap estimate draw fresh independent samples.
  `stochlift.sample()` builds a scenario set directly.
- Command line: `stochlift init` writes a commented spec template from the model (or asks a
  language model with `--llm`); `stochlift run` solves, checks and writes the report, and exits
  with status 1 when a check fails.
- Data held in pandas Series and DataFrames. Integer arrays and columns are converted to float
  when an uncertain value is written into them.
- PuLP 4 support. The PuLP adapter raised `KeyError` on models built with PuLP 4.
- Figures say "out-of-sample" rather than "hold-out", since the samples can come from either.
- Python 3.10 or later is required.

## 0.1.0 (2026-09-30)

First version: two-stage stochastic programs from PuLP, Pyomo, gurobipy, OR-Tools, highspy and
`.lp`/`.mps` models; EV, WS, RP, EEV, VSS and EVPI; eight solver-checked invariants; scenarios
from a history with a hold-out test; stability and optimality-gap estimates; report with figures;
language-model proposal of the spec.
