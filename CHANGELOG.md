# Changelog

## Unreleased

- Journal-ready figures. The default `nature` style draws every figure at its final print size
  (89 mm, 183 mm for the overview), in Arial at 5-7 pt, without titles, with TrueType fonts in PDF
  and text elements in SVG. Large numbers are scaled (×10³) with the factor in the axis label.
- A multi-panel overview figure (`fig_overview`, panels a-f), a draft legend for every figure in
  `captions.md`, and the plotted numbers in `source_data/`.
- Redesigned panels: scenario densities with quartiles (and the history, when there is one),
  out-of-sample box plots, stability bands, overlapping first-stage markers kept visible.
- `study.figures(outdir, style=...)`, `report(..., style=...)` and `--style presentation`; each
  panel is a drawing function that can be combined into custom figures.
- Every figure is PDF, SVG and 600-dpi PNG. Tests check sizes, font sizes, line widths, titles,
  fonts and source data.

## 0.3.0 (2026-10-02)

- Parallel scenario solves on a thread pool (`n_jobs`, `--jobs`): 2.4x faster on a 50-scenario
  facility-location model, with identical results.
- Gurobi backend (`solver="gurobi"`, `--solver gurobi`), checked against HiGHS on random LPs and
  MIPs and on the farmer problem.
- Export to mpi-sppy for decomposition: `study.to_mpisppy()` and `stochlift export`. mpi-sppy's
  extensive form reproduces RP and the first-stage decision on the farmer and facility examples.
- Infeasible stochastic programs are explained: scenarios that are infeasible or unbounded on
  their own are named with their unusual data, and when no common first stage exists a
  conflicting pair of scenarios is reported.
- New examples: newsvendor in OR-Tools (checked against the critical-ratio quantile) and
  generation capacity expansion in gurobipy (mean-CVaR frontier).
- Compact axis labels for large numbers; the mean-risk figure states the mean-value decision in
  text when plotting it would squash the frontier.
- The command line prints numbers with thousands separators.

## 0.2.0 (2026-10-02)

- Scenarios from distributions when there is no history (`scenarios.method: distribution`):
  normal, lognormal, uniform, triangular and discrete marginals relative to the values in the
  data, optional clipping, and a common correlation through a Gaussian copula. Out-of-sample
  tests, stability runs and the optimality-gap estimate draw fresh independent samples.
  `stochlift.sample()` builds a scenario set directly.
- Mean-CVaR objectives (`risk: {alpha, weight}` in the spec), linearized in the extensive form.
  WS, RP and EEV use the same objective, so the checks still apply; the out-of-sample test adds
  the risk-adjusted gain with a bootstrap interval. `risk_frontier()` / `--frontier` traces the
  trade-off between expected cost and CVaR, with a new figure.
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
