# StochLift

[![tests](https://github.com/yzhou364/stochlift/actions/workflows/tests.yml/badge.svg)](https://github.com/yzhou364/stochlift/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/stochlift.svg)](https://pypi.org/project/stochlift/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23101099.svg)](https://doi.org/10.5281/zenodo.23101099)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Documentation: https://yzhou364.github.io/stochlift/**

Turn the deterministic optimization model you already have into a two-stage stochastic
program with one command, check the result with the solver, and find out whether modeling
the uncertainty is worth it.

Status: v0.2, alpha. Two-stage stochastic programs with linear and mixed-integer linear
models. Python 3.10+, Windows, macOS and Linux.

## Why

Stochastic programming tools expect the model in their own form: rewritten in their syntax,
annotated with stages and random parameters, or wrapped in a scenario-creator function.
Practitioners usually have something else: a deterministic model that already runs in PuLP,
gurobipy, OR-Tools or Pyomo, plus a rough idea (or a history) of the numbers that turn out to
be wrong. StochLift starts there, without annotations or changes to the model, and answers
three questions:

1. **What does the stochastic version of my model decide?** It is built from your own model,
   without rewriting it.
2. **Is the lift correct?** Eight invariants from stochastic programming theory are checked by
   the solver. None of them needs a reference answer.
3. **Is it worth it?** The value of the stochastic solution (VSS) and of perfect information
   (EVPI), and an out-of-sample test on data that was not used to make the decision. If the
   gain is not distinguishable from noise, the report says so.

You keep your model exactly as it is. The only requirement is that it is built by a function
of its data:

```python
def build_model(data):      # returns a PuLP, Pyomo, gurobipy, OR-Tools or highspy model
    ...
```

`data` is a dictionary whose values can be numbers, nested dicts and lists, NumPy arrays, or
pandas Series and DataFrames.

StochLift calls that function once per scenario, shares the first-stage variables across
scenarios, and builds the extensive form. It never needs to know where the uncertain numbers
enter the model.

## Install

```bash
pip install "stochlift[pulp]"      # or [pyomo], or [all]; gurobipy, OR-Tools and highspy models also work
```

For development:

```bash
git clone https://github.com/yzhou364/stochlift.git && cd stochlift
pip install -e ".[dev]"
pytest
```

HiGHS is the solver (installed with `highspy`); no commercial license is needed.

## Quick start from the command line

Put `build_model` and a `DATA` dictionary in `model.py`, then:

```bash
stochlift init model.py          # inspects the model, writes uncertainty.yaml with TODOs to edit
stochlift run model.py           # solves, checks, writes report/ (summary, LaTeX table, figures)
```

`stochlift init` lists the variable groups and where each data key enters the model, so you can
decide what is first-stage and what is uncertain. Add `--history demand.csv` if you have past
observations, or `--llm anthropic:<model id>` to have a language model fill in the proposal.
`stochlift run` exits with status 1 if a check fails. See `stochlift run --help` for stability
runs (`--stability 5,10,20,40`) and the optimality-gap estimate (`--gap 50`).

## A first example: the farmer problem

`examples/farmer/model.py` is the deterministic model from Birge and Louveaux. Lifting it:

```python
import stochlift as sl
from model import DATA, build_model

scenarios = [(1/3, {"yield": {crop: f * y for crop, y in DATA["yield"].items()}})
             for f in (1.2, 1.0, 0.8)]

study = sl.lift(build_model, DATA, first_stage=["acres_*"], scenarios=scenarios)
study.review()            # the spec, the stage split, and where the uncertain data enters the model
results = study.solve()   # EV, WS, RP, EEV, VSS, EVPI
study.check()             # solver-checked invariants
study.report("report/")   # summary.md, results.json, LaTeX table, figures (PDF + PNG)
```

This reproduces the published values: expected profit 108,390 for the stochastic solution,
115,406 with perfect information, 107,240 for the mean-value solution, so VSS = 1,150 and
EVPI = 7,016. The test suite asserts these numbers.

## No history? Sample from distributions

Most people know roughly how uncertain a number is ("demand varies about 20%") without having a
table of past values. Give each uncertain entry a distribution relative to its value in the data:

```yaml
first_stage:
- acres_*
scenarios:
  method: distribution
  n: 100              # scenarios in the stochastic program
  n_test: 500         # independent samples for the out-of-sample test
  seed: 0
  correlation: 0.8    # weather moves all yields together
  distributions:
    yield: {dist: normal, cv: 0.15, min: 0}       # every entry under 'yield'
    # yield.beets: {dist: uniform, spread: 0.3}   # a single entry overrides its group
```

Marginals: `normal` (`cv` or `sd`), `lognormal` (`cv`), `uniform` (`spread`, or `low`/`high`),
`triangular` (`spread`, or `low`/`mode`/`high`) and `discrete` (`factors` or `values`, with
`probs`). The mean defaults to the value in the data. The decision is then tested on fresh
samples that were not used to make it. For the farmer problem
(`examples/farmer/uncertainty_distribution.yaml`):

```text
RP -111,661  EEV -110,721  WS -118,167  VSS 940 (0.84%)  EVPI 6,506 (5.83%)
out of sample (500 independent samples): mean gain 807, 95% interval [449, 1,175]
```

In Python, `sl.sample(DATA, {"yield": {"dist": "normal", "cv": 0.15}}, n=100, correlation=0.8)`
returns a scenario set to pass as `scenarios=`.

## Risk-averse decisions (CVaR)

The expected cost is not always what matters: a decision that is slightly cheaper on average
can be much worse in a bad season. Add a `risk` block to the spec to optimize
`(1 - weight) x expected cost + weight x CVaR`, where CVaR at `alpha` is the mean of the worst
`1 - alpha` share of outcomes (for a maximization model, the lowest values):

```yaml
risk:
  alpha: 0.9      # the worst 10% of outcomes
  weight: 0.5     # half expected cost, half CVaR
```

The model stays linear (Rockafellar-Uryasev). WS, RP and EEV are all measured with the same
objective, so the bound ordering and the other checks still apply, and the out-of-sample test
compares the risk-adjusted objective as well as the mean. `--frontier 0,0.25,0.5,0.75,1` (or
`study.risk_frontier(...)`) traces the trade-off between expected cost and CVaR.

For the farmer problem (`examples/farmer/uncertainty_cvar.yaml`), the risk-averse decision does
worse on average out of sample (by 142, interval -208 to -76) and better on the risk-adjusted
objective (by 245, interval 121 to 346). A test of the mean alone would have rejected it.

## With a history of observations

`examples/facility_location` is a Pyomo model that chooses which sites to operate for a week
and then ships to customers. `demand_history.csv` holds 260 weeks of **synthetic** demand. The
spec is one reviewed file:

```yaml
first_stage:
- open[*]
uncertain:
- demand
scenarios:
  method: kmeans
  n: 30
  seed: 0
holdout: 0.3
```

```python
study = sl.lift(build_model, DATA, spec="uncertainty.yaml", history="demand_history.csv")
study.solve(); study.check()
study.out_of_sample()      # both decisions applied to the 78 held-out weeks
study.stability(sizes=(5, 10, 20, 40, 80), reps=10)
study.saa_gap(n=50, batches=20)
study.report("report/")
```

What the report says for this example (`python run.py`, about one minute):

| | Outsourcing at 80 per unit | Outsourcing at 45 per unit |
| --- | ---: | ---: |
| VSS on the 30 scenarios | 669 (10.5% of RP) | 67 (1.1% of RP) |
| Mean gain on 78 held-out weeks | 283 | -78 |
| 95% bootstrap interval | -4 to 596 | -218 to 72 |
| Verdict in the report | not clearly | not clearly |

The in-sample VSS looks large in the first column, but on unseen weeks the interval still
touches zero, so the report does not claim a gain. That is the point of the hold-out test:
VSS computed on the scenarios that produced the decision is optimistic.

## Examples

| Folder | Library | Shows |
| --- | --- | --- |
| `examples/farmer` | PuLP | textbook values (Birge and Louveaux), distributions, CVaR |
| `examples/facility_location` | Pyomo | binary first stage, a demand history with a hold-out test |
| `examples/newsvendor_ortools` | OR-Tools | the order equals the critical-ratio quantile (`check_closed_form.py`) |
| `examples/capacity_expansion_gurobi` | gurobipy | generation capacity under correlated load, mean-CVaR frontier |

Each folder has a generated `report/`. Run any of them with `stochlift run model.py` from the folder.

## Larger models

- Scenario-by-scenario solves (perfect information, evaluating a fixed first stage, the
  out-of-sample test) run in parallel threads: `n_jobs=...` or `--jobs N` (default: the number
  of CPUs, at most 8). Results do not depend on the number of threads.
- `solver="gurobi"` (or `--solver gurobi`) solves every model with Gurobi instead of HiGHS.
- When the extensive form is too large, hand the same scenarios to
  [mpi-sppy](https://github.com/Pyomo/mpi-sppy) for decomposition. `study.to_mpisppy()` gives
  its scenario creator, and `stochlift export model.py` writes a module for
  `mpisppy.generic_cylinders`. This works for models from any supported library. The test suite
  checks that mpi-sppy's own extensive form reproduces StochLift's RP and first-stage decision.

## When there is no solution

If the stochastic program is infeasible, the error says why in terms of your data:

```text
1 of 4 scenarios have no optimal solution even when solved on their own (infeasible), so no
stochastic program over them can be solved.
  scenario 2: demand = 130 (mean 77.5)
The deterministic model itself has no solution for these data. Typical fixes: allow shortfalls
with slack variables at a penalty cost ...
```

When every scenario is feasible on its own but no first-stage decision suits them all, it names
a pair of scenarios that conflict.

## What is checked

`study.check()` needs no reference answer. Each item is computed by the solver:

| Check | What it catches |
| --- | --- |
| `builder_deterministic` | the builder uses random numbers or hidden state |
| `uncertainty_reaches_model` | a data key named as uncertain does not change the model |
| `stage_split` | no first-stage variables, or no recourse variables |
| `first_stage_rows_stable` | uncertain data constrains first-stage variables directly |
| `single_scenario_reduction` | with one scenario the lift must equal the deterministic model |
| `bound_ordering` | WS <= RP <= EEV (for minimization) |
| `decomposition_consistency` | the extensive-form optimum equals the scenario-by-scenario cost of its first stage |
| `mean_value_recourse` | the mean-value decision is infeasible in some scenario |

These checks verify that the lift is mathematically consistent with your model. They do
**not** verify that the stage assignment matches how decisions are made in practice. The
bound ordering holds for any split of the variables, so a wrong split passes it. A person
must review `uncertainty.yaml`.

## Supported modeling libraries

| Library | Read from | Tested here |
| --- | --- | --- |
| PuLP 2, 3 and 4 | the model object | yes |
| Pyomo | the model object (linear expressions only) | yes |
| gurobipy | the model object | yes, with the size-limited license that ships with the package |
| OR-Tools `pywraplp` | the model object | yes, in a separate process (see below) |
| highspy | the `Highs` object | yes |
| `.lp` / `.mps` file | HiGHS reader | yes |

Some OR-Tools builds on Linux bundle their own HiGHS and cannot be loaded in the same process as
`highspy`. Import your OR-Tools model before StochLift solves anything (as a script normally
does); StochLift then solves with SciPy's bundled HiGHS. In a notebook, restart the kernel if
OR-Tools fails to load after a solve. Models are read from the
library objects, not through MPS files, because MPS writers drop the objective sense and
constant.

## Optional: a language model drafts the spec

Nothing in StochLift requires a language model. Deciding what is first-stage and what is
uncertain is usually quick for the person who knows the business, and `stochlift init` lists
everything needed to decide. For large models with many variable groups, a language model can
draft the spec instead (`stochlift init model.py --llm anthropic:<model id>`). Its proposal is
checked against the real model, and a person still reviews it.

```python
from stochlift.llm import anthropic_llm      # or openai_llm, or any callable prompt -> reply

study = sl.lift(build_model, DATA, history="demand_history.csv",
                llm=anthropic_llm(model="<model id>"))
study.review()     # read the proposal before you trust it
```

The prompt contains the model source, a summary of the data, the variable groups, where each
data key enters the model (found by perturbing it and rebuilding), and the history columns.
The reply is parsed, validated against the real model, and sent back with the list of
problems if it is unusable (up to three rounds).

How far this has been tested: the parser, the validation and the retry loop are covered by
scripted replies. Two real replies from a language model (one per example) are stored in
`tests/fixtures` and replayed; both gave a usable spec on the first attempt. That is a smoke
test on two easy models, not a measurement of accuracy. `anthropic_llm` and `openai_llm` are
thin wrappers that have not been run against the live APIs in this repository's tests.

## How StochLift relates to other tools

| | Input | Model changes needed | VSS / EVPI | Out-of-sample verdict | Checks of the lift |
| --- | --- | --- | --- | --- | --- |
| **StochLift** | PuLP, Pyomo, gurobipy, OR-Tools, highspy, `.lp`/`.mps` | none: the unchanged `build_model(data)` | both | paired test with bootstrap interval | 8 solver-checked invariants |
| [mpi-sppy](https://github.com/Pyomo/mpi-sppy) | Pyomo; AMPL, GAMS, gurobipy guests (alpha); MPS + JSON; SMPS | a `scenario_creator` declaring the first stage | VSS (`--vss`) | MMW and bootstrap confidence intervals | configuration checks |
| [StochasticPrograms.jl](https://github.com/martinbiel/StochasticPrograms.jl) | Julia | rewrite with its macros | both | SAA confidence intervals | no |
| GAMS EMP SP, AIMMS, LINGO | their own languages | annotate random parameters and stages | LINGO: both | no | no |

Use **mpi-sppy** for large models: it has decomposition (progressive hedging, Benders),
parallel computing and a much larger set of algorithms. StochLift solves the extensive form
directly and is meant for the step before that: finding out, from the model you already have,
whether a stochastic model is worth building, and getting a lift you can trust. Many of its
statistical tools (Mak-Morton-Wood gap, CVaR) are standard, and are also in mpi-sppy.

## Limitations

- Two stages only. No multi-stage models, chance constraints or robust counterparts yet.
- Linear and mixed-integer linear models only.
- The extensive form is solved directly; for decomposition, export to mpi-sppy (see above).
- The builder is called once per distinct scenario. A slow builder makes everything slow.
- The data must be a dictionary; uncertain entries must be numbers in nested dicts, lists,
  NumPy arrays, or pandas Series and DataFrames.
- Distribution-based scenarios use one common correlation for all uncertain entries.
- CVaR is estimated from the scenarios in its tail. With `alpha: 0.9` and 100 scenarios that is
  10 scenarios, so the in-sample CVaR is optimistic; compare it with the out-of-sample values.
- `kmeans` scenarios keep the mean but shrink the tails. Compare with the hold-out test.
- The hold-out split takes the last rows of the history. It assumes rows are in time order.
- Out-of-sample intervals are bootstrap intervals of the mean paired gain. They treat
  observations as independent.

## Roadmap

Robust and chance-constrained lifts, decomposition through mpi-sppy, multi-stage models, an
MCP server, and LiftBench, a benchmark of deterministic models with known stochastic versions.

## Contributing

See `CONTRIBUTING.md`. If you lifted a model of your own, please tell us about it with the
"I lifted my model" issue template, even if everything worked.

## Citing

If StochLift helps your work, please cite the software (this DOI always resolves to the latest
version; GitHub's "Cite this repository" button gives the same entry):

```bibtex
@software{zhou_stochlift,
  author    = {Zhou, Yuqun},
  title     = {StochLift: lifting deterministic optimization models to verified two-stage
               stochastic programs},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23101099},
  url       = {https://github.com/yzhou364/stochlift}
}
```

A paper describing the method is planned.

## License

MIT.
