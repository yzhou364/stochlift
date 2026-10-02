# StochLift

Turn the deterministic optimization model you already have into a two-stage stochastic
program, check the result with the solver, and find out whether modeling the uncertainty
is worth it.

Status: v0.2, alpha. Two-stage stochastic programs with linear and mixed-integer linear
models. Python 3.10+, Windows, macOS and Linux. Not yet on PyPI.

## Why

Most tools that combine language models with optimization start from a text description of
a problem. Practitioners usually start somewhere else: a deterministic model that already
runs, plus some history of the numbers that turned out to be wrong. StochLift starts there.

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
git clone <your fork> && cd stochlift
pip install -e ".[pulp,pyomo]"
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

## Letting a language model propose the spec

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

## Supported modeling libraries

| Library | Read from | Tested here |
| --- | --- | --- |
| PuLP 2, 3 and 4 | the model object | yes |
| Pyomo | the model object (linear expressions only) | yes |
| gurobipy | the model object | yes, with the size-limited license that ships with the package |
| OR-Tools `pywraplp` | the model object | yes, in a separate process (see below) |
| highspy | the `Highs` object | yes |
| `.lp` / `.mps` file | HiGHS reader | yes |

Some OR-Tools builds bundle their own HiGHS and cannot be imported in the same process as
`highspy`. StochLift then solves with SciPy's bundled HiGHS instead. Models are read from the
library objects, not through MPS files, because MPS writers drop the objective sense and
constant.

## Limitations

- Two stages only. No multi-stage models, chance constraints or robust counterparts yet.
- Linear and mixed-integer linear models only.
- The extensive form is solved directly. There is no decomposition, so very large scenario
  counts or very large models will be slow.
- The builder is called once per distinct scenario. A slow builder makes everything slow.
- The data must be a dictionary; uncertain entries must be numbers in nested dicts, lists,
  NumPy arrays, or pandas Series and DataFrames.
- Distribution-based scenarios use one common correlation for all uncertain entries.
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

See `CITATION.cff`. A paper describing the method is planned.

## License

MIT.
