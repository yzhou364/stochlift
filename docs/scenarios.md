# Scenarios

StochLift needs a set of scenarios: realisations of the uncertain data, each with a probability.
There are three ways to get them.

## Explicit scenarios

When you know the scenarios, pass them as `(probability, overrides)` pairs. `overrides` is a
partial copy of the data dictionary with the uncertain entries changed:

```python
scenarios = [(1/3, {"yield": {"wheat": 3.0, "corn": 3.6, "beets": 24.0}}),
             (1/3, {"yield": {"wheat": 2.5, "corn": 3.0, "beets": 20.0}}),
             (1/3, {"yield": {"wheat": 2.0, "corn": 2.4, "beets": 16.0}})]
study = sl.lift(build_model, DATA, first_stage=["acres_*"], scenarios=scenarios)
```

## From a history

With past observations of the uncertain data (one row per period), the spec chooses how to turn
them into scenarios, and `holdout` keeps the last rows aside to test the decision on data it has
not seen:

```python
study = sl.lift(build_model, DATA, spec="uncertainty.yaml", history="demand_history.csv")
study.solve()
study.out_of_sample()      # both decisions applied to the held-out rows
```

`kmeans` keeps the mean of the history exactly but shrinks its tails. Compare with the
out-of-sample test before trusting a small `n`.

## From distributions

Without a history, describe how uncertain each number is relative to its value in the data:

```yaml
scenarios:
  method: distribution
  n: 100
  n_test: 500
  correlation: 0.8
  distributions:
    yield: {dist: normal, cv: 0.15, min: 0}
```

The out-of-sample test, the stability runs and the optimality-gap estimate then use fresh
independent draws. In Python, `sl.sample(DATA, {"yield": {"dist": "normal", "cv": 0.15}}, n=100,
correlation=0.8)` returns a scenario set to pass as `scenarios=`.

## How many scenarios?

`study.stability(sizes=(5, 10, 20, 40, 80))` re-solves with random scenario sets of each size and
shows how the optimum and its out-of-sample result settle. `study.saa_gap(n=50, batches=20)`
estimates the optimality gap of the solution (Mak, Morton and Wood, 1999).
