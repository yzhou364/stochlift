# The spec file

`uncertainty.yaml` is the one file a person reviews and signs off. It says which decisions are
made before the uncertainty is known, which data are uncertain, where the scenarios come from,
and, optionally, how risk-averse the decision should be. `stochlift init` writes a commented
template; `sl.Spec.load(...)` reads one; every report folder contains the spec that produced it.

```yaml
first_stage:
- open[*]
uncertain:
- demand
columns: {}
scenarios:
  method: kmeans
  n: 30
  seed: 0
holdout: 0.3
risk:
  alpha: 0.9
  weight: 0.5
notes: Sites are opened for the week before demand is known.
```

## Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `first_stage` | yes | Patterns on variable names. `*` matches any characters and `?` one character; everything else, including `[` and `]`, is literal, so `open[*]` matches Pyomo's `open[S1]`. Matching variables are decided once, before the uncertainty is known. All other variables are recourse decisions, chosen per scenario. |
| `uncertain` | with a history | Data keys whose values are uncertain: a top-level key (`demand`, every number under it) or a dotted entry (`demand.C1`, `yield.wheat`). With distributions it defaults to the keys under `scenarios.distributions`; with explicit scenarios, to the entries the scenarios override. |
| `columns` | no | Map from an uncertain entry to a history column, for when the names differ: `{"demand.C1": "north_store"}`. By default an entry matches a column with its dotted name (`demand.C1`) or its last part (`C1`) when that is unambiguous. |
| `scenarios` | no | Where the scenarios come from; see below. Default `{method: empirical}`. |
| `holdout` | no | Share of the history (its last rows) kept aside for the out-of-sample test, in `[0, 1)`. Rows are assumed to be in time order. Default 0. |
| `risk` | no | A mean-CVaR objective; see below. Absent means expected cost. |
| `notes` | no | Free text for the reviewer: why the stage split is what it is. |

## `scenarios`

| Key | Methods | Meaning |
| --- | --- | --- |
| `method` | all | `empirical`: every history row is a scenario. `sample`: `n` rows drawn with replacement. `kmeans`: `n` cluster centroids weighted by cluster size. `distribution`: `n` draws from `distributions`. `explicit` is set automatically when you pass `scenarios=` in Python. |
| `n` | sample, kmeans, distribution | Number of scenarios (default 100 for `distribution`). |
| `seed` | all random methods | Random seed. Default 0. |
| `distributions` | distribution | Map from data key or dotted entry to a marginal distribution. A more specific key overrides a less specific one. |
| `correlation` | distribution | One common correlation between all uncertain entries (Gaussian copula). Default 0. |
| `n_test` | distribution | Independent draws kept for the out-of-sample test. Default 500; 0 turns it off. |

### Marginal distributions

Parameters are relative to the entry's value in the data (its nominal value `v`) unless given in
absolute terms.

| `dist` | Parameters |
| --- | --- |
| `normal` | `mean` (default `v`), and `sd` or `cv` (sd = cv × \|mean\|) |
| `lognormal` | `mean` (default `v`, must be > 0) and `cv` |
| `uniform` | `low` and `high`, or `spread` (from `v·(1 − spread)` to `v·(1 + spread)`) |
| `triangular` | `low`, `mode` (default `v`) and `high`, or `spread` |
| `discrete` | `values` (absolute) or `factors` (multiples of `v`), and optional `probs` |

Every marginal also accepts `min` and `max` to clip the draws, for example `min: 0` for demand.

## `risk`

| Key | Meaning |
| --- | --- |
| `measure` | `cvar` (the only one so far). |
| `alpha` | CVaR level in `[0, 1)`: CVaR is the mean of the worst `1 − alpha` share of outcomes. Default 0.9. |
| `weight` | Weight on CVaR in `[0, 1]`: the objective is `(1 − weight) × expected cost + weight × CVaR`. Default 1 when `risk` is given. |

For a maximization model, "worst" means the lowest values. See [Risk-averse decisions](risk.md).

## What the checks cannot do

The solver checks verify that the lift is mathematically consistent with your model. They cannot
verify that `first_stage` matches how decisions are actually made: the bound ordering holds for
any split of the variables, so a wrong split passes it. That is what reading this file is for.
