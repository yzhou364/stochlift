# Checks and diagnostics

## The eight checks

`study.check()` (and every `stochlift run`) computes these with the solver. None needs a
reference answer.

| Check | What it catches |
| --- | --- |
| `builder_deterministic` | the builder uses random numbers or hidden state |
| `uncertainty_reaches_model` | a data key named as uncertain does not change the model |
| `stage_split` | no first-stage variables, or no recourse variables |
| `first_stage_rows_stable` | uncertain data constrains first-stage variables directly |
| `single_scenario_reduction` | with one scenario the lift must equal the deterministic model |
| `bound_ordering` | WS ≤ RP ≤ EEV (for minimization) |
| `decomposition_consistency` | the extensive-form optimum equals the scenario-by-scenario cost of its first stage |
| `mean_value_recourse` | the mean-value decision is infeasible in some scenario |

`stochlift run` exits with status 1 when a check fails, so it can guard a pipeline.

!!! warning "What the checks cannot do"
    They verify that the lift is consistent with your model, not that the stage split matches
    how decisions are made in practice. The bound ordering holds for any split. A person must
    review the spec.

## When there is no solution

If the stochastic program is infeasible, StochLift says why in terms of your data.

**Scenarios infeasible on their own.** The deterministic model has no solution for some
scenarios' data, so no stochastic program over them can work:

```text
1 of 4 scenarios have no optimal solution even when solved on their own (infeasible), so no
stochastic program over them can be solved.
  scenario 2: demand = 130 (mean 77.5)
The deterministic model itself has no solution for these data. Typical fixes: allow shortfalls
with slack variables at a penalty cost (unmet demand, overtime, outsourcing) in the deterministic
model, or narrow the distributions (min/max under scenarios.distributions).
```

**No common first stage.** Every scenario has a solution, but no single first-stage decision is
feasible in all of them (the recourse is not complete). StochLift adds scenarios one at a time
and names a pair that already conflicts. The errors are `stochlift.diagnose.ScenarioInfeasible`
and `stochlift.diagnose.NoCommonFirstStage`; both carry the scenario indices in `.scenarios`.
