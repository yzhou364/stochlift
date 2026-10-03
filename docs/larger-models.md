# Larger models

StochLift solves the extensive form directly. Three things help as models grow.

## Parallel scenario solves

The wait-and-see solves, the evaluation of a fixed first stage in every scenario, and the
out-of-sample test are independent per scenario. They run on a thread pool (the solvers release
Python's lock while they work):

```python
study = sl.lift(build_model, DATA, spec="uncertainty.yaml", n_jobs=8)
```

or `--jobs 8` on the command line. The default is the number of CPUs, at most 8. The results do
not depend on the number of threads. On a facility-location model with 20 sites, 60 customers
and 50 scenarios, `solve()` went from 74 to 30 seconds.

## Gurobi

```python
study = sl.lift(build_model, DATA, spec="uncertainty.yaml", solver="gurobi")
```

or `--solver gurobi`. Every model, including the extensive form, is then solved with Gurobi
(it needs `gurobipy` and a license that covers the model size).

## Decomposition with mpi-sppy

For extensive forms too large to solve directly, hand the same scenarios to
[mpi-sppy](https://github.com/Pyomo/mpi-sppy), which has progressive hedging, Benders and other
decomposition methods:

```python
ex = study.to_mpisppy()      # all_scenario_names, scenario_creator, scenario_creator_kwargs
```

or write a module for `mpisppy.generic_cylinders`:

```bash
stochlift export model.py --spec uncertainty.yaml -o farmer_scen.py
python -m mpisppy.generic_cylinders --module-name farmer_scen --num-scens 100 --EF --EF-solver-name appsi_highs
```

The export works for models from any supported library: each scenario is rebuilt in Pyomo from
the matrix StochLift read. The objective is in minimization form (negated for a maximization
model). Decomposition needs `mpi4py` and an MPI installation; the extensive form does not.
The test suite checks that mpi-sppy's extensive form reproduces StochLift's RP and first-stage
decision. Mean-CVaR objectives are not exported (mpi-sppy has its own `--cvar` option).
