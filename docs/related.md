# Related tools

| | Input | Model changes needed | VSS / EVPI | Out-of-sample verdict | Checks of the lift |
| --- | --- | --- | --- | --- | --- |
| **StochLift** | PuLP, Pyomo, gurobipy, OR-Tools, highspy, `.lp`/`.mps` | none: the unchanged `build_model(data)` | both | paired test with bootstrap interval | 8 solver-checked invariants |
| [mpi-sppy](https://github.com/Pyomo/mpi-sppy) | Pyomo; AMPL, GAMS, gurobipy guests (alpha); MPS + JSON; SMPS | a `scenario_creator` declaring the first stage | VSS (`--vss`) | MMW and bootstrap confidence intervals | configuration checks |
| [StochasticPrograms.jl](https://github.com/martinbiel/StochasticPrograms.jl) | Julia | rewrite with its macros | both | SAA confidence intervals | no |
| GAMS EMP SP, AIMMS, LINGO | their own languages | annotate random parameters and stages | LINGO: both | no | no |

(As of October 2026; these projects move, so check their documentation.)

**Use mpi-sppy for large models.** It has decomposition (progressive hedging, Benders), parallel
computing and a much larger set of algorithms. StochLift is meant for the step before that:
finding out, from the model you already have, whether a stochastic model is worth building, and
getting a lift you can trust. It can then [export the scenarios to
mpi-sppy](larger-models.md#decomposition-with-mpi-sppy). Many of its statistical tools
(the Mak-Morton-Wood gap, CVaR) are standard and are also in mpi-sppy.

**Language models.** Recent work generates stochastic or robust models from text descriptions
(for example DAOpt, arXiv:2511.11576, and arXiv:2508.17200). StochLift starts from model code
instead, and a language model is optional: `stochlift init --llm anthropic:<model id>` drafts the
spec, which is then validated against the real model and reviewed by a person.
