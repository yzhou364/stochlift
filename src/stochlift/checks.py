"""Solver-checked invariants.

None of these needs a reference answer. They verify that the lift is
mathematically consistent with the user's own deterministic model. They do
*not* prove that the stage assignment is the right business reading of the
problem: that is what the human review of the spec is for.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .adapters import to_linear_model
from .lift import extensive_form, first_stage_row_report
from .model import solve


@dataclass
class Check:
    name: str
    status: str      # pass | warn | fail
    detail: str

    def __str__(self) -> str:
        return f"[{self.status.upper():4}] {self.name}: {self.detail}"


def run_checks(study) -> list:
    out = []
    r = study.results
    z = study._internal
    S = study.scenarios
    models = study.models()
    mean = study.mean_model()
    tol = (1e-6 + 2 * study.opts["mip_gap"]) * max(1.0, abs(z["z_rp"]))

    # 1. the builder is a pure function of its data
    a = to_linear_model(study.build_model(copy.deepcopy(study.data)))
    b = to_linear_model(study.build_model(copy.deepcopy(study.data)))
    out.append(Check("builder_deterministic", "pass" if a.same_as(b) else "fail",
                     "two builds from the same data give the same model" if a.same_as(b) else
                     "two builds from the same data differ; the builder must not use random "
                     "numbers or hidden state"))

    # 2. every uncertain entry actually changes the model
    probe = study.probe_uncertain()
    dead = [nm for d in probe.values() if not d.get("error") for nm in d.get("dead", [])]
    broken = [k for k, d in probe.items() if d.get("error")]
    reshaped = [k for k, d in probe.items() if d.get("structure_changed")]
    if dead or broken:
        msg = []
        if dead:
            msg.append(f"changing {dead[:6]} leaves the model unchanged, so it cannot be an uncertain input")
        if broken:
            msg.append("; ".join(f"{k}: {probe[k]['error']}" for k in broken))
        out.append(Check("uncertainty_reaches_model", "fail", "; ".join(msg)))
    elif reshaped:
        out.append(Check("uncertainty_reaches_model", "warn",
                         f"changing {reshaped} changes the set of variables or constraints, "
                         "not only coefficients; check that scenarios stay comparable"))
    else:
        where = "; ".join(f"{k} -> " + ", ".join(f"{n} {part}" for part, n in
                          (("objective", d["objective"]), ("matrix", d["matrix"]),
                           ("rhs", d["rhs"]), ("bounds", d["bounds"])) if n)
                          for k, d in probe.items())
        out.append(Check("uncertainty_reaches_model", "pass", f"coefficients affected: {where}"))

    # 3. the stage split is a real split
    n_first = sum(study.is_first(nm) for nm in mean.names)
    if n_first == 0 or n_first == mean.n:
        out.append(Check("stage_split", "fail",
                         f"{n_first} of {mean.n} variables are first-stage; a two-stage model "
                         "needs both here-and-now and recourse variables"))
    else:
        out.append(Check("stage_split", "pass",
                         f"{n_first} first-stage and {mean.n - n_first} recourse variables"))

    # 4. uncertain data should not constrain the first stage on its own
    changed = first_stage_row_report(models, study.is_first)
    if changed:
        out.append(Check("first_stage_rows_stable", "warn",
                         f"constraints on first-stage variables only change with the scenario "
                         f"({', '.join(changed[:5])}); the first stage must then satisfy every "
                         "scenario's version at once"))
    else:
        out.append(Check("first_stage_rows_stable", "pass",
                         "constraints on first-stage variables only are the same in every scenario"))
    for note in study.lift_notes:
        out.append(Check("lift_note", "warn", note))

    # 5. with a single scenario the lift must reproduce the deterministic model
    ef1 = extensive_form([mean], [1.0], study.is_first, study.risk)
    s1 = solve(ef1.model, **study.opts)
    ok = s1.ok and abs(s1.objective - z["z_ev"]) <= tol
    out.append(Check("single_scenario_reduction", "pass" if ok else "fail",
                     f"one-scenario lift = {z['sign'] * s1.objective:.6g}, deterministic model = "
                     f"{z['sign'] * z['z_ev']:.6g}"))

    # 6. WS <= RP <= EEV (in minimization form)
    ok = z["z_ws"] <= z["z_rp"] + tol and z["z_rp"] <= z["z_eev"] + tol
    order = "WS <= RP <= EEV" if z["sign"] > 0 else "WS >= RP >= EEV"
    out.append(Check("bound_ordering", "pass" if ok else "fail",
                     f"{order}: WS = {r.ws:.6g}, RP = {r.rp:.6g}, EEV = {r.eev:.6g}"))

    # 7. the extensive-form optimum equals the scenario-by-scenario cost of its first stage
    back = study.risk.value(z["costs_rp"], S.probs)
    ok = np.isfinite(back) and abs(back - z["z_rp"]) <= tol
    out.append(Check("decomposition_consistency", "pass" if ok else "fail",
                     f"extensive form = {r.rp:.6g}; re-solving each scenario with the first stage "
                     f"fixed gives {z['sign'] * back:.6g}"))

    # 8. does the mean-value decision survive every scenario?
    if r.eev_infeasible:
        out.append(Check("mean_value_recourse", "warn",
                         f"the mean-value first stage has no feasible recourse in "
                         f"{r.eev_infeasible} of {r.n_scenarios} scenarios, so EEV is infinite; "
                         "add slack/penalty variables if shortfalls are allowed in practice"))
    else:
        out.append(Check("mean_value_recourse", "pass",
                         "the mean-value first stage is feasible in every scenario"))
    return out


def all_passed(checks) -> bool:
    return all(c.status != "fail" for c in checks)
