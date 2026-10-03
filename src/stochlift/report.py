"""Text review of a study, and the report written to disk."""
from __future__ import annotations

import json
import os

import numpy as np

from . import datautil as du


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "inf" if v > 0 else ("-inf" if v < 0 else "nan")
    return f"{v:,.2f}" if abs(v) < 1e4 else f"{v:,.0f}"


def review_text(study) -> str:
    """What a person should read before trusting the lift."""
    mean = study.mean_model()
    first = [nm for nm in mean.names if study.is_first(nm)]
    second = [nm for nm in mean.names if not study.is_first(nm)]
    S = study.scenarios
    lines = ["Uncertainty specification", "-------------------------", study.spec.to_yaml().rstrip(), "",
             f"Model: {mean.n} variables ({int(mean.integer.sum())} integer), {mean.m} constraints, "
             f"sense = {mean.sense}",
             f"First stage  ({len(first)}): {', '.join(first[:8])}{' ...' if len(first) > 8 else ''}",
             f"Second stage ({len(second)}): {', '.join(second[:8])}{' ...' if len(second) > 8 else ''}",
             "", f"Uncertain entries ({len(S.paths)}) and scenarios ({len(S)}):"]
    mu, lo, hi = S.mean(), S.values.min(axis=0), S.values.max(axis=0)
    for j, name in enumerate(S.names[:12]):
        lines.append(f"  {name}: mean {mu[j]:.4g}, range [{lo[j]:.4g}, {hi[j]:.4g}]")
    if len(S.paths) > 12:
        lines.append(f"  ... and {len(S.paths) - 12} more")
    if study.train is not None:
        n_test = 0 if study.test is None else len(study.test)
        lines.append(f"History: {len(study.train)} observations for scenarios, {n_test} held out")
    if study.sampler is not None:
        n_test = 0 if study.test is None else len(study.test)
        lines.append(f"Scenarios sampled from distributions (correlation {study.sampler.correlation:g}); "
                     f"{n_test} independent samples kept for the out-of-sample test:")
        for item in study.sampler.describe()[:12]:
            lines.append(f"  {item}")
    lines += ["", "Where the uncertain data enters the model:"]
    for k, d in study.probe_uncertain().items():
        if d.get("error"):
            lines.append(f"  {k}: {d['error']}")
        elif d.get("structure_changed"):
            lines.append(f"  {k}: changes the set of variables or constraints")
        else:
            parts = [f"{d[p]} {p}" for p in ("objective", "matrix", "rhs", "bounds") if d[p]]
            lines.append(f"  {k}: {', '.join(parts) or 'nothing'}"
                         f" (e.g. rows {d['example_rows']}, variables {d['example_vars']})")
    return "\n".join(lines)


def verdict(study) -> str:
    """One paragraph answering the question, preferring hold-out evidence when it exists."""
    r, o = study.results, study.oos
    if r.eev_infeasible:
        return (f"**Yes.** The decision from the deterministic (mean-value) model has no feasible "
                f"recourse in {r.eev_infeasible} of {r.n_scenarios} scenarios. The stochastic model "
                "is needed to get a decision that works in every scenario.")
    if r.vss <= 1e-6 * max(1.0, abs(r.rp)):
        return ("**No.** On these scenarios the deterministic (mean-value) decision is already "
                "optimal for the stochastic model (VSS = 0). Keep the deterministic model.")
    how = "in expectation" if r.risk_parts is None else f"in the objective ({r.objective})"
    in_sample = (f"On the scenario set the stochastic decision is better by {_fmt(r.vss)} per decision "
                 f"{how} ({r.vss_pct:.2f}% of RP); this is the value of the stochastic "
                 "solution (VSS).")
    if not o or o["n_compared"] < 2:
        return ("**Yes, on the scenario set.** " + in_sample + " There is no hold-out data, so this "
                "has not been tested on observations outside the scenario set.")
    if "risk_gain" in o:
        lo, hi = o["risk_gain_ci95"]
        test = (f"On {o['n_compared']} {o.get('source', 'held-out observations')} the same "
                f"objective improves by {_fmt(o['risk_gain'])} "
                f"(95% interval {_fmt(lo)} to {_fmt(hi)}); the mean gain is {_fmt(o['mean_gain'])}.")
    else:
        lo, hi = o["gain_ci95"]
        test = (f"On {o['n_compared']} {o.get('source', 'held-out observations')} the mean gain is "
                f"{_fmt(o['mean_gain'])} (95% interval {_fmt(lo)} to {_fmt(hi)}).")
    if lo > 0:
        return "**Yes.** " + in_sample + " " + test
    if hi < 0:
        return ("**No.** " + in_sample + " " + test + " The stochastic decision did worse on unseen "
                "data: the scenario set does not represent the hold-out period, or it is too small.")
    return ("**Not clearly.** " + in_sample + " " + test + " The interval contains zero, so the "
            "gain is not distinguishable from noise on unseen data.")


def summary_markdown(study) -> str:
    r = study.results
    better = "lower" if r.sense == "min" else "higher"
    unit = "cost" if r.sense == "min" else "value"
    col = "Expected " + unit if r.risk_parts is None else "Objective"
    L = ["# StochLift report", "",
         f"Objective sense: **{r.sense}** ({better} is better). Scenarios: **{r.n_scenarios}**.", ""]
    if r.risk_parts is not None:
        L += [f"Risk-averse objective: **{r.objective}**. EV, WS, RP and EEV below are values of "
              "this objective, so VSS and EVPI are risk-adjusted too.", ""]
    L += [
         "## Is modeling the uncertainty worth it?", ""]
    L.append(verdict(study))
    L += ["",
          f"Better forecasts are worth at most {_fmt(r.evpi)} ({r.evpi_pct:.2f}% of RP): "
          "the expected value of perfect information (EVPI).", "",
          f"| Quantity | Meaning | {col} |", "| --- | --- | ---: |",
          f"| EV | deterministic model at mean data (its own, optimistic estimate) | {_fmt(r.ev)} |",
          f"| WS | wait-and-see: each scenario solved with perfect information | {_fmt(r.ws)} |",
          f"| RP | stochastic (recourse) solution | {_fmt(r.rp)} |",
          f"| EEV | the mean-value decision evaluated over the scenarios | {_fmt(r.eev)} |",
          f"| VSS | EEV vs RP | {_fmt(r.vss)} |",
          f"| EVPI | RP vs WS | {_fmt(r.evpi)} |", ""]
    if r.risk_parts is not None:
        a = study.risk.alpha
        L += [f"| Decision | Expected {unit} | CVaR at {a:g} |", "| --- | ---: | ---: |"]
        for k, label in (("RP", "stochastic (risk-averse)"), ("EEV", "mean-value"), ("WS", "perfect information")):
            L.append(f"| {label} | {_fmt(r.risk_parts[k]['mean'])} | {_fmt(r.risk_parts[k]['cvar'])} |")
        L.append("")
    if study.frontier:
        a = study.frontier[0]["alpha"]
        hold = "holdout_mean" in study.frontier[0]
        L += ["## Mean-risk trade-off", "",
              f"Each row solves the stochastic program with a different weight on CVaR at {a:g}.", "",
              f"| CVaR weight | Expected {unit} | CVaR |" + (" Out-of-sample mean | Out-of-sample CVaR |" if hold else ""),
              "| ---: | ---: | ---: |" + (" ---: | ---: |" if hold else "")]
        for f in study.frontier:
            L.append(f"| {f['weight']:g} | {_fmt(f['mean'])} | {_fmt(f['cvar'])} |"
                     + (f" {_fmt(f['holdout_mean'])} | {_fmt(f['holdout_cvar'])} |" if hold else ""))
        L.append("")

    L += ["## First-stage decision", "", "| Variable | Mean-value model | Stochastic model |",
          "| --- | ---: | ---: |"]
    names = list(r.x_rp)
    changed = [nm for nm in names if abs(r.x_ev.get(nm, 0.0) - r.x_rp[nm]) > 1e-6]
    shown = (changed + [nm for nm in names if nm not in changed])[:25]
    for nm in shown:
        L.append(f"| {nm} | {r.x_ev.get(nm, float('nan')):,.4g} | {r.x_rp[nm]:,.4g} |")
    if len(names) > len(shown):
        L.append(f"| ... {len(names) - len(shown)} more | | |")
    L += ["", f"{len(changed)} of {len(names)} first-stage variables differ.", ""]

    if study.oos:
        o = study.oos
        L += ["## Out-of-sample test", "",
              f"Both decisions were applied to {o['n']} {o.get('source', 'held-out observations')} "
              "that were not used to build the scenarios.", "",
              f"- Mean {unit}: mean-value decision {_fmt(o['mean_ev'])}, stochastic decision {_fmt(o['mean_rp'])}.",
              f"- Mean gain of the stochastic decision: {_fmt(o['mean_gain'])} per observation, "
              f"95% bootstrap interval [{_fmt(o['gain_ci95'][0])}, {_fmt(o['gain_ci95'][1])}].",
              f"- The stochastic decision was better in {100 * o['share_rp_better']:.0f}% of the "
              f"observations and worse in {100 * o['share_ev_better']:.0f}%.",
              f"- Feasible observations: mean-value {o['n_feasible_ev']}/{o['n']}, "
              f"stochastic {o['n_feasible_rp']}/{o['n']}.", ""]
        if "risk_gain" in o:
            L[-1:-1] = [f"- Risk-adjusted objective: mean-value decision {_fmt(o['risk_ev'])}, stochastic "
                        f"decision {_fmt(o['risk_rp'])}; gain {_fmt(o['risk_gain'])}, 95% bootstrap interval "
                        f"[{_fmt(o['risk_gain_ci95'][0])}, {_fmt(o['risk_gain_ci95'][1])}]."]
    if study.gap:
        g = study.gap
        L += ["## Solution quality", "",
              f"Optimality gap of the stochastic solution (Mak-Morton-Wood estimate, {g['batches']} "
              f"batches of {g['n']} scenarios): mean {_fmt(g['mean_gap'])}, 95% upper bound "
              f"{_fmt(g['upper95'])} ({g['pct_of_RP']:.2f}% of RP).", ""]

    L += ["## Checks", ""]
    for c in study.checks:
        L.append(f"- **{c.status.upper()}** `{c.name}`: {c.detail}")
    L += ["", "The checks verify that the lift is mathematically consistent with the deterministic "
          "model. They cannot verify that the stage assignment matches how decisions are made in "
          "practice; that is confirmed by reviewing `uncertainty.yaml`.", ""]
    return "\n".join(L)


def latex_table(study) -> str:
    r = study.results
    rows = [("EV", "Mean-value model (own estimate)", r.ev), ("WS", "Wait-and-see", r.ws),
            ("RP", "Stochastic solution", r.rp), ("EEV", "Mean-value solution, evaluated", r.eev),
            ("VSS", "Value of the stochastic solution", r.vss),
            ("EVPI", "Expected value of perfect information", r.evpi)]
    infty, eol = "$\\infty$", " \\\\"
    body = "\n".join(f"{a} & {b} & {_fmt(v).replace('inf', infty)}{eol}" for a, b, v in rows)
    head = ("Expected " + ("cost" if r.sense == "min" else "value")) if r.risk_parts is None else "Objective"
    return ("\\begin{tabular}{llr}\n\\toprule\n & Quantity & " + head + " \\\\\n\\midrule\n" + body
            + "\n\\bottomrule\n\\end{tabular}\n")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else str(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_report(study, outdir, figures: bool = True, style: str = "nature") -> str:
    os.makedirs(outdir, exist_ok=True)
    study.spec.save(os.path.join(outdir, "uncertainty.yaml"))
    payload = {"results": study.results.to_dict(),
               "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in study.checks],
               "scenarios": {"names": study.scenarios.names, "probabilities": study.scenarios.probs,
                             "values": study.scenarios.values},
               "out_of_sample": study.oos, "stability": study.stability_table, "saa_gap": study.gap,
               "risk_frontier": study.frontier}
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
    with open(os.path.join(outdir, "table_values.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table(study))
    made = []
    if figures:
        from .plots import make_figures

        made = make_figures(study, outdir, style=style)
    md = summary_markdown(study)
    if made:
        md += ("\n## Figures\n\n![Overview](fig_overview.png)\n\nLegends for every figure are in "
               "`captions.md`, and the numbers behind each figure in `source_data/`. Files: "
               + ", ".join(f"`{os.path.splitext(os.path.basename(p))[0]}`" for p in made)
               + " (PDF, SVG and PNG).\n")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return os.path.join(outdir, "summary.md")
