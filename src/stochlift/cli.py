"""Command line: ``stochlift init`` writes a spec template, ``stochlift run`` writes the report.

The model is a Python file (or importable module) that defines a builder
function and its data dictionary::

    stochlift init model.py                     # inspect the model, write uncertainty.yaml
    stochlift run model.py --spec uncertainty.yaml --out report/
    stochlift run model.py:make_model --data data.json --history demand.csv

``model.py`` alone means the function ``build_model`` and the dictionary ``DATA``
in that file; ``model.py:make_model`` picks another function, and ``--data``
names another variable in the file or a ``.json`` / ``.yaml`` file.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import sys

import yaml

from . import __version__


def load_model(target: str, data: str = None):
    """``(build_model, data)`` from ``path.py[:function]`` or ``package.module[:function]``."""
    func = "build_model"
    path = target
    if ":" in target and not (len(target) > 1 and target[1] == ":" and target.count(":") == 1):
        path, func = target.rsplit(":", 1)        # leave Windows drive letters alone
    if path.endswith(".py") or os.path.exists(path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise SystemExit(f"error: model file not found: {path}")
        folder = os.path.dirname(path)
        if folder not in sys.path:
            sys.path.insert(0, folder)            # let the model import its neighbours
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(name, module)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(path)
    if not callable(getattr(module, func, None)):
        raise SystemExit(f"error: {path} has no function '{func}'. Name it with {path}:<function>.")
    build = getattr(module, func)

    if data and os.path.splitext(data)[1].lower() in (".json", ".yaml", ".yml"):
        with open(data, encoding="utf-8") as f:
            d = json.load(f) if data.lower().endswith(".json") else yaml.safe_load(f)
    else:
        var = data or "DATA"
        if not hasattr(module, var):
            raise SystemExit(f"error: {path} has no variable '{var}'. Pass the data with "
                             "--data <variable> or --data <file.json|file.yaml>.")
        d = getattr(module, var)
        if callable(d):
            d = d()
    if not isinstance(d, dict):
        raise SystemExit(f"error: the data must be a dictionary, got {type(d).__name__}")
    return build, d


# ------------------------------------------------------------------------------ init
def spec_template(build, data, history=None) -> str:
    from . import datautil as du
    from .adapters import to_linear_model
    from .llm import _probe_all, variable_groups

    model = to_linear_model(build(copy.deepcopy(data)))
    groups = variable_groups(model)
    probe = _probe_all(build, data)
    L = ["# StochLift uncertainty specification. Review every line before trusting the results.",
         f"# Model: {model.n} variables ({int(model.integer.sum())} integer), {model.m} constraints, "
         f"sense = {model.sense}", "#",
         "# Variable groups (first-stage = decided BEFORE the uncertainty is known):"]
    for pat, g in groups.items():
        kind = "integer" if g["integer"] == g["count"] else ("mixed" if g["integer"] else "continuous")
        L.append(f"#   {pat:<24} {g['count']:>5} {kind}, e.g. {', '.join(g['examples'])}")
    L += ["#", "# Numeric data keys and where they enter the model:"]
    candidates = []
    for key, d in probe.items():
        if d.get("error"):
            where = "cannot be changed (" + d["error"][:60] + ")"
        elif d.get("structure_changed"):
            where = "changes the model's size"
        else:
            parts = [f"{d[p]} {p}" for p in ("objective", "matrix", "rhs", "bounds") if d[p]]
            where = ", ".join(parts) or "does not enter the model"
            if parts:
                candidates.append(key)
        n = len(du.numeric_leaves(data[_original_key(data, key)]))
        L.append(f"#   {key:<24} {n:>5} number(s): {where}")
    first = next(iter(groups), "x*")
    unc = candidates[0] if candidates else "some_key"
    L += ["", "first_stage:            # patterns on variable names; * matches anything",
          f"- \"{first}\"            # TODO: keep only the here-and-now decisions", "",
          "uncertain:              # data keys (or dotted entries such as demand.C1)",
          f"- {unc}                 # TODO", ""]
    if history:
        L += ["columns: {}             # only if a history column differs from the entry name", "",
              "scenarios:", "  method: kmeans         # empirical | kmeans | sample", "  n: 30",
              "  seed: 0", "", "holdout: 0.3            # last share of the history kept for testing", ""]
    else:
        L += ["scenarios:",
              "  method: distribution   # no history: sample from distributions",
              "  n: 100                 # scenarios in the stochastic program",
              "  n_test: 500            # independent samples for the out-of-sample test",
              "  seed: 0",
              "  correlation: 0.0       # common correlation between all uncertain entries",
              "  distributions:         # parameters are relative to the value in the data",
              f"    {unc}: {{dist: normal, cv: 0.2, min: 0}}   # TODO",
              "    # other marginals: lognormal {cv}, uniform {spread} or {low, high},",
              "    # triangular {spread} or {low, mode, high}, discrete {factors|values, probs}", ""]
    L += ["notes: \"\"", ""]
    return "\n".join(L)


def _original_key(data, key: str):
    for k in data:
        if str(k) == key:
            return k
    return key


def cmd_init(args) -> int:
    build, data = load_model(args.model, args.data)
    if args.llm:
        from .llm import propose_spec

        spec = propose_spec(build, data, history=args.history, llm=_llm(args.llm), verbose=True)
        text = ("# Proposed by a language model. Review every line before trusting the results.\n"
                + spec.to_yaml())
    else:
        text = spec_template(build, data, args.history)
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"error: {args.output} exists; use --force to overwrite")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.output}. Edit the TODO lines, then: stochlift run {args.model} --spec {args.output}")
    return 0


def _llm(name: str):
    from .llm import anthropic_llm, openai_llm

    vendor, _, model = name.partition(":")
    if vendor == "anthropic" and model:
        return anthropic_llm(model=model)
    if vendor == "openai" and model:
        return openai_llm(model=model)
    raise SystemExit("error: --llm must look like anthropic:<model id> or openai:<model id>")


# ------------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    from .checks import all_passed
    from .study import lift

    build, data = load_model(args.model, args.data)
    kwargs = {"mip_gap": args.mip_gap}
    if args.time_limit:
        kwargs["time_limit"] = args.time_limit
    study = lift(build, data, spec=args.spec, history=args.history, **kwargs)
    if not args.quiet:
        study.review()
        print()
    r = study.solve()
    checks = study.check()
    for c in checks:
        print(c)
    if study.test is not None and not args.no_out_of_sample:
        study.out_of_sample()
    if args.stability:
        study.stability(sizes=[int(s) for s in args.stability.split(",")], reps=args.reps)
    if args.gap:
        study.saa_gap(n=args.gap, batches=args.batches)
    path = study.report(args.out, figures=not args.no_figures)
    print(f"\nRP {r.rp:,.6g}  EEV {r.eev:,.6g}  WS {r.ws:,.6g}  VSS {r.vss:,.6g} "
          f"({r.vss_pct:.2f}%)  EVPI {r.evpi:,.6g} ({r.evpi_pct:.2f}%)")
    if study.oos:
        o = study.oos
        print(f"out of sample ({o['n_compared']} {o['source']}): mean gain {o['mean_gain']:,.6g}, "
              f"95% interval [{o['gain_ci95'][0]:,.6g}, {o['gain_ci95'][1]:,.6g}]")
    print(f"report: {path}")
    if not all_passed(checks):
        print("some checks FAILED; see the report before using these results", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="stochlift", description=(
        "Lift a deterministic optimization model to a two-stage stochastic program and measure "
        "whether it is worth it."))
    p.add_argument("--version", action="version", version=f"stochlift {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(q):
        q.add_argument("model", help="model.py, model.py:function or package.module:function "
                                     "(default function: build_model)")
        q.add_argument("--data", help="variable in the model file (default DATA) or a .json/.yaml file")
        q.add_argument("--history", help="CSV of past observations of the uncertain data")

    q = sub.add_parser("init", help="inspect the model and write a spec template to edit")
    common(q)
    q.add_argument("-o", "--output", default="uncertainty.yaml")
    q.add_argument("--force", action="store_true", help="overwrite an existing file")
    q.add_argument("--llm", help="let a language model propose the spec, e.g. anthropic:<model id>")
    q.set_defaults(func=cmd_init)

    q = sub.add_parser("run", help="solve, check and write the report")
    common(q)
    q.add_argument("--spec", default="uncertainty.yaml", help="the reviewed spec (default uncertainty.yaml)")
    q.add_argument("--out", default="report", help="report folder (default report)")
    q.add_argument("--stability", metavar="SIZES", help="comma-separated scenario counts, e.g. 5,10,20,40")
    q.add_argument("--reps", type=int, default=10, help="samples per size for --stability")
    q.add_argument("--gap", type=int, metavar="N", help="estimate the optimality gap with batches of N")
    q.add_argument("--batches", type=int, default=20)
    q.add_argument("--mip-gap", type=float, default=1e-6)
    q.add_argument("--time-limit", type=float, help="seconds per solve")
    q.add_argument("--no-figures", action="store_true")
    q.add_argument("--no-out-of-sample", action="store_true")
    q.add_argument("-q", "--quiet", action="store_true", help="do not print the review")
    q.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
