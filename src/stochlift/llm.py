"""LLM front end: read the user's model code and propose an uncertainty spec.

The language model only *proposes*. Its answer is parsed into a :class:`Spec`,
checked against the real model (do the data keys exist, do the patterns match
variables, does the uncertain data reach the model), and any problem is sent
back for another attempt. A person still reviews the resulting YAML.

``llm`` is any callable ``prompt: str -> str``; :func:`anthropic_llm` and
:func:`openai_llm` build one from the vendor SDKs.
"""
from __future__ import annotations

import copy
import inspect
import re
from collections import OrderedDict
from typing import Callable, Optional

import yaml

from . import datautil as du
from . import scenarios as sc
from .adapters import to_linear_model
from .spec import Spec

PROMPT = """\
You are helping turn a deterministic optimization model into a two-stage stochastic program.

The user has a working deterministic model: a Python function that builds the model from a
data dictionary. Your job is to propose (1) which data entries are uncertain, and (2) which
decision variables are first-stage, meaning they must be fixed BEFORE the uncertain data is
known. All other variables are second-stage (recourse): they are chosen after the uncertainty
is revealed and may differ in every scenario.

Typical first-stage decisions: investment, capacity, opening or building, planting, staffing
levels, orders placed in advance. Typical second-stage decisions: shipping, production
adjustments, buying or selling on the spot market, overtime, shortage, outsourcing, inventory
that results from realized demand.

## Model source code
```python
{source}
```

## Data dictionary
{data_summary}

## Variables in the built model
{variables}

## Where each numeric data key enters the model
(found by changing the key by 10% and rebuilding the model)
{probe}

## Historical observations
{history}

## Answer format
Reply with one YAML block and nothing else:

```yaml
first_stage:        # patterns on variable names; * matches any characters
- "pattern*"
uncertain:          # top-level data keys (or dotted entries such as yield.wheat)
- key
columns: {{}}         # only if a history column name differs from the data entry: {{"demand.C1": "col_name"}}
scenarios:
  method: kmeans    # empirical (each observation is a scenario), kmeans, or sample
  n: 30             # omit for empirical
  seed: 0
holdout: 0.3        # share of the history kept aside for an out-of-sample test; 0 if there is no history
notes: one or two sentences: why these are first-stage, and anything you are unsure about
```

Rules:
- Use variable-name patterns exactly as the names appear above.
- Only list data keys that exist in the data dictionary and enter the model.
- If there is a history, the uncertain keys must be the ones the history columns describe.
- With fewer than about 60 observations use method empirical; otherwise kmeans with n about 30.
- Do not invent data, scenarios or probabilities.
{feedback}"""


def _source(build_model) -> str:
    try:
        module = inspect.getmodule(build_model)
        text = inspect.getsource(module) if module is not None else ""
        if 0 < len(text.splitlines()) <= 250:
            return text.rstrip()
        return inspect.getsource(build_model).rstrip()
    except (OSError, TypeError):
        return "# source code not available"


def variable_groups(model) -> "OrderedDict[str, dict]":
    """Group variables by name pattern, e.g. ``open[*]`` or ``acres_*``."""
    groups: OrderedDict = OrderedDict()
    stems = {}
    for nm in model.names:
        if "[" in nm:
            stems[nm] = nm.split("[", 1)[0] + "[*]"
        elif "(" in nm:
            stems[nm] = nm.split("(", 1)[0] + "(*)"
    plain = [nm for nm in model.names if nm not in stems]
    prefixes = {}
    for nm in plain:
        if "_" in nm:
            prefixes.setdefault(nm.split("_", 1)[0], []).append(nm)
    for nm in plain:
        head = nm.split("_", 1)[0]
        stems[nm] = head + "_*" if "_" in nm and len(prefixes[head]) > 1 else nm
    for j, nm in enumerate(model.names):
        g = groups.setdefault(stems[nm], {"count": 0, "integer": 0, "examples": []})
        g["count"] += 1
        g["integer"] += int(model.integer[j])
        if len(g["examples"]) < 3:
            g["examples"].append(nm)
    return groups


def _probe_all(build_model, data) -> dict:
    from .study import _diff

    base = to_linear_model(build_model(copy.deepcopy(data)))
    out = {}
    for key, value in data.items():
        leaves = du.numeric_leaves(value, (key,))
        if not leaves:
            continue
        paths = [p for p, _ in leaves]
        new = [v * 1.1 if v != 0 else 0.1 for _, v in leaves]
        try:
            pert = to_linear_model(build_model(du.apply(data, paths, new)))
            out[str(key)] = _diff(base, pert, lambda nm: False)
        except Exception as e:
            out[str(key)] = {"error": f"{type(e).__name__}: {e}"}
    return out


def build_prompt(build_model, data: dict, history=None, feedback: Optional[str] = None) -> str:
    model = to_linear_model(build_model(copy.deepcopy(data)))
    var_lines = []
    for pat, g in variable_groups(model).items():
        kind = "integer/binary" if g["integer"] == g["count"] else ("mixed" if g["integer"] else "continuous")
        var_lines.append(f"- {pat}: {g['count']} {kind} variable(s), e.g. {', '.join(g['examples'])}")
    probe_lines = []
    for key, d in _probe_all(build_model, data).items():
        if d.get("error"):
            probe_lines.append(f"- {key}: the model cannot be rebuilt when this changes ({d['error']})")
        elif d.get("structure_changed"):
            probe_lines.append(f"- {key}: changes the set of variables or constraints (a size or index, not a coefficient)")
        else:
            parts = [f"{d[p]} {label}" for p, label in (("objective", "objective coefficients"),
                     ("matrix", "constraint coefficients"), ("rhs", "right-hand sides"),
                     ("bounds", "variable bounds")) if d[p]]
            where = ", ".join(parts) if parts else "does not enter the model"
            rows = f" (rows such as {', '.join(d['example_rows'])})" if d["example_rows"] else ""
            probe_lines.append(f"- {key}: {where}{rows}")
    if history is None:
        hist = "None. Scenarios will be supplied by the user; set holdout to 0 and method to empirical."
    else:
        import pandas as pd

        df = pd.read_csv(history) if not hasattr(history, "columns") else history
        hist = (f"{len(df)} observations (rows). Columns: {', '.join(map(str, df.columns))}\n"
                f"First rows:\n{df.head(3).to_string(index=False)}")
    fb = f"\n## Problems with your previous answer\n{feedback}\nFix them and answer again.\n" if feedback else ""
    return PROMPT.format(source=_source(build_model), data_summary=du.summarize(data),
                         variables="\n".join(var_lines), probe="\n".join(probe_lines),
                         history=hist, feedback=fb)


def parse_spec(text: str) -> Spec:
    """Extract the YAML block from a model reply."""
    fenced = re.findall(r"^```[ \t]*(\w*)[ \t]*\n(.*?)^```[ \t]*$", text, flags=re.S | re.M)
    blocks = [body for lang, body in fenced if lang.lower() in ("yaml", "yml")]
    blocks += [body for lang, body in fenced if lang.lower() not in ("yaml", "yml")]
    first_error = None
    for block in blocks + [text]:
        try:
            d = yaml.safe_load(block)
        except yaml.YAMLError as e:
            first_error = first_error or f"invalid YAML: {str(e).splitlines()[0]}"
            continue
        if not isinstance(d, dict) or "first_stage" not in d:
            continue
        for key, default in (("columns", {}), ("notes", ""), ("holdout", 0.0), ("uncertain", [])):
            if d.get(key) is None:
                d[key] = default
        try:
            return Spec.from_dict(d)
        except (ValueError, TypeError) as e:
            first_error = first_error or str(e)
    raise ValueError(f"no usable YAML spec in the reply: {first_error}" if first_error
                     else "no YAML block with a 'first_stage' field in the reply")


def validate(build_model, data: dict, spec: Spec, history=None) -> list:
    """Problems that make a spec unusable for this model. Empty list = usable."""
    problems = []
    model = to_linear_model(build_model(copy.deepcopy(data)))
    for pat in spec.first_stage:
        one = Spec(first_stage=[pat])
        if not any(one.is_first_stage(nm) for nm in model.names):
            problems.append(f"first_stage pattern '{pat}' matches no variable; variables are "
                            f"{', '.join(variable_groups(model))}")
    n_first = sum(spec.is_first_stage(nm) for nm in model.names)
    if n_first == model.n:
        problems.append("every variable is first-stage; at least one variable must be recourse")
    paths = []
    if history is not None and not spec.uncertain:
        problems.append("a history is given but 'uncertain' is empty")
    try:
        paths = du.expand_keys(data, spec.uncertain) if spec.uncertain else []
    except KeyError as e:
        problems.append(str(e).strip('"'))
    if paths:
        from .study import probe_paths

        for key, d in probe_paths(build_model, data, paths, lambda nm: False).items():
            if d.get("error"):
                problems.append(f"uncertain key '{key}': {d['error']}")
            elif d.get("structure_changed"):
                problems.append(f"uncertain key '{key}' changes the model's size, not a coefficient")
            elif d["dead"]:
                problems.append(f"uncertain entries {d['dead'][:6]} do not enter the model")
        if history is not None:
            import pandas as pd

            df = pd.read_csv(history) if not hasattr(history, "columns") else history
            try:
                sc.match_columns(paths, df.columns, spec.columns)
            except KeyError as e:
                problems.append(str(e).strip('"'))
    method = spec.scenarios.get("method")
    if method not in ("empirical", "kmeans", "sample", "explicit"):
        problems.append(f"unknown scenarios.method '{method}'")
    if method in ("kmeans", "sample") and not spec.scenarios.get("n"):
        problems.append(f"scenarios.method '{method}' needs n")
    return problems


def propose_spec(build_model, data: dict, history=None, llm: Callable[[str], str] = None,
                 max_rounds: int = 3, verbose: bool = False) -> Spec:
    """Ask ``llm`` for a spec, validate it against the model, and retry with feedback."""
    if llm is None:
        raise ValueError("propose_spec needs llm, a callable prompt -> reply")
    feedback = None
    problems = []
    for round_ in range(max_rounds):
        reply = llm(build_prompt(build_model, data, history, feedback))
        try:
            spec = parse_spec(reply)
            problems = validate(build_model, data, spec, history)
        except ValueError as e:
            problems = [str(e)]
        if verbose:
            print(f"[stochlift] round {round_ + 1}: {'ok' if not problems else problems}")
        if not problems:
            return spec
        feedback = "\n".join(f"- {p}" for p in problems)
    raise ValueError("the language model did not produce a usable spec after "
                     f"{max_rounds} attempts. Last problems:\n{feedback}")


def anthropic_llm(model: str, max_tokens: int = 1500, **kwargs) -> Callable[[str], str]:
    """``prompt -> reply`` using the ``anthropic`` SDK (reads ANTHROPIC_API_KEY)."""
    import anthropic

    client = anthropic.Anthropic(**kwargs)

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")

    return call


def openai_llm(model: str, **kwargs) -> Callable[[str], str]:
    """``prompt -> reply`` using the ``openai`` SDK (reads OPENAI_API_KEY)."""
    import openai

    client = openai.OpenAI(**kwargs)

    def call(prompt: str) -> str:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content

    return call
