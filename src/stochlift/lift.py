"""Build the extensive form (deterministic equivalent) of a two-stage program.

The lift needs no knowledge of *where* the uncertain data enters the model.
Each scenario is simply the user's own model built with that scenario's data;
the first-stage columns are shared across scenarios and everything else is
copied per scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import LinearModel


@dataclass
class ExtensiveForm:
    model: LinearModel
    first_names: list          # first-stage variable names (columns 0..k-1)
    blocks: list               # per scenario: (start column, second-stage names)
    notes: list                # human-readable remarks collected while lifting


def first_stage_names(models, is_first) -> list:
    names, seen = [], set()
    for m in models:
        for nm in m.names:
            if nm not in seen and is_first(nm):
                seen.add(nm)
                names.append(nm)
    return names


def extensive_form(models, probs, is_first, risk=None) -> ExtensiveForm:
    """The deterministic equivalent. With an active ``risk`` (see :mod:`stochlift.risk`) the
    objective is ``(1 - w) E[f] + w CVaR_alpha(f)``, linearized with one free column ``eta``
    and one non-negative column per scenario after all model columns."""
    probs = np.asarray(probs, dtype=float)
    if len(models) != len(probs):
        raise ValueError("one probability per scenario model is required")
    sign = models[0].sign
    if any(m.sign != sign for m in models):
        raise ValueError("the objective sense changes between scenarios")

    F = first_stage_names(models, is_first)
    fpos = {nm: j for j, nm in enumerate(F)}
    k = len(F)
    notes = []

    c = [np.zeros(k)]
    lb = [np.full(k, -np.inf)]
    ub = [np.full(k, np.inf)]
    integer = [np.zeros(k, dtype=bool)]
    names = list(F)
    bound_changes = set()
    cost_changes = set()
    first_cost_seen = {}
    first_bounds_seen = {}
    rows, cols, vals, row_lb, row_ub, row_names = [], [], [], [], [], []
    seen_first_rows = {}
    blocks = []
    offset = 0.0
    ncol, nrow = k, 0
    scenario_cost = []          # per scenario: (columns, coefficients, constant) of its total cost

    for s, (m, p) in enumerate(zip(models, probs)):
        is_f = np.array([nm in fpos for nm in m.names], dtype=bool)
        colmap = np.empty(m.n, dtype=np.int64)
        second = []
        for j, nm in enumerate(m.names):
            if is_f[j]:
                g = fpos[nm]
                colmap[j] = g
                c[0][g] += p * m.c[j]
                if nm in first_cost_seen and first_cost_seen[nm] != m.c[j]:
                    cost_changes.add(nm)
                first_cost_seen.setdefault(nm, m.c[j])
                if nm in first_bounds_seen and first_bounds_seen[nm] != (m.lb[j], m.ub[j]):
                    bound_changes.add(nm)
                first_bounds_seen.setdefault(nm, (m.lb[j], m.ub[j]))
                lb[0][g] = max(lb[0][g], m.lb[j])
                ub[0][g] = min(ub[0][g], m.ub[j])
                integer[0][g] |= m.integer[j]
            else:
                colmap[j] = ncol + len(second)
                second.append(nm)
        sec = ~is_f
        c.append(p * m.c[sec])
        lb.append(m.lb[sec])
        ub.append(m.ub[sec])
        integer.append(m.integer[sec])
        names.extend(f"{nm}@s{s}" for nm in second)
        blocks.append((ncol, second))
        ncol += len(second)
        offset += p * m.offset
        nz = np.flatnonzero(m.c)
        scenario_cost.append((colmap[nz], m.c[nz], float(m.offset)))

        A = m.A.tocsr()
        for i in range(m.m):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            idx, val = A.indices[lo:hi], A.data[lo:hi]
            if len(idx) and is_f[idx].all():
                # a row on first-stage variables only: keep one copy of identical rows
                key = (tuple(sorted(zip(colmap[idx].tolist(), val.tolist()))),
                       m.row_lb[i], m.row_ub[i])
                if key in seen_first_rows:
                    continue
                seen_first_rows[key] = m.row_names[i]
            rows.extend([nrow] * len(idx))
            cols.extend(colmap[idx].tolist())
            vals.extend(val.tolist())
            row_lb.append(m.row_lb[i])
            row_ub.append(m.row_ub[i])
            row_names.append(f"{m.row_names[i]}@s{s}")
            nrow += 1

    if cost_changes:
        notes.append("objective coefficients of first-stage variables differ between scenarios "
                     f"({', '.join(sorted(cost_changes)[:5])}); their expectation is used")
    if bound_changes:
        notes.append("bounds of first-stage variables differ between scenarios "
                     f"({', '.join(sorted(bound_changes)[:5])}); the intersection is used")

    if risk is not None and risk.active:
        # (1 - w) E[f] + w (eta + sum_s p_s u_s / (1 - alpha)),  u_s >= f_s - eta,  u_s >= 0
        w, a = float(risk.weight), float(risk.alpha)
        c = [(1 - w) * part for part in c]
        offset *= (1 - w)
        S = len(models)
        eta = ncol
        c.append(np.concatenate([[w], w * probs / (1 - a)]))
        lb.append(np.concatenate([[-np.inf], np.zeros(S)]))
        ub.append(np.full(S + 1, np.inf))
        integer.append(np.zeros(S + 1, dtype=bool))
        names.extend(["__cvar_eta"] + [f"__cvar_excess@s{s}" for s in range(S)])
        for s, (cc, cv, const) in enumerate(scenario_cost):
            # f_s(x) - eta - u_s <= -const
            rows.extend([nrow] * (len(cc) + 2))
            cols.extend(cc.tolist() + [eta, eta + 1 + s])
            vals.extend(cv.tolist() + [-1.0, -1.0])
            row_lb.append(-np.inf)
            row_ub.append(-const)
            row_names.append(f"__cvar_excess@s{s}")
            nrow += 1
        ncol += S + 1

    A = sp.coo_matrix((vals, (rows, cols)), shape=(nrow, ncol)).tocsr()
    ef = LinearModel(names=names, c=np.concatenate(c), offset=offset, lb=np.concatenate(lb),
                     ub=np.concatenate(ub), integer=np.concatenate(integer), A=A,
                     row_lb=np.array(row_lb), row_ub=np.array(row_ub), row_names=row_names,
                     sign=sign)
    return ExtensiveForm(ef, F, blocks, notes)


def first_stage_row_report(models, is_first) -> list:
    """Rows that involve first-stage variables only but change with the scenario.

    Such a row means uncertain data constrains a here-and-now decision; the
    extensive form then enforces it for every scenario at once.
    """
    seen = {}
    changed = set()
    for m in models:
        is_f = np.array([is_first(nm) for nm in m.names], dtype=bool)
        A = m.A.tocsr()
        for i in range(m.m):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            idx = A.indices[lo:hi]
            if len(idx) and is_f[idx].all():
                key = (tuple(sorted((m.names[j], v) for j, v in zip(idx, A.data[lo:hi]))),
                       m.row_lb[i], m.row_ub[i])
                name = m.row_names[i]
                if name in seen and seen[name] != key:
                    changed.add(name)
                seen.setdefault(name, key)
    return sorted(changed)
