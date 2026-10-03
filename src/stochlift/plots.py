"""Publication figures, written at their final print size.

Every figure has three parts kept together here: a drawing function that fills a
(sub)figure, a legend text (written to ``captions.md``), and its source data
(written to ``source_data/<figure>.csv``), as journals ask for all three. The same
drawing functions build the standalone figures and the multi-panel overview.

Colour follows the entity in every figure: the mean-value decision is blue with
circles, the stochastic decision orange with squares, perfect information green
with diamonds. See :mod:`stochlift.figstyle` for sizes and fonts.
"""
from __future__ import annotations

import os

import numpy as np

from .figstyle import (AXIS, C_EV, C_RP, C_WS, CONTEXT, GRID, INK, INK2, L_EV, L_RP, L_WS, M_EV, M_RP,
                       M_WS, MM, MUTED, WASH_EV, WASH_RP, get_style)


# ------------------------------------------------------------------------- helpers
def _num(v: float, scale: float = None) -> str:
    """A number for a label; ``scale`` (e.g. |RP|) fixes the precision for a figure."""
    if not np.isfinite(v):
        return "infeasible" if v > 0 else "−inf"
    ref = abs(v) if scale is None else abs(scale)
    if ref >= 1e7:
        return _tick(v)
    if ref >= 1000:
        text = f"{v:,.0f}"
    else:
        text = f"{v:,.1f}" if ref >= 100 else f"{v:,.3g}"
    return text.replace("-", "−")


def _tick(v: float) -> str:
    """Compact tick label: 950, 12,500, 3.4M, 1.25B. Uses a true minus sign."""
    a = abs(v)
    for scale, suffix in ((1e9, "B"), (1e6, "M")):
        if a >= scale:
            return f"{v / scale:,.3g}{suffix}".replace("-", "−")
    text = f"{v:,.0f}" if a >= 1000 or float(v).is_integer() else f"{v:,.2g}"
    return text.replace("-", "−")


def _ticks(ax, axis="x", n=5):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))
    target.set_major_formatter(FuncFormatter(lambda v, _: _tick(v)))


SUPERSCRIPT = {3: "³", 6: "⁶", 9: "⁹"}


def _scaled(ax, axis, label, n=5, note=""):
    """Ticks in units of 10^3, 10^6 or 10^9 when the numbers are large, with the factor in
    the axis label ("Expected cost (×10³)") instead of long tick labels."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3))
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    big = max(abs(lo), abs(hi))
    exp = 0
    if big >= 1e4:
        exp = min(9, 3 * int(np.floor(np.log10(big) / 3)))
    ticks = target.get_major_locator().tick_values(lo, hi)
    step = (ticks[1] - ticks[0]) / 10 ** exp if len(ticks) > 1 else 1.0
    dec = next((d for d in range(6) if abs(round(step, d) - step) <= 1e-9 * max(1.0, abs(step))), 6)
    target.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 10 ** exp:,.{dec}f}".replace("-", "−")))
    inside = [f"×10{SUPERSCRIPT[exp]}"] if exp else []
    if note:
        inside.append(note)
    if inside:
        label = f"{label} ({'; '.join(inside)})"
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(label)


def _unit(study) -> str:
    return "cost" if study.results.sense == "min" else "value"


def _objective_label(study) -> str:
    return f"Expected {_unit(study)}" if not study.risk.active else "Risk-adjusted objective"


def _better(study) -> str:
    return ("lower" if study.results.sense == "min" else "higher") + " is better"


def _short(name: str, n: int = 22) -> str:
    return name if len(name) <= n else name[: n - 8] + "…" + name[-7:]


def _row_guides(ax, ys, st):
    for y in ys:
        ax.axhline(y, color=GRID, lw=st.thin, zorder=0)


def _hide_y_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _legend(fig, st, ncol=2, handles=None):
    kw = {"loc": "outside lower center", "ncols": ncol, "fontsize": st.small}
    if handles is not None:
        kw["handles"] = handles
    return fig.legend(**kw)


def _entity_handles(st, *which):
    from matplotlib.lines import Line2D

    spec = {"ev": (C_EV, M_EV, L_EV), "rp": (C_RP, M_RP, L_RP), "ws": (C_WS, M_WS, L_WS)}
    return [Line2D([], [], color=spec[k][0], marker=spec[k][1], ls="none", ms=st.marker * 1.1,
                   mec="white", mew=0.6 * st.thin / 0.5, label=spec[k][2]) for k in which]


def _mew(st):
    return 0.6 * st.thin / 0.5


# ---------------------------------------------------------------------- value
def draw_value(study, fig, st):
    """Expected outcome of each decision on one scale, with VSS and EVPI as brackets."""
    r = study.results
    ax = fig.add_subplot()
    rows = [("Perfect information (WS)", r.ws, C_WS, M_WS),
            ("Stochastic decision (RP)", r.rp, C_RP, M_RP),
            ("Mean-value decision (EEV)", r.eev, C_EV, M_EV),
            ("Deterministic estimate (EV)", r.ev, MUTED, "o")]
    ys = [3, 2, 1, 0]
    core = [v for _, v, _, _ in rows[:3] if np.isfinite(v)]
    clo, chi = min(core), max(core)
    cspan = (chi - clo) or max(1.0, abs(chi))
    # the deterministic estimate is context: when it lies far from the others it would squash
    # them, so it is marked at the edge with its value instead
    ev_off = np.isfinite(r.ev) and not (clo - 1.5 * cspan <= r.ev <= chi + 1.5 * cspan)
    finite = core + ([r.ev] if np.isfinite(r.ev) and not ev_off else [])
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or max(1.0, abs(hi))
    right = hi + 0.06 * span
    left = lo - 0.06 * span
    _row_guides(ax, ys, st)
    for y, (label, v, color, marker) in zip(ys, rows):
        if label.startswith("Deterministic"):
            if ev_off:
                inset = 0.015 * (right - left)
                edge, mk = (left + inset, "<") if v < lo else (right - inset, ">")
                ax.plot([edge], [y], marker=mk, ms=st.marker, mfc="white", mec=MUTED, mew=st.line, ls="none",
                        zorder=3)
                ax.text(edge + (2.5 if v < lo else -2.5) * inset, y, "off scale", ha="left" if v < lo else "right",
                        va="center", fontsize=st.small - 0.5, color=INK2)
            else:
                ax.plot([v], [y], marker="o", ms=st.marker * 1.15, mfc="white", mec=MUTED, mew=st.line,
                        ls="none", zorder=3)
        elif np.isfinite(v):
            ax.plot([v], [y], marker=marker, ms=st.marker * 1.3, color=color, mec="white", mew=_mew(st),
                    ls="none", zorder=3)
        else:
            ax.plot([right], [y], marker=">", ms=st.marker, color=color, ls="none", zorder=3, clip_on=False)
        ax.text(1.02, y, _num(v, r.rp), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=st.small, color=INK if not label.startswith("Deterministic") else INK2)
    cap = 0.09

    def bracket(y, a, b, text):
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) < 1e-9 * max(1, abs(a)):
            return
        ax.plot([a, b], [y, y], color=INK2, lw=st.thin * 1.6, solid_capstyle="butt", zorder=2)
        for x in (a, b):
            ax.plot([x, x], [y - cap, y + cap], color=INK2, lw=st.thin * 1.6, zorder=2)
        if abs(b - a) >= 0.3 * (right - left):
            ax.text((a + b) / 2, y + cap * 1.4, text, ha="center", va="bottom", fontsize=st.small, color=INK)
        else:   # a narrow bracket: label beside it, on the side with more room
            room_left = min(a, b) - left > right - max(a, b)
            x = min(a, b) if room_left else max(a, b)
            off = -0.012 * (right - left) if room_left else 0.012 * (right - left)
            ax.text(x + off, y, text, ha="right" if room_left else "left", va="center", fontsize=st.small,
                    color=INK)

    bracket(2.45, r.ws, r.rp, f"EVPI {_num(r.evpi, r.rp)}")
    if np.isfinite(r.eev):
        bracket(1.45, r.rp, r.eev, f"VSS {_num(r.vss, r.rp)}")
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.5, 3.75)
    ax.set_xlim(left, right)
    _hide_y_axis(ax)
    _scaled(ax, "x", _objective_label(study), 4, note=_better(study))
    if st.titles:
        ax.set_title("What the uncertainty costs, and what modelling it recovers")


def caption_value(study) -> str:
    r = study.results
    risk = f" Objective: {r.objective}." if study.risk.active else ""
    infeasible = (f" The mean-value decision has no feasible recourse in {r.eev_infeasible} of "
                  f"{r.n_scenarios} scenarios, so its EEV is infinite (arrow)." if r.eev_infeasible else "")
    return ("Value of modelling the uncertainty. Expected outcome of the stochastic decision (RP, orange "
            "square), of the mean-value decision evaluated over the scenarios (EEV, blue circle) and "
            "with perfect information (WS, green diamond); the open circle is the deterministic model's "
            f"own estimate at the mean data (EV = {_num(r.ev, r.rp)}). Brackets: value of the stochastic solution, VSS = "
            f"{_num(r.vss, r.rp)} ({r.vss_pct:.2f}% of RP), and expected value of perfect information, "
            f"EVPI = {_num(r.evpi, r.rp)} ({r.evpi_pct:.2f}% of RP). {r.n_scenarios} scenarios."
            + risk + infeasible)


def source_value(study):
    import pandas as pd

    r = study.results
    return pd.DataFrame({"quantity": ["EV", "WS", "RP", "EEV", "VSS", "EVPI"],
                         "value": [r.ev, r.ws, r.rp, r.eev, r.vss, r.evpi]})


def height_value(study, st):
    return 40


# ---------------------------------------------------------------- first stage
def _first_stage_rows(study, max_rows):
    r = study.results
    names = list(r.x_rp)
    diff = {nm: abs(r.x_ev.get(nm, 0.0) - r.x_rp[nm]) for nm in names}
    tol = {nm: 1e-6 * max(1.0, abs(r.x_ev.get(nm, 0.0)), abs(r.x_rp[nm])) for nm in names}
    changed = sorted((nm for nm in names if diff[nm] > tol[nm]), key=lambda nm: -diff[nm])
    rest = sorted((nm for nm in names if diff[nm] <= tol[nm]), key=lambda nm: -abs(r.x_rp[nm]))
    return (changed + rest)[:max_rows], changed, names


def draw_first_stage(study, fig, st, max_rows=12):
    """The here-and-now decision of the mean-value and the stochastic model."""
    r = study.results
    shown, changed, names = _first_stage_rows(study, max_rows)
    shown = shown[::-1]
    ax = fig.add_subplot()
    _row_guides(ax, range(len(shown)), st)
    same = False
    vals = [v for nm in shown for v in (r.x_ev.get(nm, np.nan), r.x_rp[nm]) if np.isfinite(v)]
    span = (max(vals) - min(vals)) or 1.0
    for y, nm in enumerate(shown):
        a, b = r.x_ev.get(nm, np.nan), r.x_rp[nm]
        if nm in changed and abs(a - b) < 0.025 * span:
            # the two markers would hide each other: a ring around the square keeps both visible
            ax.plot([a], [y], marker="o", ms=st.marker * 2.0, mfc="none", mec=C_EV, mew=st.line, ls="none",
                    zorder=3)
            ax.plot([b], [y], marker=M_RP, ms=st.marker * 1.0, color=C_RP, mec="white", mew=_mew(st), ls="none",
                    zorder=4)
            continue
        if nm not in changed:
            same = True
            ax.plot([b], [y], marker="D", ms=st.marker, color=MUTED, mec="white", mew=_mew(st), ls="none",
                    zorder=3)
            continue
        ax.plot([a, b], [y, y], color=CONTEXT, lw=st.line * 1.4, zorder=1, solid_capstyle="butt")
        ax.plot([a], [y], marker=M_EV, ms=st.marker * 1.25, color=C_EV, mec="white", mew=_mew(st), ls="none",
                zorder=3)
        ax.plot([b], [y], marker=M_RP, ms=st.marker * 1.15, color=C_RP, mec="white", mew=_mew(st), ls="none",
                zorder=4)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([_short(nm) for nm in shown])
    ax.set_ylim(-0.6, len(shown) - 0.4)
    ax.margins(x=0.08)
    _hide_y_axis(ax)
    _scaled(ax, "x", "First-stage value", 5)
    handles = _entity_handles(st, "ev", "rp")
    if same:
        from matplotlib.lines import Line2D

        handles.append(Line2D([], [], color=MUTED, marker="D", ls="none", ms=st.marker, label="Same in both"))
    _legend(fig, st, ncol=len(handles), handles=handles)
    if st.titles:
        ax.set_title(f"{len(changed)} of {len(names)} first-stage decisions change")


def caption_first_stage(study) -> str:
    shown, changed, names = _first_stage_rows(study, 12)
    more = f" The {len(shown)} variables that change most are shown." if len(names) > len(shown) else ""
    return (f"First-stage decision. Value of each first-stage variable in the mean-value model (blue "
            f"circles) and in the stochastic model (orange squares); {len(changed)} of {len(names)} "
            f"variables differ.{more}")


def source_first_stage(study):
    import pandas as pd

    r = study.results
    return pd.DataFrame({"variable": list(r.x_rp), "mean_value_model": [r.x_ev.get(n, np.nan) for n in r.x_rp],
                         "stochastic_model": list(r.x_rp.values())})


def height_first_stage(study, st):
    return 16 + 5.2 * min(len(study.results.x_rp), 12)


# ------------------------------------------------------------------ scenarios
def _scenario_rows(study, max_rows):
    S = study.scenarios
    mu = S.mean()
    ok = np.abs(mu) > 1e-12
    cv = np.where(ok, np.sqrt(S.probs @ (S.values - mu) ** 2) / np.where(ok, np.abs(mu), 1), 0)
    return [j for j in np.argsort(-cv, kind="stable") if ok[j]][:max_rows]


def _weighted_quantiles(x, w, qs):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return np.interp(qs, cum, x)


def draw_scenarios(study, fig, st, max_rows=8):
    """How the uncertain entries vary across the scenarios (and the history, if any)."""
    from scipy.stats import gaussian_kde

    S = study.scenarios
    mu = S.mean()
    rows = _scenario_rows(study, max_rows)[::-1]
    ax = fig.add_subplot()
    dense = len(S) >= 12
    for y, j in enumerate(rows):
        x = S.values[:, j] / mu[j]
        if dense and np.ptp(x) > 0:
            grid = np.linspace(x.min(), x.max(), 160)
            dens = gaussian_kde(x, weights=S.probs)(grid)
            dens = 0.36 * dens / dens.max()
            ax.fill_between(grid, y, y + dens, color=WASH_EV, lw=0, zorder=1)
            ax.plot(grid, y + dens, color=C_EV, lw=st.thin * 1.2, zorder=2)
            if study.train is not None:
                h = study.train[:, j] / mu[j]
                if np.ptp(h) > 0:
                    hg = np.linspace(h.min(), h.max(), 160)
                    hd = gaussian_kde(h)(hg)
                    ax.plot(hg, y + 0.36 * hd / hd.max(), color=INK2, lw=st.thin, ls=(0, (2, 1.2)), zorder=2)
            q05, q25, q50, q75, q95 = _weighted_quantiles(x, S.probs, [0.05, 0.25, 0.5, 0.75, 0.95])
            ax.plot([q05, q95], [y - 0.08, y - 0.08], color=INK2, lw=st.thin * 1.2, solid_capstyle="butt")
            ax.plot([q25, q75], [y - 0.08, y - 0.08], color=INK2, lw=st.line * 2.4, solid_capstyle="butt")
            ax.plot([q50], [y - 0.08], marker="o", ms=st.marker * 0.75, color="white", mec=INK2,
                    mew=st.thin, ls="none", zorder=4)
        else:
            sizes = (st.marker * 1.6) ** 2 * (0.35 + 0.65 * S.probs / S.probs.max())
            ax.scatter(x, np.full(len(x), y), s=sizes, color=C_EV, edgecolor="white", linewidth=_mew(st),
                       zorder=3)
    ax.axvline(1.0, color=AXIS, lw=st.thin, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_short(S.names[j]) for j in rows])
    ax.set_ylim(-0.45, len(rows) - 0.3)
    _hide_y_axis(ax)
    ax.set_xlabel("Value relative to the mean")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if dense:
        handles = [Patch(facecolor=WASH_EV, edgecolor=C_EV, lw=st.thin, label="Scenarios (density)"),
                   Line2D([], [], color=INK2, lw=st.line * 2.4, label="25–75th percentile")]
        if study.train is not None:
            handles.append(Line2D([], [], color=INK2, lw=st.thin, ls=(0, (2, 1.2)), label="History"))
    else:
        handles = [Line2D([], [], color=C_EV, marker="o", ls="none", ms=st.marker * 1.2,
                          label="Scenario (area ∝ probability)")]
    _legend(fig, st, ncol=len(handles), handles=handles)
    if st.titles:
        ax.set_title(f"{len(S)} scenarios")


def caption_scenarios(study) -> str:
    S = study.scenarios
    rows = _scenario_rows(study, 8)
    which = (f"the {len(rows)} most variable of {len(S.paths)} uncertain entries" if len(S.paths) > len(rows)
             else f"the {len(rows)} uncertain entries")
    method = study.spec.scenarios.get("method", "explicit")
    src = {"distribution": "sampled from the specified distributions", "kmeans": "k-means centroids of the history",
           "empirical": "the observations of the history", "sample": "resampled from the history",
           "explicit": "given explicitly"}.get(method, method)
    if len(S) >= 12:
        how = ("Shaded: probability-weighted density; bar: 25th–75th percentiles; line: 5th–95th "
               "percentiles; white dot: median.")
        if study.train is not None:
            how += " Dashed: density of the history used to build the scenarios."
    else:
        how = "Each dot is a scenario; its area is proportional to its probability."
    return (f"Scenarios. Values of {which} across the {len(S)} scenarios ({src}), relative to their "
            f"probability-weighted means. {how}")


def source_scenarios(study):
    import pandas as pd

    S = study.scenarios
    df = pd.DataFrame(S.values, columns=S.names)
    df.insert(0, "probability", S.probs)
    df.insert(0, "scenario", range(len(S)))
    return df


def height_scenarios(study, st):
    return 18 + 7.5 * min(len(study.scenarios.paths), 8)


# ------------------------------------------------------------- out of sample
def draw_out_of_sample(study, fig, st):
    """Distribution of the realized outcome of both decisions out of sample."""
    o = study.oos
    ax = fig.add_subplot()
    rows = [(o["costs_rp"], C_RP, WASH_RP, M_RP, L_RP), (o["costs_ev"], C_EV, WASH_EV, M_EV, L_EV)]
    for y, (c, color, wash, marker, label) in enumerate(rows):
        c = c[~np.isnan(c)]
        if len(c) == 0:
            continue
        q05, q25, q50, q75, q95 = np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.95])
        ax.plot([q05, q25], [y, y], color=color, lw=st.thin * 1.4, solid_capstyle="butt")
        ax.plot([q75, q95], [y, y], color=color, lw=st.thin * 1.4, solid_capstyle="butt")
        from matplotlib.patches import Rectangle

        ax.add_patch(Rectangle((q25, y - 0.22), q75 - q25, 0.44, facecolor=wash, edgecolor=color, lw=st.thin * 1.2,
                               zorder=2))
        ax.plot([q50, q50], [y - 0.22, y + 0.22], color=color, lw=st.line, solid_capstyle="butt", zorder=3)
        ax.plot([c.mean()], [y], marker=marker, ms=st.marker * 1.15, color=color, mec="white", mew=_mew(st),
                ls="none", zorder=4)
        ax.text(1.02, y, f"mean {_num(float(c.mean()))}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=st.small, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([L_RP, L_EV])
    ax.set_ylim(-0.6, 1.6)
    _hide_y_axis(ax)
    _scaled(ax, "x", f"Realized {_unit(study)} per scenario", 5)
    if st.titles:
        ax.set_title(f"Out of sample: {o['n_compared']} unseen scenarios")


def caption_out_of_sample(study) -> str:
    o = study.oos
    risk = ""
    if "risk_gain" in o:
        risk = (f" Risk-adjusted objective: mean-value decision {_num(o['risk_ev'])}, stochastic decision "
                f"{_num(o['risk_rp'])}.")
    return (f"Out-of-sample outcomes. Realized {_unit(study)} of each decision on {o['n_compared']} "
            f"{o['source']} that were not used to make it. Box: 25th–75th percentiles with the median; "
            f"whiskers: 5th–95th percentiles; marker: mean." + risk)


def source_out_of_sample(study):
    import pandas as pd

    o = study.oos
    sign = 1.0 if study.results.sense == "min" else -1.0
    return pd.DataFrame({"scenario": range(len(o["costs_ev"])), "mean_value_decision": o["costs_ev"],
                         "stochastic_decision": o["costs_rp"],
                         "gain_of_stochastic_decision": sign * (o["costs_ev"] - o["costs_rp"])})


def height_out_of_sample(study, st):
    return 30


# ----------------------------------------------------------------------- gain
def draw_gain(study, fig, st):
    """Paired out-of-sample gain of the stochastic decision, scenario by scenario."""
    o = study.oos
    sign = 1.0 if study.results.sense == "min" else -1.0
    d = sign * (o["costs_ev"] - o["costs_rp"])
    d = d[~np.isnan(d)]
    ax = fig.add_subplot()
    lo, hi = o["gain_ci95"]
    span = d.max() - d.min()
    nb = int(np.clip(np.ceil(np.sqrt(len(d))) * 1.4, 8, 30))
    step = span / nb if span > 0 else max(abs(d[0]), 1.0) / 4
    k0, k1 = np.floor(d.min() / step), np.ceil(d.max() / step)        # a bin edge sits on zero
    if k1 <= k0:
        k0, k1 = k0 - 1, k0 + 1
    edges = np.arange(k0, k1 + 1) * step
    counts, _ = np.histogram(d, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    w = step * 0.86
    ax.bar(centers[centers < 0], counts[centers < 0], width=w, color=C_EV, lw=0, label=f"{L_EV} better")
    ax.bar(centers[centers > 0], counts[centers > 0], width=w, color=C_RP, lw=0, label=f"{L_RP} better")
    cmax = max(counts.max(), 1)
    ax.plot([0, 0], [0, cmax * 1.3], color=AXIS, lw=st.thin, zorder=1)
    top = cmax * 1.16
    ax.plot([lo, hi], [top, top], color=INK, lw=st.line, solid_capstyle="butt")
    if hi - lo > 0.03 * (edges[-1] - edges[0]):        # caps on a very short interval read as a blot
        for x in (lo, hi):
            ax.plot([x, x], [top - cmax * 0.03, top + cmax * 0.03], color=INK, lw=st.line)
    ax.plot([o["mean_gain"]], [top], marker="o", ms=st.marker, color=INK, mec="white", mew=_mew(st), ls="none")
    better = 100 * o["share_rp_better"]
    ax.text(0.99, 0.99, f"mean {_num(o['mean_gain'])}, 95% CI {_num(lo)} to {_num(hi)}\n"
            f"stochastic better in {better:.0f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=st.small, color=INK, linespacing=1.25)
    ax.set_ylim(0, cmax * 1.45)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= cmax * 1.05])
    pad = 0.04 * (edges[-1] - edges[0])
    ax.set_xlim(min(edges[0], lo) - pad, max(edges[-1], hi) + pad)
    ax.set_ylabel("Scenarios")
    _scaled(ax, "x", "Gain of the stochastic decision per scenario", 5)
    _legend(fig, st, ncol=2)
    if st.titles:
        ax.set_title("Paired comparison out of sample")


def caption_gain(study) -> str:
    o = study.oos
    lo, hi = o["gain_ci95"]
    risk = ""
    if "risk_gain" in o:
        rlo, rhi = o["risk_gain_ci95"]
        risk = (f" On the risk-adjusted objective the gain is {_num(o['risk_gain'])} (95% interval {_num(rlo)} "
                f"to {_num(rhi)}).")
    return (f"Paired out-of-sample comparison. Gain of the stochastic decision over the mean-value decision "
            f"in each of {o['n_compared']} {o['source']} (positive: the stochastic decision is better). "
            f"Black dot and bar: mean gain {_num(o['mean_gain'])} and its 95% bootstrap interval "
            f"({_num(lo)} to {_num(hi)}; {o.get('n_boot', 10000):,} resamples, treating scenarios as "
            f"independent). The stochastic decision is better in {100 * o['share_rp_better']:.0f}% and "
            f"worse in {100 * o['share_ev_better']:.0f}% of the scenarios." + risk)


source_gain = source_out_of_sample


def height_gain(study, st):
    return 50


# ------------------------------------------------------------------ stability
def draw_stability(study, fig, st):
    """Does the answer settle as the number of scenarios grows?"""
    rows = study.stability_table
    sizes = sorted({row["n"] for row in rows})
    ax = fig.add_subplot()
    xs = np.arange(len(sizes))
    series = [("in_sample", C_EV, WASH_EV, M_EV, "In-sample optimum")]
    if "holdout" in rows[0]:
        series.append(("holdout", C_RP, WASH_RP, M_RP, "Same decision, out of sample"))
    for key, color, wash, marker, label in series:
        vals = [np.array([row[key] for row in rows if row["n"] == n], dtype=float) for n in sizes]
        mean = np.array([np.nanmean(v) for v in vals])
        ax.fill_between(xs, [np.nanmin(v) for v in vals], [np.nanmax(v) for v in vals], color=wash, lw=0,
                        alpha=0.85, zorder=1)
        ax.plot(xs, mean, color=color, lw=st.line, marker=marker, ms=st.marker * 1.1, mec="white",
                mew=_mew(st), label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Scenarios per sampled set")
    _scaled(ax, "y", _objective_label(study), 5)
    _legend(fig, st, ncol=len(series))
    if st.titles:
        ax.set_title("Stability across resampled scenario sets")


def caption_stability(study) -> str:
    rows = study.stability_table
    sizes = sorted({row["n"] for row in rows})
    reps = sum(row["n"] == sizes[0] for row in rows)
    hold = (" and its result out of sample (orange squares)" if "holdout" in rows[0] else "")
    return (f"Stability. Optimal value of the stochastic program on {reps} randomly drawn scenario sets of "
            f"each size (blue circles){hold}. Line: mean over the sets; band: range.")


def source_stability(study):
    import pandas as pd

    return pd.DataFrame(study.stability_table)


def height_stability(study, st):
    return 50


# ------------------------------------------------------------------- frontier
def draw_frontier(study, fig, st):
    """Expected outcome against CVaR as the weight on CVaR grows."""
    from .risk import cvar

    rows = study.frontier
    a = rows[0]["alpha"]
    res = study.results
    ax = fig.add_subplot()
    x = [r["cvar"] for r in rows]
    y = [r["mean"] for r in rows]
    hold = "holdout_mean" in rows[0]
    ax.plot(x, y, color=C_RP, lw=st.line, marker=M_RP, ms=st.marker * 1.1, mec="white", mew=_mew(st),
            label="Stochastic decision on the scenarios", zorder=3)
    if hold:
        ax.plot([r["holdout_cvar"] for r in rows], [r["holdout_mean"] for r in rows], color=C_RP, lw=st.thin * 1.6,
                ls=(0, (3, 1.5)), marker=M_RP, ms=st.marker, mfc="white", mec=C_RP, mew=st.thin * 1.6,
                label="Same decisions, out of sample", zorder=2)
    for r in rows:
        ax.annotate(f"{r['weight']:g}", (r["cvar"], r["mean"]), textcoords="offset points",
                    xytext=(3, 3), fontsize=st.small - 0.5, color=INK2)
    c_ev = res.scenario_costs_ev
    note = None
    if c_ev is not None and np.isfinite(c_ev).all():
        sign = -1.0 if res.sense == "max" else 1.0
        internal = sign * c_ev
        ev_x = sign * cvar(internal, study.scenarios.probs, a)
        ev_y = sign * float(study.scenarios.probs @ internal)
        xs = x + ([r["holdout_cvar"] for r in rows] if hold else [])
        ys = y + ([r["holdout_mean"] for r in rows] if hold else [])
        sx = max(np.ptp(xs), 1e-9 * max(1.0, abs(np.mean(xs))))
        sy = max(np.ptp(ys), 1e-9 * max(1.0, abs(np.mean(ys))))
        far = (not min(xs) - 2 * sx <= ev_x <= max(xs) + 2 * sx) or (not min(ys) - 2 * sy <= ev_y <= max(ys) + 2 * sy)
        if far:
            note = f"{L_EV}: expected {_tick(ev_y)}, CVaR {_tick(ev_x)} (off scale)"
        else:
            ax.plot([ev_x], [ev_y], marker=M_EV, ms=st.marker * 1.25, color=C_EV, mec="white", mew=_mew(st),
                    ls="none", label=L_EV, zorder=4)
    if note:
        ax.text(0.99, 0.99, note, transform=ax.transAxes, ha="right", va="top", fontsize=st.small - 0.5,
                color=INK2)
    worst = "highest costs" if res.sense == "min" else "lowest values"
    _scaled(ax, "x", f"CVaR at {a:g}, mean of the worst {100 * (1 - a):g}%", 5)
    _scaled(ax, "y", f"Expected {_unit(study)}", 5)
    _legend(fig, st, ncol=2)
    if st.titles:
        ax.set_title("Mean–risk trade-off (labels: weight on CVaR)")


def caption_frontier(study) -> str:
    rows = study.frontier
    a = rows[0]["alpha"]
    hold = " Open squares: the same decisions out of sample." if "holdout_mean" in rows[0] else ""
    return (f"Mean–risk trade-off. Expected {_unit(study)} against CVaR at {a:g} of the decisions that "
            f"minimize (1 − w) × expectation + w × CVaR for the weights w shown, evaluated on the "
            f"scenario set (filled squares).{hold} The blue circle is the mean-value decision.")


def source_frontier(study):
    import pandas as pd

    return pd.DataFrame([{k: v for k, v in r.items() if k != "first_stage"} for r in study.frontier])


def height_frontier(study, st):
    return 58


# ------------------------------------------------------------------- assembly
PANELS = ["value", "first_stage", "scenarios", "out_of_sample", "gain", "stability", "frontier"]
FILES = {"value": "fig_value", "first_stage": "fig_first_stage", "scenarios": "fig_scenarios",
         "out_of_sample": "fig_out_of_sample", "gain": "fig_gain", "stability": "fig_stability",
         "frontier": "fig_risk_frontier"}


def available(study) -> list:
    out = ["value", "first_stage", "scenarios"]
    if study.oos and study.oos["n_compared"] > 1:
        out += ["out_of_sample", "gain"]
    if study.stability_table:
        out.append("stability")
    if study.frontier:
        out.append("frontier")
    return out


def _height(study, key, st, width_mm):
    return globals()[f"height_{key}"](study, st) * width_mm / 89


def figure(study, key: str, style="nature"):
    """One figure as a matplotlib ``Figure``, at one-column width."""
    import matplotlib as mpl
    from matplotlib.figure import Figure

    st = get_style(style)
    with mpl.rc_context(st.rc()):
        h = min(_height(study, key, st, st.single), st.max_height)
        fig = Figure(figsize=(st.single * MM, h * MM))
        globals()[f"draw_{key}"](study, fig, st)
    return fig


def overview(study, style="nature", panels=None):
    """All available panels in one two-column figure, lettered a, b, c, ..."""
    import matplotlib as mpl
    from matplotlib.figure import Figure

    st = get_style(style)
    keys = list(panels or [k for k in available(study) if k != "out_of_sample"])[:6]
    with mpl.rc_context(st.rc()):
        half = st.double / 2
        layout = []                       # list of rows; a row is one full-width key or two keys
        rest = list(keys)
        if len(rest) % 2:
            layout.append([rest.pop(0)])
        while rest:
            layout.append([rest.pop(0), rest.pop(0)])
        # a panel keeps its natural height when it spans both columns: it only gets wider
        heights = [max(_height(study, k, st, half) for k in row) for row in layout]
        total = min(sum(heights), st.max_height)
        fig = Figure(figsize=(st.double * MM, total * MM))
        gs = fig.add_gridspec(len(layout), 2, height_ratios=heights)
        letter = iter("abcdefghij")
        for i, row in enumerate(layout):
            for j, key in enumerate(row):
                sf = fig.add_subfigure(gs[i, :] if len(row) == 1 else gs[i, j])
                globals()[f"draw_{key}"](study, sf, st)
                sf.suptitle(next(letter), x=0.0, y=1.0, ha="left", va="top", fontsize=st.panel,
                            fontweight="bold")
    return fig, keys


def save(fig, outdir, name, style="nature") -> list:
    """Write ``fig`` in every format of the style. Font type and font lookup are read when a
    file is written, so writing happens inside the style too (PDF Type 42 fonts, SVG text)."""
    import matplotlib as mpl

    st = get_style(style)
    paths = []
    with mpl.rc_context(st.rc()):
        for ext in st.formats:
            path = os.path.join(outdir, f"{name}.{ext}")
            fig.savefig(path, dpi=st.dpi if ext in ("png", "tif", "tiff") else None)
            paths.append(path)
    return paths


def caption(study, key) -> str:
    return globals()[f"caption_{key}"](study)


def source(study, key):
    return globals()[f"source_{key}"](study)


def make_figures(study, outdir, style="nature") -> list:
    """Write every available figure, the overview, ``captions.md`` and ``source_data/``.

    Returns the paths of the PDF files.
    """
    st = get_style(style)
    os.makedirs(os.path.join(outdir, "source_data"), exist_ok=True)
    made, legends = [], []
    keys = available(study)
    for n, key in enumerate(keys, start=1):
        paths = save(figure(study, key, st), outdir, FILES[key], st)
        made.append(next((p for p in paths if p.endswith(".pdf")), paths[0]))
        source(study, key).to_csv(os.path.join(outdir, "source_data", f"{FILES[key]}.csv"), index=False)
        text = caption(study, key)
        head, _, body = text.partition(". ")
        legends.append(f"**Fig. {n} | {head}.** {body}  \n`{FILES[key]}` · source data: "
                       f"`source_data/{FILES[key]}.csv`")
    fig, panel_keys = overview(study, st)
    paths = save(fig, outdir, "fig_overview", st)
    made.insert(0, next((p for p in paths if p.endswith(".pdf")), paths[0]))
    parts = []
    for letter, key in zip("abcdefghij", panel_keys):
        head, _, body = caption(study, key).partition(". ")
        parts.append(f"**{letter},** {head.lower() if not head.startswith(('EVPI', 'VSS')) else head}: {body}")
    legends.insert(0, "**Overview | Lifting the deterministic model to a two-stage stochastic program.** "
                   + " ".join(parts) + "  \n`fig_overview`")
    with open(os.path.join(outdir, "captions.md"), "w", encoding="utf-8") as f:
        f.write("# Figure legends\n\nDraft legends for each figure, generated from the results. Edit them to "
                "describe your model.\n\n" + "\n\n".join(legends) + "\n")
    return made
