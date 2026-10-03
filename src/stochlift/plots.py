"""Publication figures. Every figure is written as vector PDF and 300-dpi PNG.

Colors follow the entity, not the chart: the mean-value (deterministic)
decision is always blue with round markers, the stochastic decision is always
orange with square markers, perfect information is always green with diamonds.
The three colors were checked for color-vision-deficiency separation; marker
shape and direct labels carry the same identity for grayscale print.
"""
from __future__ import annotations

import os

import numpy as np

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_EV, C_RP, C_WS, C_CONTEXT = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1"
M_EV, M_RP, M_WS = "o", "s", "D"
L_EV, L_RP, L_WS = "Mean-value decision", "Stochastic decision", "Perfect information"


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    })
    return plt


def _num(v: float, scale: float = None) -> str:
    """Format a number; ``scale`` (e.g. |RP|) fixes the precision for a whole figure."""
    if not np.isfinite(v):
        return "infeasible"
    ref = abs(v) if scale is None else abs(scale)
    if ref >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.1f}" if ref >= 100 else f"{v:,.2f}"


def _legend_below(fig, ax, ncol=2, gap_in=0.55):
    """Legend under the x-axis label, a fixed distance below the axes."""
    h_in = fig.get_size_inches()[1] * ax.get_position().height
    ax.legend(loc="upper left", bbox_to_anchor=(0, -gap_in / h_in), ncol=ncol, handletextpad=0.4,
              columnspacing=1.5, borderaxespad=0)


def _tick(v: float) -> str:
    """Compact tick label: 950, 12,500, 3.4M, 1.25B."""
    a = abs(v)
    for scale, suffix in ((1e9, "B"), (1e6, "M")):
        if a >= scale:
            return f"{v / scale:,.3g}{suffix}"
    if a >= 1000 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:,.2g}"


def _thousands(ax, axis="x"):
    from matplotlib.ticker import FuncFormatter

    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(FuncFormatter(lambda v, _: _tick(v)))


def _save(fig, outdir, name) -> str:
    path = os.path.join(outdir, name)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path + ".pdf"


def _unit(study) -> str:
    return "cost" if study.results.sense == "min" else "value"


# ----------------------------------------------------------------------------- 1
def fig_value(study, outdir, title=True) -> str:
    """WS, RP, EEV and the mean-value model's own estimate on one scale."""
    plt = _plt()
    r = study.results
    rows = [("Perfect information (WS)", r.ws, C_WS, M_WS),
            ("Stochastic decision (RP)", r.rp, C_RP, M_RP),
            ("Mean-value decision, evaluated (EEV)", r.eev, C_EV, M_EV)]
    finite = [v for _, v, _, _ in rows if np.isfinite(v)] + [r.ev]
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or max(1.0, abs(hi))
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys = [3, 2, 1]
    sc = r.rp
    x_right = hi + 0.14 * span

    def value_label(y, v, color=INK):
        ax.text(1.02, y, _num(v, sc), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=color, clip_on=False)

    for y, (label, v, color, marker) in zip(ys, rows):
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        if np.isfinite(v):
            ax.plot([v], [y], marker=marker, ms=8, color=color, mec="white", mew=1.5, zorder=3, ls="none")
        value_label(y, v)
    # the deterministic model's own, optimistic estimate as context
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.plot([r.ev], [0], marker="o", ms=7, mfc="white", mec=MUTED, mew=1.5, ls="none", zorder=3)
    value_label(0, r.ev, INK2)

    def gap(y, a, b, text):
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) < 1e-9 * max(1, abs(a)):
            return
        ax.plot([a, b], [y, y], color=INK2, lw=1.2, solid_capstyle="butt", zorder=2)
        for x in (a, b):
            ax.plot([x, x], [y - 0.09, y + 0.09], color=INK2, lw=1.2, zorder=2)
        ax.text((a + b) / 2, y + 0.14, text, ha="center", va="bottom", fontsize=8.5, color=INK)

    gap(2.42, r.ws, r.rp, f"EVPI = {_num(r.evpi, sc)}")
    gap(1.42, r.rp, r.eev, f"VSS = {_num(r.vss, sc)}")
    ax.set_yticks(ys + [0])
    ax.set_yticklabels([row[0] for row in rows] + ["Mean-value model's own estimate (EV)"])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(lo - 0.12 * span, x_right)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Expected {_unit(study)} ({'lower' if r.sense == 'min' else 'higher'} is better)")
    _thousands(ax)
    if title:
        ax.set_title("What the uncertainty costs, and what modeling it recovers")
    return _save(fig, outdir, "fig_value")


# ----------------------------------------------------------------------------- 2
def fig_out_of_sample(study, outdir, title=True) -> str:
    """Empirical distribution of hold-out results for both decisions."""
    plt = _plt()
    o = study.oos
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for costs, color, label, marker in ((o["costs_ev"], C_EV, L_EV, M_EV), (o["costs_rp"], C_RP, L_RP, M_RP)):
        c = np.sort(costs[~np.isnan(costs)])
        if len(c) == 0:
            continue
        y = np.arange(1, len(c) + 1) / len(c)
        ax.step(np.concatenate([[c[0]], c]), np.concatenate([[0], y]), where="post", color=color, lw=2,
                solid_joinstyle="round", label=f"{label} (mean {_num(float(c.mean()))})")
        ax.plot([c.mean()], [np.searchsorted(c, c.mean()) / len(c)], marker=marker, ms=8, color=color,
                mec="white", mew=1.5, ls="none", zorder=3)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of out-of-sample scenarios at or below")
    ax.set_xlabel(f"Realized {_unit(study)} per scenario (marker = mean)")
    _thousands(ax)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(f"Out-of-sample results on {o['n_compared']} unseen scenarios")
    return _save(fig, outdir, "fig_out_of_sample")


# ----------------------------------------------------------------------------- 3
def fig_gain(study, outdir, title=True) -> str:
    """Paired hold-out gain of the stochastic decision, observation by observation."""
    plt = _plt()
    o = study.oos
    sign = 1.0 if study.results.sense == "min" else -1.0
    d = sign * (o["costs_ev"] - o["costs_rp"])
    d = d[~np.isnan(d)]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    lo, hi = o["gain_ci95"]
    sc = max(np.abs(d).max(), abs(study.results.rp) * 1e-3, 1e-12)
    span = d.max() - d.min()
    nb = int(np.clip(np.ceil(np.sqrt(len(d))) * 1.6, 8, 24))
    step = span / nb if span > 0 else max(abs(d[0]), 1.0) / 4
    k0, k1 = np.floor(d.min() / step), np.ceil(d.max() / step)        # a bin edge sits on zero
    if k1 <= k0:                                                      # every gain is the same value
        k0, k1 = k0 - 1, k0 + 1
    edges = np.arange(k0, k1 + 1) * step
    counts, _ = np.histogram(d, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers[centers < 0], counts[centers < 0], width=step * 0.88, color=C_EV,
           label="Mean-value decision better")
    ax.bar(centers[centers > 0], counts[centers > 0], width=step * 0.88, color=C_RP,
           label="Stochastic decision better")
    ax.axvline(0, color=AXIS, lw=0.8)
    cmax = counts.max()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, cmax * 1.42)
    ax.set_yticks([t for t in ax.get_yticks() if t <= cmax * 1.05])
    ax.set_ylim(0, cmax * 1.42)
    top = cmax * 1.18
    ax.plot([lo, hi], [top, top], color=INK, lw=1.5, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [top - cmax * 0.025, top + cmax * 0.025], color=INK, lw=1.5)
    ax.plot([o["mean_gain"]], [top], marker="o", ms=6, color=INK, mec="white", mew=1.2, ls="none")
    ax.text(1.0, 0.985, f"Mean gain {_num(o['mean_gain'], sc)}, 95% interval {_num(lo, sc)} to {_num(hi, sc)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
    pad = 0.05 * (edges[-1] - edges[0])
    ax.set_xlim(min(edges[0], lo) - pad, max(edges[-1], hi) + pad)
    ax.set_ylabel("Out-of-sample scenarios")
    ax.set_xlabel("Gain from the stochastic decision, per scenario")
    _thousands(ax)
    _legend_below(fig, ax)
    if title:
        ax.set_title("Paired comparison out of sample")
    return _save(fig, outdir, "fig_gain")


# ----------------------------------------------------------------------------- 4
def fig_first_stage(study, outdir, title=True, max_rows=18) -> str:
    """The here-and-now decision: mean-value model versus stochastic model."""
    plt = _plt()
    r = study.results
    names = list(r.x_rp)
    diff = {nm: abs(r.x_ev.get(nm, 0.0) - r.x_rp[nm]) for nm in names}
    changed = [nm for nm in names if diff[nm] > 1e-6]
    rest = sorted((nm for nm in names if nm not in changed), key=lambda nm: -abs(r.x_rp[nm]))
    shown = (sorted(changed, key=lambda nm: -diff[nm]) + rest)[:max_rows]
    shown = shown[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.9 + 0.34 * len(shown)))
    any_same = False
    for y, nm in enumerate(shown):
        a, b = r.x_ev.get(nm, np.nan), r.x_rp[nm]
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        if nm not in changed:
            any_same = True
            ax.plot([b], [y], marker="D", ms=6.5, color=MUTED, mec="white", mew=1.5, ls="none", zorder=3)
            continue
        ax.plot([a, b], [y, y], color=C_CONTEXT, lw=2, zorder=1, solid_capstyle="round")
        ax.plot([a], [y], marker=M_EV, ms=8, color=C_EV, mec="white", mew=1.5, ls="none", zorder=3)
        ax.plot([b], [y], marker=M_RP, ms=7.5, color=C_RP, mec="white", mew=1.5, ls="none", zorder=4)
    ax.plot([], [], marker=M_EV, ms=8, color=C_EV, ls="none", label=L_EV)
    ax.plot([], [], marker=M_RP, ms=7.5, color=C_RP, ls="none", label=L_RP)
    if any_same:
        ax.plot([], [], marker="D", ms=6.5, color=MUTED, ls="none", label="Same in both")
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown)
    ax.set_ylim(-0.6, len(shown) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.margins(x=0.08)
    ax.set_xlabel("First-stage value")
    _thousands(ax)
    _legend_below(fig, ax, ncol=3)
    if title:
        more = f" (showing {len(shown)} of {len(names)})" if len(names) > len(shown) else ""
        ax.set_title(f"{len(changed)} of {len(names)} first-stage decisions change{more}")
    return _save(fig, outdir, "fig_first_stage")


# ----------------------------------------------------------------------------- 5
def fig_scenarios(study, outdir, title=True, max_rows=10) -> str:
    """Scenarios against the history they were built from, relative to the mean."""
    plt = _plt()
    S = study.scenarios
    mu = S.mean()
    ok = np.abs(mu) > 1e-12
    cv = np.where(ok, np.sqrt(S.probs @ (S.values - mu) ** 2) / np.where(ok, np.abs(mu), 1), 0)
    order = [j for j in np.argsort(-cv) if ok[j]][:max_rows][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 1.0 + 0.36 * len(order)))
    rng = np.random.default_rng(0)
    size_scale = 110 if len(S) <= 10 else 55
    for y, j in enumerate(order):
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        if study.train is not None:
            h = study.train[:, j] / mu[j]
            ax.plot(h, y + rng.uniform(-0.16, 0.16, len(h)), ls="none", marker="o", ms=2.2,
                    color=C_CONTEXT, alpha=0.7, zorder=1)
        ax.scatter(S.values[:, j] / mu[j], np.full(len(S), y), s=14 + size_scale * S.probs / S.probs.max(),
                   color=C_EV, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(1.0, color=AXIS, lw=0.8, zorder=0)
    ax.scatter([], [], s=45, color=C_EV, label="Scenario (area = probability)")
    if study.train is not None:
        ax.plot([], [], ls="none", marker="o", ms=3, color=C_CONTEXT, label="Historical observation")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([S.names[j] for j in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Value relative to the scenario mean")
    _legend_below(fig, ax)
    if title:
        more = f", {len(order)} most variable of {len(S.paths)} entries" if len(S.paths) > len(order) else ""
        ax.set_title(f"{len(S)} scenarios{more}")
    return _save(fig, outdir, "fig_scenarios")


# ----------------------------------------------------------------------------- 6
def fig_stability(study, outdir, title=True) -> str:
    """Does the answer settle as the number of scenarios grows?"""
    plt = _plt()
    rows = study.stability_table
    sizes = sorted({row["n"] for row in rows})
    has_hold = "holdout" in rows[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    series = [("in_sample", C_EV, M_EV, "In-sample optimum")]
    if has_hold:
        series.append(("holdout", C_RP, M_RP, "Out-of-sample result of the same decision"))
    xs = np.arange(len(sizes))
    dodge = 0.07 if len(series) > 1 else 0.0
    for k, (key, color, marker, label) in enumerate(series):
        vals = [np.array([row[key] for row in rows if row["n"] == n], dtype=float) for n in sizes]
        x = xs + (k - (len(series) - 1) / 2) * 2 * dodge
        mean = np.array([np.nanmean(v) for v in vals])
        ax.vlines(x, [np.nanmin(v) for v in vals], [np.nanmax(v) for v in vals], color=color, lw=1.2,
                  alpha=0.55)
        ax.plot(x, mean, color=color, lw=2, marker=marker, ms=7, mec="white", mew=1.5, label=label,
                solid_joinstyle="round")
    reps = sum(row["n"] == sizes[0] for row in rows)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Scenarios per sampled set ({reps} sets each; bars show the range)")
    ax.set_ylabel(f"Expected {_unit(study)}" if not study.risk.active else "Risk-adjusted objective")
    _thousands(ax, "y")
    _legend_below(fig, ax)
    if title:
        ax.set_title("Stability across resampled scenario sets")
    return _save(fig, outdir, "fig_stability")


# ----------------------------------------------------------------------------- 7
def fig_risk_frontier(study, outdir, title=True) -> str:
    """Expected outcome against CVaR as the weight on CVaR grows."""
    from .risk import cvar

    plt = _plt()
    rows = study.frontier
    a = rows[0]["alpha"]
    unit = _unit(study)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = [r["cvar"] for r in rows]
    y = [r["mean"] for r in rows]
    ax.plot(x, y, color=C_RP, lw=2, marker=M_RP, ms=7, mec="white", mew=1.5,
            label="Stochastic decision, by CVaR weight", solid_joinstyle="round")
    for r in rows:
        ax.annotate(f"{r['weight']:g}", (r["cvar"], r["mean"]), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=INK2)
    if "holdout_mean" in rows[0]:
        ax.plot([r["holdout_cvar"] for r in rows], [r["holdout_mean"] for r in rows], color=C_RP, lw=1.2,
                ls="--", marker=M_RP, ms=6, mfc="white", mec=C_RP, mew=1.2,
                label="Same decisions, out of sample")
    res = study.results
    c_ev = res.scenario_costs_ev
    if c_ev is not None and np.isfinite(c_ev).all():
        sign = -1.0 if res.sense == "max" else 1.0
        internal = sign * c_ev
        ev_x = sign * cvar(internal, study.scenarios.probs, a)
        ev_y = sign * float(study.scenarios.probs @ internal)
        xs = x + ([r["holdout_cvar"] for r in rows] if "holdout_mean" in rows[0] else [])
        ys = y + ([r["holdout_mean"] for r in rows] if "holdout_mean" in rows[0] else [])
        span_x = max(np.ptp(xs), 1e-9 * max(1.0, abs(np.mean(xs))))
        span_y = max(np.ptp(ys), 1e-9 * max(1.0, abs(np.mean(ys))))
        far = (min(abs(ev_x - min(xs)), abs(ev_x - max(xs))) > 2 * span_x and not min(xs) <= ev_x <= max(xs)) or               (min(abs(ev_y - min(ys)), abs(ev_y - max(ys))) > 2 * span_y and not min(ys) <= ev_y <= max(ys))
        if far:
            # drawing it would squash the frontier into a corner: state its values instead
            ax.text(0.98, 0.97, f"{L_EV} (off the chart):\nexpected {_tick(ev_y)}, CVaR {_tick(ev_x)}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C_EV)
        else:
            ax.plot([ev_x], [ev_y], ls="none", marker=M_EV, ms=8, color=C_EV, mec="white", mew=1.5, label=L_EV)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    worst = "highest costs" if res.sense == "min" else "lowest values"
    ax.set_xlabel(f"CVaR at {a:g}: mean of the {100 * (1 - a):g}% {worst}")
    ax.set_ylabel(f"Expected {unit}")
    _thousands(ax, "x")
    _thousands(ax, "y")
    _legend_below(fig, ax, ncol=2)
    if title:
        ax.set_title("Mean-risk trade-off (labels: weight on CVaR)")
    return _save(fig, outdir, "fig_risk_frontier")


def make_figures(study, outdir, title=True) -> list:
    made = [fig_value(study, outdir, title), fig_first_stage(study, outdir, title),
            fig_scenarios(study, outdir, title)]
    if study.oos and study.oos["n_compared"] > 1:
        made += [fig_out_of_sample(study, outdir, title), fig_gain(study, outdir, title)]
    if study.stability_table:
        made.append(fig_stability(study, outdir, title))
    if study.frontier:
        made.append(fig_risk_frontier(study, outdir, title))
    return made
