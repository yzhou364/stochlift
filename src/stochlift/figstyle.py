"""Figure styles. ``nature`` follows the artwork guidelines of Nature-family journals.

- Widths are final print sizes: 89 mm (one column) and 183 mm (two columns).
- Text is a sans-serif face (Arial or Helvetica when installed) at 5-7 pt, panel
  labels are bold lower-case letters at 8 pt, and figures carry no titles: what a
  figure shows goes in its legend (see ``captions.md`` in the report).
- Lines are at least 0.5 pt; vector output keeps text editable (PDF Type 42 fonts,
  SVG text elements).

``presentation`` is the same design at slide scale, with titles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

MM = 1 / 25.4                       # inches per millimetre

# Color follows the entity, in every figure. The three slots pass the all-pairs
# colour-vision checks on white; marker shape repeats the identity for print.
C_EV, C_RP, C_WS = "#2a78d6", "#eb6834", "#1baf7a"
M_EV, M_RP, M_WS = "o", "s", "D"
L_EV, L_RP, L_WS = "Mean-value decision", "Stochastic decision", "Perfect information"
INK, INK2, MUTED = "#111111", "#4d4c48", "#8a8984"
GRID, AXIS, CONTEXT = "#e4e3dd", "#3a3a38", "#c4c3bb"
WASH_EV, WASH_RP = "#cfe0f5", "#f9d3c3"     # light steps of the two decision hues

FONTS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


@dataclass(frozen=True)
class Style:
    name: str
    single: float                   # one-column width, mm
    double: float                   # two-column width, mm
    size: float                     # axis labels, legends (pt)
    small: float                    # tick labels, annotations (pt)
    panel: float                    # panel letters (pt)
    line: float                     # data lines (pt)
    thin: float                     # axes, ticks, rules (pt)
    marker: float                   # marker size (pt)
    titles: bool                    # draw a title on each figure
    formats: tuple = ("pdf", "svg", "png")
    dpi: int = 600
    max_height: float = 230.0       # mm, leaves room for the legend on the page
    fonts: list = field(default_factory=lambda: list(FONTS))

    def rc(self) -> dict:
        face = _first_available(tuple(self.fonts))
        faces = [face] if face == "DejaVu Sans" else [face, "DejaVu Sans"]   # DejaVu covers rare glyphs
        return {
            # the face names themselves, not the generic "sans-serif": generic names are resolved
            # when the file is written, which may be outside this style's context
            "font.family": faces, "font.sans-serif": faces, "font.size": self.size,
            "mathtext.fontset": "custom", "mathtext.rm": face, "mathtext.it": f"{face}:italic",
            "axes.labelsize": self.size, "axes.titlesize": self.size + 1, "axes.titleweight": "bold",
            "axes.titlelocation": "left", "axes.titlepad": 4 * self.size / 7,
            "xtick.labelsize": self.small, "ytick.labelsize": self.small, "legend.fontsize": self.small,
            "axes.linewidth": self.thin, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
            "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
            "xtick.major.width": self.thin, "ytick.major.width": self.thin,
            "xtick.major.size": 2.5 * self.size / 7, "ytick.major.size": 2.5 * self.size / 7,
            "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.pad": 2 * self.size / 7, "ytick.major.pad": 2 * self.size / 7,
            "axes.labelpad": 3 * self.size / 7,
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
            "grid.color": GRID, "grid.linewidth": self.thin,
            "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0, "legend.labelspacing": 0.3,
            "lines.linewidth": self.line, "lines.markersize": self.marker, "lines.solid_capstyle": "round",
            "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "savefig.facecolor": "white", "figure.facecolor": "white", "savefig.dpi": self.dpi,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 1.5 * MM, "figure.constrained_layout.w_pad": 1.5 * MM,
            "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
        }


STYLES = {
    "nature": Style("nature", single=89, double=183, size=7, small=6, panel=8, line=1.0, thin=0.5,
                    marker=4.0, titles=False),
    "presentation": Style("presentation", single=150, double=300, size=12, small=10.5, panel=14,
                          line=2.0, thin=0.9, marker=7.5, titles=True, dpi=200, max_height=400),
}


def get_style(style) -> Style:
    if isinstance(style, Style):
        return style
    if style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}, got {style!r}")
    return STYLES[style]


@lru_cache(maxsize=None)
def _first_available(fonts: tuple) -> str:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in fonts if f in available), "DejaVu Sans")


def font_in_use(style) -> str:
    """The first face of the style's list that matplotlib can find (Arial or Helvetica for
    journals; on Linux install them, or Liberation Sans, which has the same metrics as Arial)."""
    return _first_available(tuple(get_style(style).fonts))
