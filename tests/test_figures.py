"""Figures meet journal artwork rules: final size, 5-8 pt text, no titles, editable text,
source data that matches the results, and a legend for every figure."""
import os
import re

import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

import stochlift as sl
from stochlift import plots
from stochlift.figstyle import MM, get_style


@pytest.fixture(scope="module")
def study(farmer):
    spec = {"first_stage": ["acres_*"], "risk": {"alpha": 0.8, "weight": 0.5},
            "scenarios": {"method": "distribution", "n": 30, "n_test": 60, "seed": 1, "correlation": 0.8,
                          "distributions": {"yield": {"dist": "normal", "cv": 0.15, "min": 0}}}}
    s = sl.lift(farmer.build_model, farmer.DATA, spec=spec)
    s.solve()
    s.out_of_sample()
    s.stability(sizes=(5, 10), reps=2)
    s.risk_frontier(weights=(0.0, 0.5, 1.0))
    return s


def _texts(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def test_every_figure_is_available(study):
    assert plots.available(study) == ["value", "first_stage", "scenarios", "out_of_sample", "gain",
                                      "stability", "frontier"]


@pytest.mark.parametrize("key", plots.PANELS)
def test_nature_size_text_and_lines(study, key):
    fig = plots.figure(study, key, "nature")
    w, h = fig.get_size_inches()
    assert w / MM == pytest.approx(89, abs=0.01) and h / MM <= 230
    sizes = {t.get_fontsize() for t in _texts(fig)}
    assert sizes and min(sizes) >= 5 and max(sizes) <= 8, sizes
    widths = [ln.get_linewidth() for ln in fig.findobj(Line2D) if ln.get_visible() and ln.get_linestyle() != "None"]
    assert min(widths) >= 0.25
    assert all(not ax.get_title(loc=loc) for ax in fig.axes for loc in ("left", "center", "right"))  # no titles


def test_presentation_style_has_titles(study):
    fig = plots.figure(study, "value", "presentation")
    assert any(ax.get_title(loc="left") for ax in fig.axes)
    assert fig.get_size_inches()[0] / MM == pytest.approx(get_style("presentation").single)


def test_overview_is_two_columns_with_panel_letters(study):
    fig, keys = plots.overview(study, "nature")
    assert fig.get_size_inches()[0] / MM == pytest.approx(183, abs=0.01)
    assert fig.get_size_inches()[1] / MM <= 230
    letters = [t.get_text() for t in _texts(fig) if re.fullmatch(r"[a-f]", t.get_text())]
    assert letters == list("abcdef")[:len(keys)] and len(keys) == 6
    assert {t.get_fontsize() for t in _texts(fig) if t.get_text() in letters} == {8}


def test_written_files(study, tmp_path):
    made = study.figures(tmp_path)
    assert os.path.basename(made[0]) == "fig_overview.pdf" and len(made) == 8
    pdf = (tmp_path / "fig_value.pdf").read_bytes()
    assert b"/Type3" not in pdf                                   # TrueType text, editable in Illustrator
    box = [float(v) for v in re.search(rb"/MediaBox \[\s*([\d. ]+)\]", pdf).group(1).split()]
    assert (box[2] - box[0]) == pytest.approx(89 / 25.4 * 72, abs=0.6)   # exactly one column wide
    assert (tmp_path / "fig_value.svg").read_text(encoding="utf-8").count("<text") > 5
    assert (tmp_path / "fig_overview.png").stat().st_size > 50_000


def test_source_data_and_legends(study, tmp_path):
    import pandas as pd

    study.figures(tmp_path)
    value = pd.read_csv(tmp_path / "source_data" / "fig_value.csv").set_index("quantity")["value"]
    r = study.results
    assert value["RP"] == pytest.approx(r.rp) and value["VSS"] == pytest.approx(r.vss)
    gain = pd.read_csv(tmp_path / "source_data" / "fig_gain.csv")
    assert gain["gain_of_stochastic_decision"].mean() == pytest.approx(study.oos["mean_gain"])
    text = (tmp_path / "captions.md").read_text(encoding="utf-8")
    assert "**Overview |" in text and "**a,**" in text and "**f,**" in text
    assert text.count("**Fig. ") == 7 and "95% bootstrap interval" in text


def test_report_embeds_the_overview(study, tmp_path):
    study.report(tmp_path)
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "![Overview](fig_overview.png)" in md and "captions.md" in md
