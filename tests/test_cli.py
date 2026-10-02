"""The command line: init writes a template, run writes the report."""
import json
import os
import shutil

import pytest

from stochlift.cli import load_model, main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FARMER = os.path.join(ROOT, "examples", "farmer")


@pytest.fixture
def farmer_dir(tmp_path):
    pytest.importorskip("pulp")
    for name in ("model.py", "uncertainty_distribution.yaml"):
        shutil.copy(os.path.join(FARMER, name), tmp_path / name)
    return tmp_path


def test_load_model_variants(farmer_dir):
    build, data = load_model(str(farmer_dir / "model.py"))      # absolute path with a drive letter on Windows
    assert callable(build) and "yield" in data
    build2, _ = load_model(f"{farmer_dir / 'model.py'}:build_model")
    assert build2 is build or build2.__name__ == "build_model"
    (farmer_dir / "data.json").write_text(json.dumps({**data, "land": 400}))
    _, d = load_model(str(farmer_dir / "model.py"), str(farmer_dir / "data.json"))
    assert d["land"] == 400
    with pytest.raises(SystemExit, match="no function 'make'"):
        load_model(f"{farmer_dir / 'model.py'}:make")
    with pytest.raises(SystemExit, match="no variable 'NOPE'"):
        load_model(str(farmer_dir / "model.py"), "NOPE")


def test_init_writes_a_template_and_keeps_existing_files(farmer_dir, capsys):
    out = farmer_dir / "u.yaml"
    assert main(["init", str(farmer_dir / "model.py"), "-o", str(out)]) == 0
    text = out.read_text()
    assert "acres_*" in text and "method: distribution" in text and "yield" in text
    with pytest.raises(SystemExit, match="exists"):
        main(["init", str(farmer_dir / "model.py"), "-o", str(out)])
    assert main(["init", str(farmer_dir / "model.py"), "-o", str(out), "--force"]) == 0


def test_run_writes_the_report(farmer_dir, capsys):
    spec = farmer_dir / "uncertainty_distribution.yaml"
    spec.write_text(spec.read_text().replace("n: 100", "n: 20").replace("n_test: 500", "n_test: 30"))
    out = farmer_dir / "report"
    code = main(["run", str(farmer_dir / "model.py"), "--spec", str(spec), "--out", str(out),
                 "--stability", "5,10", "--reps", "2", "--gap", "10", "--batches", "3", "-q"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "VSS" in printed and "out of sample (30 independent samples" in printed
    results = json.loads((out / "results.json").read_text())
    assert results["saa_gap"]["batches"] == 3 and len(results["stability"]) == 4
    assert (out / "fig_stability.pdf").exists() and (out / "summary.md").exists()


def test_run_reports_failed_checks(farmer_dir, capsys):
    spec = farmer_dir / "uncertainty_distribution.yaml"
    # every variable first-stage: the stage_split check fails and the exit code says so
    spec.write_text(spec.read_text().replace("- acres_*", "- \"*\"").replace("n: 100", "n: 5")
                    .replace("n_test: 500", "n_test: 0"))
    code = main(["run", str(farmer_dir / "model.py"), "--spec", str(spec), "--out",
                 str(farmer_dir / "r"), "--no-figures", "-q"])
    assert code == 1
