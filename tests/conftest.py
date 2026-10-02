import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def farmer():
    pytest.importorskip("pulp")
    return _load(os.path.join(ROOT, "examples", "farmer", "model.py"), "farmer_model")


@pytest.fixture(scope="session")
def facility():
    pytest.importorskip("pyomo")
    return _load(os.path.join(ROOT, "examples", "facility_location", "model.py"), "facility_model")


@pytest.fixture(scope="session")
def facility_history():
    return os.path.join(ROOT, "examples", "facility_location", "demand_history.csv")


@pytest.fixture(scope="session")
def fixtures():
    return os.path.join(ROOT, "tests", "fixtures")


def farmer_scenarios(data):
    return [(1 / 3, {"yield": {c: f * y for c, y in data["yield"].items()}}) for f in (1.2, 1.0, 0.8)]


def pulp_var(prob, name, lb=None, ub=None, cat="Continuous"):
    """A PuLP variable on ``prob`` that works with PuLP 2/3 (``LpVariable``) and 4 (``add_variable``)."""
    import pulp

    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lb, ub, cat=cat)
    return pulp.LpVariable(name, lb, ub, cat=cat)
