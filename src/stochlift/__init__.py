"""StochLift: lift a deterministic optimization model to a two-stage stochastic program."""
from .adapters import UnsupportedModel, to_linear_model
from .checks import Check, all_passed
from .model import LinearModel, solve
from .scenarios import ScenarioSet, explicit, from_history
from .spec import Spec
from .study import Results, Study, lift

__version__ = "0.1.0"
__all__ = ["lift", "Study", "Results", "Spec", "ScenarioSet", "explicit", "from_history",
           "to_linear_model", "LinearModel", "solve", "Check", "all_passed", "UnsupportedModel"]
