"""The uncertainty specification: the one file a person reviews and signs off."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class Spec:
    first_stage: list                      # glob patterns on variable names
    uncertain: list = field(default_factory=list)   # data keys or dotted leaf names
    columns: dict = field(default_factory=dict)     # leaf name -> history column
    scenarios: dict = field(default_factory=lambda: {"method": "empirical"})
    holdout: float = 0.0                   # share of history (last rows) kept for testing
    risk: dict = field(default_factory=dict)        # {alpha, weight}: mean-CVaR objective
    notes: str = ""

    def __post_init__(self):
        if isinstance(self.first_stage, str):
            self.first_stage = [self.first_stage]
        if isinstance(self.uncertain, str):
            self.uncertain = [self.uncertain]
        self.first_stage = [str(p) for p in self.first_stage]
        self.uncertain = [str(k) for k in self.uncertain]
        if not self.first_stage:
            raise ValueError("spec needs at least one first_stage pattern")
        if not 0.0 <= float(self.holdout) < 1.0:
            raise ValueError("holdout must be in [0, 1)")
        self.scenarios = dict(self.scenarios or {"method": "empirical"})
        self.scenarios.setdefault("method", "empirical")
        self.risk = dict(self.risk or {})
        from .risk import Risk

        Risk.from_spec(self.risk)          # validate early

    def is_first_stage(self, name: str) -> bool:
        """Match ``name`` against the patterns. Only ``*`` and ``?`` are wildcards,
        so Pyomo-style names such as ``open[S1]`` can be matched by ``open[*]``."""
        if getattr(self, "_compiled_for", None) != tuple(self.first_stage):
            self._compiled = [re.compile("".join(".*" if ch == "*" else "." if ch == "?" else re.escape(ch)
                                                 for ch in pat) + r"\Z", re.S) for pat in self.first_stage]
            self._compiled_for = tuple(self.first_stage)
        return any(rx.match(name) for rx in self._compiled)

    def to_dict(self) -> dict:
        d = asdict(self)
        keys = ["first_stage", "uncertain", "columns", "scenarios", "holdout", "risk", "notes"]
        if not d["risk"]:
            keys.remove("risk")
        return {k: d[k] for k in keys}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        known = {"first_stage", "uncertain", "columns", "scenarios", "holdout", "risk", "notes"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown spec fields: {sorted(unknown)}; allowed: {sorted(known)}")
        if "first_stage" not in d:
            raise ValueError("spec is missing 'first_stage'")
        return cls(**d)

    @classmethod
    def load(cls, source) -> "Spec":
        if isinstance(source, Spec):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)
        if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
            with open(source, encoding="utf-8") as f:
                return cls.from_dict(yaml.safe_load(f))
        if isinstance(source, str):
            return cls.from_dict(yaml.safe_load(source))
        raise TypeError("spec must be a Spec, a dict, a YAML string or a path")
