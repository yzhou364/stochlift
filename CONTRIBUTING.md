# Contributing

Bug reports, models and pull requests are welcome.

## The most useful contribution: a model

If you lifted a model of your own, please open an issue with the "I lifted my model" template,
even if everything worked. Knowing which domains, libraries and model sizes people use decides
what gets built next. Models that can be shared (or simplified so they can be) may become
examples or part of the benchmark.

## Development setup

```bash
git clone https://github.com/yzhou364/stochlift.git && cd stochlift
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Optional libraries are tested when installed: `pip install gurobipy ortools`.

## Rules for changes

- Every number StochLift reports must be checkable. A new feature comes with a test against a
  value that does not come from StochLift itself: a textbook result, a closed form, brute force,
  or an independent solve.
- A bug fix comes with a regression test in `tests/test_regressions.py`.
- Keep the user contract unchanged: `build_model(data)` returns a model, and nothing else about
  the user's code is assumed.
- Tests must pass with PuLP 3 and PuLP 4, and on Windows, macOS and Linux (CI runs all of them).
