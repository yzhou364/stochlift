# Command line

`stochlift` has three commands. A model is `model.py` (the function `build_model` and the dictionary `DATA`), `model.py:function`, or `package.module:function`; `--data` names another variable in the file or a `.json` / `.yaml` file.

## stochlift init

```text
usage: stochlift init [-h] [--data DATA] [--history HISTORY] [-o OUTPUT] [--force] [--llm LLM]
                      model

positional arguments:
  model                model.py, model.py:function or package.module:function (default function:
                       build_model)

options:
  -h, --help           show this help message and exit
  --data DATA          variable in the model file (default DATA) or a .json/.yaml file
  --history HISTORY    CSV of past observations of the uncertain data
  -o, --output OUTPUT
  --force              overwrite an existing file
  --llm LLM            let a language model propose the spec, e.g. anthropic:<model id>
```

## stochlift run

```text
usage: stochlift run [-h] [--data DATA] [--history HISTORY] [--spec SPEC] [--out OUT]
                     [--stability SIZES] [--reps REPS] [--gap N] [--batches BATCHES]
                     [--frontier WEIGHTS] [--alpha ALPHA] [--solver {highs,gurobi}] [--jobs JOBS]
                     [--mip-gap MIP_GAP] [--time-limit TIME_LIMIT] [--no-figures]
                     [--no-out-of-sample] [-q]
                     model

positional arguments:
  model                 model.py, model.py:function or package.module:function (default function:
                        build_model)

options:
  -h, --help            show this help message and exit
  --data DATA           variable in the model file (default DATA) or a .json/.yaml file
  --history HISTORY     CSV of past observations of the uncertain data
  --spec SPEC           the reviewed spec (default uncertainty.yaml)
  --out OUT             report folder (default report)
  --stability SIZES     comma-separated scenario counts, e.g. 5,10,20,40
  --reps REPS           samples per size for --stability
  --gap N               estimate the optimality gap with batches of N
  --batches BATCHES
  --frontier WEIGHTS    mean-CVaR trade-off for these CVaR weights, e.g. 0,0.25,0.5,0.75,1
  --alpha ALPHA         CVaR level for --frontier (default: the spec's, or 0.9)
  --solver {highs,gurobi}
                        solver for every model (gurobi needs gurobipy and a license)
  --jobs JOBS           parallel scenario solves (default: CPU count, at most 8)
  --mip-gap MIP_GAP
  --time-limit TIME_LIMIT
                        seconds per solve
  --no-figures
  --no-out-of-sample
  -q, --quiet           do not print the review
```

## stochlift export

```text
usage: stochlift export [-h] [--data DATA] [--history HISTORY] [--spec SPEC] [-o OUTPUT] model

positional arguments:
  model                model.py, model.py:function or package.module:function (default function:
                       build_model)

options:
  -h, --help           show this help message and exit
  --data DATA          variable in the model file (default DATA) or a .json/.yaml file
  --history HISTORY    CSV of past observations of the uncertain data
  --spec SPEC
  -o, --output OUTPUT
```
