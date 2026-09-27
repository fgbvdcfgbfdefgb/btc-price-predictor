"""Assemble notebooks/btc_price.ipynb from the tested source modules.

Everything except download_data.py is inlined, so the notebook is a single
self-contained artefact that stays byte-identical to the code that was
actually validated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "btc_price.ipynb"

SIBLINGS = ("evolution", "models", "system_detect", "features", "train",
            "download_data", "evaluate", "predict")


def clean(name: str, cut_at: tuple[str, ...] = ()) -> str:
    src = (SRC / f"{name}.py").read_text()

    # drop the module docstring (it becomes a markdown cell instead)
    src = re.sub(r'^"""(?:.|\n)*?"""\n', "", src, count=1)

    lines, out = src.splitlines(), []
    skipping_paren = False
    for ln in lines:
        if skipping_paren:                     # tail of a multi-line import
            if ")" in ln:
                skipping_paren = False
            continue
        if any(ln.startswith(c) for c in cut_at):
            break
        s = ln.strip()
        if s == "from __future__ import annotations":
            continue
        if s.startswith("sys.path.insert"):
            continue
        if re.match(rf"^\s*from ({'|'.join(SIBLINGS)}) import ", ln):
            # a parenthesised import spans several lines - drop them all
            if "(" in ln and ")" not in ln:
                skipping_paren = True
            continue
        if re.match(rf"^\s*import ({'|'.join(SIBLINGS)})\s*$", ln):
            continue
        out.append(ln)

    body = "\n".join(out)
    body = body.replace('Path(__file__).resolve().parents[1]', "ROOT")
    body = body.replace('Path(__file__).resolve().parent', '(ROOT / "src")')
    body = re.sub(r"\n{3,}", "\n\n\n", body).strip("\n")
    return body


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""
# btc_price — evolutionary BTC price predictor (1 min → 12 h)

End-to-end, self-contained notebook.

| Step | What happens |
|---|---|
| 0 | Setup and configuration |
| 1 | **Compute detection** — GPUs, **NVLink**, cores, RAM → execution plan |
| 2 | Feature engineering (57 causal features, 9 horizons) |
| 3 | Evolvable model zoo (MLP / GRU / TCN / Transformer) |
| 4 | Genetic operators (tournament, BLX crossover, mutation, elitism) |
| 5 | **Island-model evolutionary training** — GPU + CPU islands, resumable, 5 h |
| 6 | Held-out evaluation |
| 7 | Forecast 1 min → 12 h |
| 8 | Publish to GitHub |

The dataset is produced by `src/download_data.py` (one year of 1-minute
BTCUSDT OHLCV from Binance's public dumps) and is **not** part of this
notebook — run that script once, or pull `data/processed/btc_1m.parquet`
from the repo.

**Targets are log returns, not raw prices.** Predicting price directly
degenerates into echoing the last close and looks deceptively accurate.
A skill score below 1.0 means the model genuinely beats "assume no change".
"""))

cells.append(md("## 0 · Setup"))
cells.append(code('''
import os, sys, json, math, time, pickle, random, signal, hashlib, copy, warnings
import argparse, shutil, subprocess, mimetypes
import urllib.request, urllib.error
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repo root: works whether the notebook sits in notebooks/ or at the top level
ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
CACHE = ROOT / "data" / "processed" / "cache"
CKPT  = Path(os.environ.get("BTC_CKPT_DIR", ROOT / "checkpoints"))
LOGS  = ROOT / "logs"
DATA  = ROOT / "data" / "processed" / "btc_1m.parquet"
for d in (CACHE, CKPT, CKPT / "elites", LOGS):
    d.mkdir(parents=True, exist_ok=True)

_STOP = {"flag": False}
warnings.filterwarnings("ignore", category=UserWarning)

print("root   :", ROOT)
print("torch  :", torch.__version__)
print("dataset:", DATA, "-", "FOUND" if DATA.exists() else "MISSING (run src/download_data.py)")
'''))

# ---------------------------------------------------------------- 1. hardware
cells.append(md("""
## 1 · System detection — GPUs and NVLink

Decides how the evolution is parallelised:

| Detected | Strategy |
|---|---|
| ≥2 GPUs **with NVLink** | `ddp-nvlink` — NCCL with P2P over NVLink, one island per GPU |
| ≥2 GPUs, PCIe only | `ddp-pcie` — NCCL, P2P disabled |
| 1 GPU | concurrent training instances sharing the device |
| 0 GPUs | parallel CPU worker processes |

In every case the population is evaluated in parallel and the **best
individual seeds the next generation**, which is exactly the "run multiple
training instances and pick the best" fallback.
"""))
cells.append(code(clean("system_detect", cut_at=('if __name__',))))
cells.append(code('''
SYS = detect()
apply_nvlink_env(SYS)
print(render(SYS))
(LOGS / "system_info.json").write_text(json.dumps(SYS.to_dict(), indent=2))
'''))

# ---------------------------------------------------------------- 2. features
cells.append(md("""
## 2 · Features and targets

57 strictly causal features — momentum, realised volatility, moving-average
distance, channel position, Bollinger, RSI/MACD/ATR, candle micro-structure,
volume z-scores, taker-buy flow imbalance, VWAP distance, return
autocorrelation and cyclical time-of-day / day-of-week.

Targets: forward log return `log(close[t+h]/close[t])` for
**1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h**.

Splits are chronological with a **720-minute embargo** between them, so no
training row can peek at an outcome that overlaps validation or test.
"""))
cells.append(code(clean("features", cut_at=('if __name__',))))
cells.append(code('''
DATASET = prepare()          # ~10 s on the full year
print("\\nfeature matrix:", DATASET["X"].shape, " targets:", DATASET["Y"].shape)
pd.DataFrame({
    "horizon": DATASET["horizon_names"],
    "minutes": DATASET["horizon_steps"],
    "target std (log-ret)": DATASET["y_std"],
}).set_index("horizon")
'''))

# ---------------------------------------------------------------- 3. models
cells.append(md("""
## 3 · Model zoo

Four backbones the genetic algorithm can choose between. Each maps a window
`(batch, lookback, n_features)` to `(batch, 9)` standardised log returns.
"""))
cells.append(code(clean("models")))

# ---------------------------------------------------------------- 4. GA
cells.append(md("""
## 4 · Genetic operators

A genome encodes both **architecture** and **optimisation** hyper-parameters.
Reproduction = elitism + tournament selection + BLX-α crossover + per-gene
mutation + a trickle of random immigrants to keep diversity up.
"""))
cells.append(code(clean("evolution")))
cells.append(code('''
_demo_rng = random.Random(0)
pd.DataFrame([random_genome(_demo_rng) for _ in range(5)])
'''))

# ---------------------------------------------------------------- 5. trainer
cells.append(md("""
## 5 · Evolutionary trainer

Each individual is trained for `inner_steps` gradient steps, then scored on
validation by its **skill** = MSE ÷ MSE-of-predicting-zero, averaged over the
nine horizons (plus a small parsimony penalty). Lower is better; **< 1.0 beats
the naive baseline**.

Robustness built in:

* **Memory guard** — predicts each genome's activation footprint and shrinks
  its batch size; an over-budget genome is scored badly instead of OOM-killing
  the run.
* **Crash isolation** — a worker that dies is retried serially; one bad
  individual can never take down the evolution.
* **Resumability** — `checkpoints/state.json` is written atomically every
  generation with the population, RNG state and cumulative elapsed time.
  Re-running resumes exactly where it stopped, and SIGINT/SIGTERM checkpoints
  before exiting.
* **Wall-clock budget** — `--max-hours 5` stops cleanly and saves the model.
"""))
cells.append(code(clean("train", cut_at=('def cli(', 'if __name__'))))

cells.append(md("""
### Launch

`max_hours=5` is the requested budget. Re-run this cell after an interruption
and it picks up from the last completed generation — set `fresh=True` to start
over.

> **Multi-GPU:** run it from the shell instead, so `torchrun` can spawn one
> rank per GPU and NCCL can use NVLink:
> ```bash
> ./scripts/run_multigpu.sh                 # population-parallel, 1 island/GPU
> ./scripts/run_multigpu.sh --parallel ddp  # all GPUs on one individual
> ```
> Notebook kernels can't be `spawn`-pickled, so in-notebook GPU runs use a
> single worker.
"""))
cells.append(code('''
args = argparse.Namespace(
    # --- budget -------------------------------------------------------
    generations=1000,
    pop_size=24,
    inner_steps=2000,         # more training per individual
    max_hours=5.0,            # <- the 5-hour budget; carried across resumes
    # --- evolution rate: deliberately LOW ------------------------------
    elite_frac=0.30,          # strong elitism
    mutate_rate=0.12,         # gentle mutation
    mutate_sigma=0.15,
    immigrant_frac=0.04,      # little random churn
    parsimony=0.002,
    # --- hardware: use everything --------------------------------------
    islands_per_gpu=4,        # several islands per GPU when VRAM allows
    cpu_islands=-1,           # -1 = auto from spare cores, 0 = disable
    no_cpu_islands=False,
    mem_budget_mb=0,          # 0 = auto from detected RAM/VRAM
    parallel="population",
    calibrate=True,
    gen_timeout_mult=2.5,
    # --- misc -----------------------------------------------------------
    subsample=None,           # e.g. 120_000 for a fast smoke test
    seed=1337,
    lamarckian=True,
    fresh=False,              # True = ignore any existing checkpoint
    rebuild_cache=False,
)
run(args)
'''))

cells.append(md("### Evolution history"))
cells.append(code('''
_st = json.loads((CKPT / "state.json").read_text())
hist = pd.DataFrame(_st["history"])
print(f"generations completed : {_st['generation']}")
print(f"wall-clock spent      : {_st['elapsed_seconds']/3600:.3f} h")
print(f"best fitness          : {_st['best']['fitness']:.5f}")

if len(hist):
    ax = hist.plot(x="generation", y=["best_fitness", "mean_fitness"],
                   figsize=(9, 4), grid=True,
                   title="Evolution progress (lower = better)")
    ax.axhline(1.0, color="crimson", ls="--", lw=1,
               label="zero-return baseline")
    ax.set_ylabel("skill  (MSE / baseline MSE)")
    ax.legend()
hist.tail(10)
'''))

# ---------------------------------------------------------------- 6. eval
cells.append(md("""
## 6 · Held-out evaluation

The test window is the most recent stretch of the year and was never touched
during evolution.

* **skill / R²-vs-0** — beat "assume no change"?
* **dir_acc** — directional accuracy; the number that actually matters
* **IC** — Spearman rank correlation of prediction vs outcome
"""))
cells.append(code(clean("evaluate", cut_at=('if __name__',))
                  .replace("def main(", "def evaluate_split(")))
cells.append(code('''
_res = evaluate_split("test")
pd.DataFrame(_res["rows"]).set_index("horizon")
'''))

# ---------------------------------------------------------------- 7. predict
cells.append(md("""
## 7 · Forecast

Loads the evolved champion and projects BTC forward from the newest bar.
Bands are ±1 MAE-derived sigma from the held-out evaluation.
"""))
cells.append(code(clean("predict", cut_at=('def main(', 'if __name__'))))
cells.append(code('''
fc = forecast()
print(f"as of {fc['as_of']}   spot ${fc['spot']:,.2f}   model: evolved {fc['backbone']}")
pd.DataFrame(fc["forecasts"])[
    ["horizon", "valid_at", "pred_return_pct", "pred_price", "direction"]
].set_index("horizon")
'''))

# ---------------------------------------------------------------- 8. github
cells.append(md("""
## 8 · Publish to GitHub

The token is read from the `GITHUB_TOKEN` environment variable. **Never paste
a token into a notebook cell** — the value gets saved inside the `.ipynb` and
travels with every copy of it.

```bash
export GITHUB_TOKEN=github_pat_xxxxxxxx
python src/upload_to_github.py --repo btc-price-predictor --mode release
```
"""))
cells.append(code('''
if os.environ.get("GITHUB_TOKEN"):
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "upload_to_github.py"),
         "--repo", "btc-price-predictor", "--mode", "release"],
        cwd=ROOT, text=True, capture_output=True)
    print(r.stdout or "", r.stderr or "")
else:
    print("GITHUB_TOKEN not set - skipping upload.")
    print("Run:  export GITHUB_TOKEN=...  in your shell, restart the kernel,")
    print("then re-run this cell. Do not hard-code the token here.")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
print(f"  cells    : {len(cells)}")
print(f"  code      : {sum(1 for c in cells if c['cell_type']=='code')}")
print(f"  markdown  : {sum(1 for c in cells if c['cell_type']=='markdown')}")
print(f"  size      : {OUT.stat().st_size/1024:.0f} KB")
