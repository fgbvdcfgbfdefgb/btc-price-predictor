# BTC price predictor — evolutionary, multi-horizon (1 min → 12 h)

One year of 1-minute BTCUSDT candles, 57 causal features, and an
**island-model genetic algorithm** that evolves the architecture *and* the
optimiser hyper-parameters of a multi-horizon forecaster.

Training is **hardware-saturating** (every GPU, several islands per GPU, plus
spare CPU cores), **resumable**, and bounded by a **wall-clock budget**.

```bash
./setup.sh cu124 && ./scripts/run_multigpu.sh
```

---

## Using every inch of the machine

`src/system_detect.py` inventories the box and emits a concrete
**ExecutionPlan** before a single gradient is computed.

| Resource | What is detected | How it gets used |
|---|---|---|
| GPUs | count, name, free/total VRAM, SM count, compute capability | one island per GPU, **or several per GPU** when VRAM allows |
| NVLink | `nvidia-smi nvlink --status` + `topo -m`, active links, GB/s, peer pairs, full-mesh | `NCCL_P2P_LEVEL=NVL`, `NCCL_NVLS_ENABLE=1`, P2P on |
| P2P | `torch.cuda.can_device_access_peer` matrix | confirms NVLink is really usable, not just cabled |
| Tensor cores | compute capability ≥ 8.0 | **TF32 matmul + bf16 autocast**; fp16 + GradScaler below |
| CPU | physical vs logical cores, NUMA nodes, AVX-512 / AMX / VNNI | spare cores become **extra islands next to the GPUs** |
| RAM | total, available, swap | dataset held resident; per-island activation budgets |
| Disk | free space | archive cache |

**Dataset placement** is chosen automatically, best first:

1. **VRAM** — the whole feature matrix is parked on the GPU, so batches are a
   pure on-device gather and **never cross PCIe**. The dataset is only ~160 MB,
   so this is almost always available and it is the single biggest speed-up.
2. **System RAM** — one resident copy, shared **copy-on-write** by every forked
   CPU island (N islands, one copy).
3. **mmap** — page-cache fallback when RAM is tight.

Two multi-GPU modes:

```bash
./scripts/run_multigpu.sh                 # population-parallel (default)
./scripts/run_multigpu.sh --mode ddp      # all GPUs cooperate on one individual
```

`population` gives the better GA throughput — each rank trains whole
individuals and NVLink carries only the fitness gather and elite migration.
`ddp` is for models too large for a single card; gradients all-reduce over
NVLink every step.

Under `torchrun`, **each rank** runs `islands_per_gpu` islands on its own GPU
*plus* its share of the spare CPU cores, so nothing sits idle.

---

## Deliberately low evolution rate

Compute is spent **refining** good solutions rather than churning the
population:

| Knob | Default | Effect |
|---|---|---|
| `--elite-frac` | **0.30** | top 30% survive verbatim |
| `--mutate-rate` | **0.12** | only ~1 gene in 8 is touched per child |
| `--mutate-sigma` | **0.15** | small jitter on continuous genes |
| `--immigrant-frac` | **0.04** | barely any random restarts |
| `--inner-steps` | **2000** | each individual is trained properly before judging |

Plus **Lamarckian warm-starting**: a child whose genome hash matches a cached
elite inherits its weights, so good solutions keep training across generations
instead of restarting from scratch.

---

## Results, stated honestly

Held-out test set — the most recent ~8 weeks, never touched during evolution
(**78,516 samples**, full year of data, champion = evolved GRU, 74k params).
This is **2 generations on 2 CPU cores**, which is a smoke test, not a result:

| horizon | skill ↓ | R² vs zero | dir. acc | IC | MAE (bps) |
|---|---|---|---|---|---|
| 1m  | 1.0026 | −0.0026 | 50.57% | 0.014 | 3.1 |
| 5m  | 1.0044 | −0.0044 | 51.52% | 0.026 | 7.1 |
| 15m | 1.0051 | −0.0051 | 51.59% | 0.042 | 12.2 |
| 30m | 1.0052 | −0.0052 | 51.13% | 0.044 | 17.3 |
| 1h  | 1.0088 | −0.0088 | 51.32% | 0.024 | 24.2 |
| 2h  | 1.0057 | −0.0057 | 50.03% | 0.008 | 34.6 |
| 4h  | 1.0127 | −0.0127 | 48.61% | −0.033 | 51.1 |
| 8h  | 1.0198 | −0.0198 | 46.94% | −0.046 | 75.3 |
| 12h | 1.0190 | −0.0190 | 46.85% | −0.001 | 94.2 |
| **mean** | **1.0092** | **−0.0092** | **49.84%** | **0.009** | — |

`skill` is MSE divided by the MSE of simply predicting "no change"; **below 1.0
means the model adds information**. It does not, yet. Validation reached 0.9993
while test came out at 1.0092 — that gap *is* the GA overfitting the validation
split through repeated selection, and it is the thing to watch as you scale up.

A caution worth repeating: on a 17k-sample subsample the same pipeline reported
53% mean directional accuracy and 59% at 12h. On the full 78k test set that
evaporated to 49.8%. **1-minute bars are massively autocorrelated, so a short
window contains far fewer independent observations than its row count suggests.**
Always read the full-year numbers.

Nothing here is trading advice. Short-horizon crypto is close to efficient; a
real edge is small, fragile, and easily eaten by fees and slippage.

---

## Quick start

```bash
./setup.sh              # CPU
./setup.sh cu124        # CUDA 12.4 — needed for multi-GPU / NVLink

python3 src/download_data.py    # 1 year of 1m candles (525,600 rows)
python3 src/system_detect.py    # full inventory + execution plan
./scripts/run_multigpu.sh       # 5-hour evolutionary run
```

### Launcher

```
./scripts/run_multigpu.sh [options] [-- extra args for train.py]

  --hours N            wall-clock budget           (default 5)
  --mode population    one island per GPU          (default)
  --mode ddp           all GPUs on one individual
  --pop N              population size             (default: 4 x islands)
  --steps N            gradient steps / individual (default 2000)
  --islands-per-gpu N  concurrent instances per GPU, VRAM permitting (4)
  --cpu-islands N      -1 auto from spare cores, 0 to disable
  --no-cpu-islands     GPUs only
  --background         detach with nohup and tail the log
  --fresh              ignore the checkpoint and start over
  --no-eval            skip the evaluation + forecast at the end
```

It preflights dependencies, downloads the dataset if missing, prints the full
inventory, sets the right NCCL/NVLink environment, sizes the population to the
hardware, resumes automatically, logs to `logs/train-<timestamp>.log`, and runs
the held-out evaluation plus a forecast when it finishes.

---

## The data

`src/download_data.py` pulls monthly + daily archives from
**`data.binance.vision`** (Binance's public S3 dumps — no API key, and not
geo-blocked the way `api.binance.com` is in many regions).

| | |
|---|---|
| rows | 525,600 |
| range | 2025-09-26 → 2026-09-25 (UTC) |
| gaps | **0 missing minutes** |
| price range | $57,881.98 – $126,114.50 |
| columns | `ts, open, high, low, close, volume, quote_volume, trades, taker_buy_base` |

Shipped as `data/processed/btc_1m.parquet` (24 MB, zstd) and `btc_1m.csv.gz`
(18 MB), and attached to the `dataset-v1` release.

> Binance switched kline timestamps from milliseconds to microseconds partway
> through 2025; the loader normalises both.

---

## How the evolution works

```
population of genomes
        |
        v
 [GPU0 isl 0..k] [GPU1 isl 0..k] ... [CPU isl 0..m]   <- all concurrent
   train + score    train + score       train + score
        |                |                    |
        +-------- gather fitnesses -----------+        <- NCCL over NVLink
                         |
        elitism 30% + tournament + BLX-a crossover
            + 12% mutation + 4% immigrants
                         |
                         v
                 next generation          -> checkpoint every generation
```

**Genome** — backbone (`mlp` / `gru` / `tcn` / `attn`), lookback, hidden width,
depth, dropout, activation, learning rate, weight decay, batch size, Huber
delta, feature dropout, gradient clip, per-horizon loss-weight exponent.

**Fitness** — mean over the nine horizons of `MSE / MSE(predict 0)`, plus a
parsimony penalty per million parameters. Lower is better.

**Features** — 57 strictly causal signals: momentum over 13 lookbacks, realised
volatility, MA distance, channel position, Bollinger, RSI/MACD/ATR, candle
micro-structure, volume z-scores, taker-buy flow imbalance, VWAP distance,
return autocorrelation, cyclical time-of-day / day-of-week.

**Splits** — chronological with a **720-minute embargo**, so no training row can
see an outcome overlapping validation or test.

---

## Resumability and robustness

Every generation atomically writes `checkpoints/state.json` with the
population, RNG state, history, champion and **cumulative elapsed time**, so
the 5-hour budget survives restarts.

```bash
./scripts/run_multigpu.sh --hours 5    # run
# Ctrl-C, crash, reboot, spot eviction...
./scripts/run_multigpu.sh --hours 5    # resumes at the next generation
./scripts/run_multigpu.sh --fresh      # start over
```

SIGINT/SIGTERM finish the current generation, checkpoint, then exit.

Hardened against the failure modes that actually broke earlier runs here:

* **Predictive memory guard** — estimates each genome's activation footprint
  and shrinks batch size, then lookback, to fit the island it landed on.
  Anything still over budget is scored badly instead of OOM-killing the run.
* **Fork-aware RAM model** — forked islands share PyTorch's pages copy-on-write,
  so only their marginal heap is charged against the budget.
* **Crash isolation** — a worker killed by the OS is retried serially.
* **Straggler deadline** — a generation is capped at 2.5× the recent
  generation time, so a slow CPU island can never stall the GPUs. Truncated
  individuals are flagged and cannot be crowned champion.
* **Safe start method** — CPU islands are `spawn`ed rather than forked when a
  CUDA context is live.

---

## Troubleshooting

### `OSError: [Errno 12] Cannot allocate memory` while RAM is mostly free

This error is almost never about RAM. It means a *process-level* resource ran
out. Run the diagnostic first - it checks each candidate and names the culprit:

```bash
python3 scripts/diagnose.py
```

The usual causes, in order of how often they bite:

| cause | check | fix |
|---|---|---|
| **`/dev/shm` too small** - the classic container default of 64 MB | `df -h /dev/shm` | `docker run --shm-size=16g ...`, or in k8s mount an `emptyDir{medium: Memory}` at `/dev/shm` |
| **disk full** - checkpoints and the feature cache cannot be written | `df -h .` | free space; the launcher now refuses to start below 2 GB |
| **`vm.max_map_count` too low** - each CUDA process makes tens of thousands of mappings | `cat /proc/sys/vm/max_map_count` | `sysctl -w vm.max_map_count=262144` |
| **`vm.overcommit_memory=2`** - forbids over-committing, so spawn fails | `cat /proc/sys/vm/overcommit_memory` | `sysctl -w vm.overcommit_memory=0` |
| **`RLIMIT_AS` set** - CUDA reserves tens of GB of *virtual* address space | `ulimit -v` | `ulimit -v unlimited` |

The trainer now defends itself on all of these:

* `run_multigpu.sh` and `train.py` **refuse to start** on a full disk or a
  `/dev/shm` under 256 MB, instead of failing 5 hours later.
* The planner **caps islands by `/dev/shm` size** (~256 MB per worker) and by
  the host RAM a CUDA context costs, so it will not spawn 23 workers into a
  64 MB `/dev/shm`.
* `RLIMIT_AS` is **never set when CUDA is present**. It is only used as a cheap
  OOM guard on small CPU-only boxes.
* Training **aborts after 2 consecutive generations in which every individual
  failed** (`--max-dead-generations`), prints the dominant error and a
  diagnosis, and leaves the last good checkpoint untouched - rather than
  burning the whole time budget producing `best 1000000.00000`.

Conservative configuration that almost always runs:

```bash
./scripts/run_multigpu.sh --islands-per-gpu 1 --no-cpu-islands
```

### A run reports a suspiciously good score

Check `n_test` in `data/processed/cache/meta.json`. A cache built with
`--subsample` has a much smaller test set and its metrics are not comparable to
a full-year run. The cache now rebuilds itself automatically when the
`--subsample` value changes.

## Layout

```
src/download_data.py    Binance public dumps -> clean parquet
src/system_detect.py    full compute inventory + ExecutionPlan + calibration
src/features.py         57 causal features, 9 horizons, embargoed splits
src/models.py           MLP / GRU / TCN / Transformer backbones
src/evolution.py        genome space + genetic operators (low default rates)
src/train.py            island-model GA, DDP, checkpointing, budget
src/evaluate.py         held-out metrics
src/predict.py          1m -> 12h forecast with uncertainty bands
src/upload_to_github.py publish (token from $GITHUB_TOKEN only)
notebooks/btc_price.ipynb   everything except the downloader, inlined
scripts/run_multigpu.sh     complete launcher
scripts/publish.py          normal incremental publish (always fast-forward)
scripts/wipe_and_republish.py  deliberate nuke-and-replace (rewrites history)
scripts/diagnose.py         explains ENOMEM / environment failures
scripts/make_notebook.py    regenerates the notebook from src/
scripts/exec_notebook_check.py  runs every notebook cell as a test
```

Both distributed modes are verified on a 2-rank **gloo** process group
(`all_gather_object`, `broadcast_object_list`, `DistributedDataParallel`,
sharded validation via `all_reduce`) — the same code paths NCCL drives on real
GPUs.

---

## Security

Every publish path reads `GITHUB_TOKEN` from the environment and passes it
through `GIT_ASKPASS`, so it never lands in `.git/config`, the notebook, or any
committed file.

```bash
export GITHUB_TOKEN=github_pat_xxxx
python3 scripts/publish.py -m "what changed"      # normal path
```

### Publishing: use `scripts/publish.py`

`scripts/publish.py` fetches the remote, points `HEAD` at `origin/main` with
`git reset --mixed` (which leaves the working tree untouched) and commits on
top. The push is always a **fast-forward**, so anyone with the repo cloned just
runs `git pull` and gets it cleanly.

`scripts/wipe_and_republish.py` is the opposite: it force-pushes a fresh orphan
root commit, discarding all history. It exists for a deliberate
"delete everything and start over", and it will **break every existing clone** -
a subsequent `git pull` fails with *refusing to merge unrelated histories*, and
any open GitHub web-editor tab will report an unresolved conflict against the
rewritten file. If that happens, resync the clone with:

```bash
git fetch origin && git reset --hard origin/main
```

Prefer `publish.py` unless you specifically want history erased.

Never paste a token into a notebook cell — the value is saved inside the
`.ipynb` and travels with every copy. If a token has ever appeared in a chat, a
screenshot, or a commit, **revoke it**.

---

## Licence

MIT for the code. Market data belongs to Binance and is redistributed here for
research use.
