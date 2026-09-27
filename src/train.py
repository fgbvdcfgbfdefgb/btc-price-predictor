"""
Island-model evolutionary trainer for multi-horizon BTC return forecasting.

Uses every resource the box has
-------------------------------
  * one island per GPU, or SEVERAL per GPU when VRAM allows
  * spare physical CPU cores run extra islands alongside the GPUs
  * the dataset is parked in VRAM when it fits (batches never cross PCIe),
    otherwise in system RAM shared copy-on-write across forked islands,
    otherwise memory-mapped
  * TF32 matmul + bf16 autocast on Ampere and newer, fp16 below that
  * NCCL over NVLink for the fitness gather, elite migration and DDP

Parallelism
-----------
  torchrun, --parallel population :  one rank per GPU, each rank trains whole
                                     individuals; NVLink carries the gather.
  torchrun, --parallel ddp        :  all ranks cooperate on ONE individual,
                                     gradients all-reduce over NVLink.
  no torchrun                     :  heterogeneous local pool of GPU islands
                                     + CPU islands, evaluated concurrently.

Resumability
------------
  Every generation writes checkpoints/state.json + best_model.pt atomically.
  SIGINT/SIGTERM checkpoint and exit cleanly. Re-running the same command
  resumes at the next generation with RNG state restored and the wall-clock
  budget carried over.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolution import (ELITE_FRAC, IMMIGRANT_FRAC, MUTATE_RATE,  # noqa: E402
                       MUTATE_SIGMA, cost_proxy, genome_key,
                       next_generation, random_genome)
from models import build_model, count_params                      # noqa: E402
from system_detect import (apply_nvlink_env, apply_perf_env,      # noqa: E402
                           calibrate, detect, render)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed" / "cache"
CKPT = Path(os.environ.get("BTC_CKPT_DIR", ROOT / "checkpoints"))
LOGS = ROOT / "logs"
_STOP = {"flag": False}
_DATA: dict | None = None          # process-global; forked islands inherit it


# --------------------------------------------------------------------------- #
# data cache                                                                   #
# --------------------------------------------------------------------------- #
def build_cache(subsample: int | None = None, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_p = CACHE / "meta.json"
    if meta_p.exists() and not force:
        cached = json.loads(meta_p.read_text())
        # A cache built with a different --subsample silently changes the size
        # of the test set, which makes every reported metric incomparable.
        # Rebuild instead of quietly reusing it.
        if cached.get("subsample") == subsample:
            return cached
        print(f"  [cache] built with subsample={cached.get('subsample')} but "
              f"subsample={subsample} was requested -> rebuilding", flush=True)

    from features import prepare
    d = prepare(subsample=subsample)
    np.save(CACHE / "X.npy", d["X"])
    np.save(CACHE / "Y.npy", d["Y"])
    for k in ("train_idx", "val_idx", "test_idx"):
        np.save(CACHE / f"{k}.npy", d[k].astype(np.int64))
    np.save(CACHE / "y_std.npy", d["y_std"])
    np.save(CACHE / "close.npy", d["close"])
    with open(CACHE / "scaler.pkl", "wb") as f:
        pickle.dump(d["scaler"], f)
    meta = {
        "n_features": int(d["X"].shape[1]),
        "n_horizons": int(d["Y"].shape[1]),
        "feature_names": d["feature_names"],
        "horizon_names": d["horizon_names"],
        "horizon_steps": d["horizon_steps"],
        "n_rows": int(d["X"].shape[0]),
        "n_train": int(len(d["train_idx"])),
        "n_val": int(len(d["val_idx"])),
        "n_test": int(len(d["test_idx"])),
        "subsample": subsample,
        "bytes": int(d["X"].nbytes + d["Y"].nbytes),
    }
    meta_p.write_text(json.dumps(meta, indent=2))
    return meta


def dataset_mb() -> float:
    try:
        m = json.loads((CACHE / "meta.json").read_text())
        if m.get("bytes"):
            return m["bytes"] / 1e6
        return (m["n_rows"] * (m["n_features"] + m["n_horizons"]) * 4) / 1e6
    except Exception:
        return 160.0


def load_cache(placement: str = "mmap", reuse: bool = True) -> dict:
    """placement: 'ram' loads into memory, anything else memory-maps."""
    global _DATA
    if reuse and _DATA is not None:
        return _DATA
    # The derived cache is regenerable and therefore not committed. On a fresh
    # clone it simply will not be there - build it rather than exploding.
    if not (CACHE / "meta.json").exists():
        print("  [cache] not present - building it now (one-off, ~20 s)",
              flush=True)
        build_cache()
    m = None if placement == "ram" else "r"
    d = {
        "X": np.load(CACHE / "X.npy", mmap_mode=m),
        "Y": np.load(CACHE / "Y.npy", mmap_mode=m),
        "train_idx": np.load(CACHE / "train_idx.npy"),
        "val_idx": np.load(CACHE / "val_idx.npy"),
        "test_idx": np.load(CACHE / "test_idx.npy"),
        "y_std": np.load(CACHE / "y_std.npy"),
        "meta": json.loads((CACHE / "meta.json").read_text()),
    }
    if reuse:
        _DATA = d
    return d


class WindowSource:
    """Serves (B, L, F) windows, from GPU-resident tensors when possible."""

    def __init__(self, X, Y, device: torch.device, on_device: bool):
        self.device, self.on_device = device, on_device
        if on_device:
            self.X = torch.as_tensor(np.ascontiguousarray(X)).to(device)
            self.Y = torch.as_tensor(np.ascontiguousarray(Y)).to(device)
        else:
            self.X, self.Y = X, Y

    def batch(self, idx: np.ndarray, offsets, ystd):
        if self.on_device:
            i = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            xb = self.X[i[:, None] + offsets]
            yb = self.Y[i] / ystd
            return xb, yb
        win = idx[:, None] + offsets
        xb = torch.from_numpy(np.ascontiguousarray(self.X[win])).to(
            self.device, non_blocking=True)
        yb = torch.from_numpy(np.ascontiguousarray(self.Y[idx])).to(
            self.device, non_blocking=True)
        return xb, yb / ystd


# --------------------------------------------------------------------------- #
# memory guard                                                                 #
# --------------------------------------------------------------------------- #
def estimate_peak_mb(g: dict, n_feat: int) -> float:
    B, L, H, D = g["batch_size"], g["lookback"], g["hidden"], g["depth"]
    inp = B * L * n_feat * 4
    bb = g["backbone"]
    if bb == "mlp":
        act = B * H * 4 * D * 4 + B * L * n_feat * 4 * 2
    elif bb in ("gru", "tcn"):
        act = B * L * H * 4 * D * 6
    else:
        heads = max(1, min(8, H // 32))
        act = B * L * H * 4 * D * 8 + B * heads * L * L * 4 * D * 3
    return (inp + act) * 3.0 / 1e6


LOOKBACKS = [16, 32, 64, 96, 128]


def fit_to_memory(g: dict, n_feat: int, budget_mb: float) -> tuple[dict, bool]:
    """Adapt the genome to the island it landed on: batch first, then window."""
    g = dict(g)
    while estimate_peak_mb(g, n_feat) > budget_mb and g["batch_size"] > 32:
        g["batch_size"] //= 2
    while estimate_peak_mb(g, n_feat) > budget_mb and g["lookback"] > LOOKBACKS[0]:
        smaller = [l for l in LOOKBACKS if l < g["lookback"]]
        g["lookback"] = smaller[-1]
    return g, estimate_peak_mb(g, n_feat) <= budget_mb


def _limit_address_space(mb: int) -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = int(mb * 1024 * 1024)
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    except Exception:
        pass


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dt = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dt), precision == "fp16"
    import contextlib
    return contextlib.nullcontext(), False


# --------------------------------------------------------------------------- #
# distributed (torchrun) support                                               #
# --------------------------------------------------------------------------- #
def dist_init():
    import torch.distributed as dist
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local % torch.cuda.device_count())
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return True, dist.get_rank(), dist.get_world_size(), local


def ddp_evaluate(task: dict, rank: int, world: int) -> dict:
    """All ranks cooperate on ONE individual; gradients all-reduce over NVLink."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    g = task["genome"]
    device = torch.device(task["device"])
    D = load_cache(task.get("placement", "mmap"))
    tr, va = D["train_idx"], D["val_idx"]
    n_feat, n_out = D["meta"]["n_features"], D["meta"]["n_horizons"]
    ystd = torch.from_numpy(D["y_std"]).to(device)

    g, _ = fit_to_memory(g, n_feat, task.get("mem_budget_mb", 400))
    L = g["lookback"]
    tr, va = tr[tr >= L - 1], va[va >= L - 1]
    src = WindowSource(D["X"], D["Y"], device, task.get("on_device", False))
    offsets = torch.arange(-L + 1, 1, device=device) if src.on_device \
        else np.arange(-L + 1, 1, dtype=np.int64)

    torch.manual_seed(task["seed"])
    base = build_model(g, n_feat, n_out).to(device)
    n_par = count_params(base)
    model = DDP(base, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=g["lr"],
                            weight_decay=g["weight_decay"])
    steps = task["inner_steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=g["lr"],
                                                total_steps=steps, pct_start=0.25)
    hs = np.asarray(D["meta"]["horizon_steps"], dtype=np.float64)
    w = hs ** g["horizon_w"]
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    lossfn = nn.HuberLoss(delta=g["huber_delta"], reduction="none")
    ac, need_scaler = _autocast(device, task.get("precision", "fp32"))
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
    shard = max(32, g["batch_size"] // world)
    rng = np.random.default_rng(task["seed"] + rank)

    t0 = time.time()
    model.train()
    for _ in range(steps):
        bidx = tr[rng.integers(0, len(tr), size=shard)]
        xb, yb = src.batch(bidx, offsets, ystd)
        with ac:
            loss = (lossfn(model(xb), yb) * w).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), g["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()

    model.eval()
    part = va[rank::world]
    se = torch.zeros(n_out, device=device)
    var = torch.zeros(n_out, device=device)
    hit = torch.zeros(n_out, device=device)
    cnt = torch.zeros(1, device=device)
    with torch.no_grad():
        vb = max(256, min(8192, g["batch_size"] * 4))
        for i in range(0, len(part), vb):
            b = part[i:i + vb]
            xb, yb = src.batch(b, offsets, ystd)
            with ac:
                p = base(xb)
            p = p.float()
            se += ((p - yb) ** 2).sum(0)
            var += (yb ** 2).sum(0)
            hit += ((p.sign() == yb.sign()) & (yb != 0)).sum(0)
            cnt += len(b)
    for t in (se, var, hit, cnt):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    skill = (se / var.clamp_min(1e-12)).cpu().numpy()
    dacc = (hit / cnt.clamp_min(1)).cpu().numpy()

    state_path = None
    if rank == 0 and task.get("save_state"):
        state_path = str(task["save_state"])
        torch.save(base.state_dict(), state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    dist.barrier()

    return {"genome": g, "key": genome_key(g),
            "fitness": float(np.mean(skill)) + task.get("parsimony", 0) * n_par / 1e6,
            "skill_per_h": skill.tolist(), "mean_skill": float(np.mean(skill)),
            "dir_acc": dacc.tolist(), "mean_dir_acc": float(np.mean(dacc)),
            "params": int(n_par), "seconds": round(time.time() - t0, 1),
            "state_path": state_path, "steps_done": steps, "truncated": False,
            "device": task["device"]}


# --------------------------------------------------------------------------- #
# fitness evaluation of one genome  (= one training instance)                  #
# --------------------------------------------------------------------------- #
def evaluate_genome(task: dict) -> dict:
    """Crash-safe wrapper: a bad genome scores badly, it never kills the run."""
    try:
        return _evaluate_inner(task)
    except MemoryError:
        err = "MemoryError"
    except RuntimeError as e:
        err = f"RuntimeError: {str(e)[:160]}"
    except Exception as e:                                  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:160]}"
    return _dead(task, err)


def _dead(task: dict, why: str) -> dict:
    g = task["genome"]
    return {"genome": g, "key": genome_key(g), "fitness": 1e6, "skill_per_h": [],
            "mean_skill": float("nan"), "dir_acc": [], "mean_dir_acc": float("nan"),
            "params": 0, "seconds": 0.0, "state_path": None, "steps_done": 0,
            "truncated": True, "error": why, "device": task.get("device", "?")}


def _evaluate_inner(task: dict) -> dict:
    g = task["genome"]
    device = torch.device(task["device"])
    torch.set_num_threads(task.get("threads", 1))
    if device.type == "cpu" and task.get("rlimit_mb"):
        # never cap address space when CUDA is in the picture: the driver
        # reserves tens of GB of *virtual* memory and the cap turns that into
        # a spurious ENOMEM
        _limit_address_space(task["rlimit_mb"])
    seed = task["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    D = load_cache(task.get("placement", "mmap"))
    tr, va = D["train_idx"], D["val_idx"]
    n_feat, n_out = D["meta"]["n_features"], D["meta"]["n_horizons"]
    ystd = torch.from_numpy(D["y_std"]).to(device)

    g, fits = fit_to_memory(g, n_feat, task.get("mem_budget_mb", 400))
    if not fits:
        return _dead({**task, "genome": g},
                     f"too large ({estimate_peak_mb(g, n_feat):.0f}MB)")

    L = g["lookback"]
    tr, va = tr[tr >= L - 1], va[va >= L - 1]
    on_dev = bool(task.get("on_device", False))
    src = WindowSource(D["X"], D["Y"], device, on_dev)
    offsets = torch.arange(-L + 1, 1, device=device) if on_dev \
        else np.arange(-L + 1, 1, dtype=np.int64)

    model = build_model(g, n_feat, n_out).to(device)
    n_par = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=g["lr"],
                            weight_decay=g["weight_decay"])
    steps = task["inner_steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=g["lr"], total_steps=steps, pct_start=0.25)

    hs = np.asarray(D["meta"]["horizon_steps"], dtype=np.float64)
    w = hs ** g["horizon_w"]
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    lossfn = nn.HuberLoss(delta=g["huber_delta"], reduction="none")
    bs = g["batch_size"]
    rng = np.random.default_rng(seed)
    ac, need_scaler = _autocast(device, task.get("precision", "fp32"))
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)

    if task.get("init_state") and Path(task["init_state"]).exists():
        try:
            sd = torch.load(task["init_state"], map_location=device,
                            weights_only=True)
            model.load_state_dict(sd, strict=True)
        except Exception:
            pass

    t0 = time.time()
    model.train()
    step = -1
    for step in range(steps):
        bidx = tr[rng.integers(0, len(tr), size=min(bs, len(tr)))]
        xb, yb = src.batch(bidx, offsets, ystd)
        if g["feat_drop"] > 0:
            mask = (torch.rand(1, 1, n_feat, device=device) > g["feat_drop"])
            xb = xb * mask.to(xb.dtype)
        with ac:
            loss = (lossfn(model(xb), yb) * w).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), g["grad_clip"])
        scaler.step(opt); scaler.update(); sched.step()
        if task.get("deadline") and (step & 31) == 0 and time.time() > task["deadline"]:
            break
    steps_done = step + 1
    truncated = steps_done < steps

    model.eval()
    se = np.zeros(n_out); var = np.zeros(n_out); hit = np.zeros(n_out); n = 0
    with torch.no_grad():
        vb = max(256, min(8192, bs * 4))
        for i in range(0, len(va), vb):
            bidx = va[i:i + vb]
            xb, yb = src.batch(bidx, offsets, ystd)
            with ac:
                p = model(xb)
            p = p.float()
            se += ((p - yb) ** 2).sum(0).cpu().numpy()
            var += (yb ** 2).sum(0).cpu().numpy()
            hit += ((p.sign() == yb.sign()) & (yb != 0)).sum(0).cpu().numpy()
            n += len(bidx)

    mse, base = se / max(n, 1), var / max(n, 1)
    skill = mse / np.maximum(base, 1e-12)
    dacc = hit / max(n, 1)

    fitness = float(np.mean(skill)) + task.get("parsimony", 0.0) * n_par / 1e6
    if not np.isfinite(fitness):
        fitness = 1e6

    state_path = None
    if task.get("save_state"):
        state_path = str(Path(task["save_state"]))
        torch.save(model.state_dict(), state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    return {
        "genome": g, "key": genome_key(g), "fitness": fitness,
        "skill_per_h": skill.tolist(), "mean_skill": float(np.mean(skill)),
        "dir_acc": dacc.tolist(), "mean_dir_acc": float(np.mean(dacc)),
        "params": int(n_par), "seconds": round(time.time() - t0, 1),
        "state_path": state_path, "steps_done": steps_done,
        "truncated": truncated, "device": task["device"],
    }


# --------------------------------------------------------------------------- #
# heterogeneous population evaluation                                          #
# --------------------------------------------------------------------------- #
def evaluate_population(tasks: list[dict], n_gpu: int, n_cpu: int,
                        ctx_gpu, ctx_cpu) -> list[dict]:
    """GPU islands and CPU islands chew through the population concurrently.

    A worker killed by the OS is retried serially, so one bad individual can
    never take down the evolution run.
    """
    gpu_tasks = [(i, t) for i, t in enumerate(tasks) if t["device"] != "cpu"]
    cpu_tasks = [(i, t) for i, t in enumerate(tasks) if t["device"] == "cpu"]

    if (n_gpu + n_cpu) <= 1 or len(tasks) == 1:
        return [evaluate_genome(t) for t in tasks]

    results: dict[int, dict] = {}
    try:
        with ExitStack() as stack:
            futs = {}
            if gpu_tasks:
                gex = stack.enter_context(ProcessPoolExecutor(
                    max_workers=max(1, min(n_gpu, len(gpu_tasks))),
                    mp_context=ctx_gpu))
                for i, t in gpu_tasks:
                    futs[gex.submit(evaluate_genome, t)] = i
            if cpu_tasks:
                cex = stack.enter_context(ProcessPoolExecutor(
                    max_workers=max(1, min(n_cpu, len(cpu_tasks))),
                    mp_context=ctx_cpu))
                for i, t in cpu_tasks:
                    futs[cex.submit(evaluate_genome, t)] = i
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:                       # noqa: BLE001
                    results[i] = _dead(tasks[i], f"worker died: {type(e).__name__}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [pool broken: {type(e).__name__}: {str(e)[:200]}] "
              "retrying survivors serially", flush=True)

    out = []
    for i, t in enumerate(tasks):
        r = results.get(i)
        if r is not None and "worker died" not in str(r.get("error", "")):
            out.append(r)
        else:
            out.append(evaluate_genome(t))
    return out


# --------------------------------------------------------------------------- #
# checkpointing                                                                #
# --------------------------------------------------------------------------- #
def save_state(state: dict) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    tmp = CKPT / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, CKPT / "state.json")


def load_state() -> dict | None:
    p = CKPT / "state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _handler(signum, frame):
    _STOP["flag"] = True
    print(f"\n[signal {signum}] finishing current generation, then checkpointing...",
          flush=True)


# --------------------------------------------------------------------------- #
# main evolution loop                                                          #
# --------------------------------------------------------------------------- #
def run(args) -> None:
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    for d in (CKPT, CKPT / "elites", LOGS):
        d.mkdir(parents=True, exist_ok=True)

    # cache must exist before the planner can size anything
    pre = (CACHE / "meta.json").exists() and not args.rebuild_cache
    if not pre:
        build_cache(subsample=args.subsample, force=args.rebuild_cache)

    info = detect(dataset_mb=dataset_mb(),
                  allow_cpu_islands=not args.no_cpu_islands,
                  force_cpu_islands=(args.cpu_islands
                                     if args.cpu_islands >= 0 else None),
                  max_islands_per_gpu=args.islands_per_gpu)
    apply_nvlink_env(info)
    apply_perf_env(info)
    IS_DIST, RANK, WORLD, LOCAL = dist_init()
    main_proc = RANK == 0
    plan = info.plan

    def log(*a, **k):
        if main_proc:
            print(*a, **k, flush=True)

    if IS_DIST:
        import torch.distributed as dist
        log(f"[distributed] backend={dist.get_backend()} world_size={WORLD} "
            f"mode={args.parallel}")
    log(render(info))

    if info.blockers and not args.ignore_blockers:
        log("\n" + "!" * 70)
        log("REFUSING TO START - the environment cannot support a training run:")
        for b in info.blockers:
            log(f"  * {b}")
        log("")
        log("Run  python3 scripts/diagnose.py  for a full breakdown.")
        log("Override with --ignore-blockers if you are certain.")
        log("!" * 70)
        raise SystemExit(3)

    meta = build_cache(subsample=args.subsample, force=args.rebuild_cache)
    if main_proc:
        (LOGS / "system_info.json").write_text(json.dumps(info.to_dict(), indent=2))
    if IS_DIST:
        import torch.distributed as dist
        dist.barrier()

    log(f"\ndata: {meta['n_rows']:,} rows | {meta['n_features']} features | "
        f"{meta['n_horizons']} horizons | train {meta['n_train']:,} "
        f"val {meta['n_val']:,} test {meta['n_test']:,}\n")

    # ---- island layout ----------------------------------------------------- #
    if IS_DIST:
        # this rank owns one GPU; pack several islands on it and claim its
        # share of the spare CPU cores as extra islands
        dev = f"cuda:{LOCAL % info.gpu_count}" if info.gpu_count else "cpu"
        n_gpu_isl = max(1, plan.islands_per_gpu) if info.gpu_count else 0
        n_cpu_isl = plan.cpu_islands // WORLD if info.gpu_count else 1
        gpu_devices = [dev] * max(1, n_gpu_isl)
        if not info.gpu_count:
            n_gpu_isl, gpu_devices = 0, []
        gpu_budget = plan.gpu_mem_budget_mb or plan.cpu_mem_budget_mb
        cpu_budget = plan.cpu_mem_budget_mb
    else:
        n_gpu_isl = plan.gpu_islands
        n_cpu_isl = plan.cpu_islands
        gpu_devices = [f"cuda:{i % max(1, info.gpu_count)}"
                       for i in range(n_gpu_isl)]
        gpu_budget = plan.gpu_mem_budget_mb
        cpu_budget = plan.cpu_mem_budget_mb
    total_isl = max(1, n_gpu_isl + n_cpu_isl)
    if args.mem_budget_mb:                      # explicit override wins
        gpu_budget = cpu_budget = args.mem_budget_mb

    if args.calibrate and main_proc:
        devs = sorted(set(gpu_devices)) + (["cpu"] if n_cpu_isl else [])
        log("calibrating device throughput...")
        for d, sp in calibrate(devs).items():
            log(f"    {d:<9} {sp:>8.1f} steps/s")
        import gc
        gc.collect()
        if info.cuda_available:
            torch.cuda.empty_cache()
        log("")

    # An RLIMIT_AS cap is a cheap OOM guard on small CPU-only boxes, but it is
    # actively harmful once CUDA is present: the driver reserves tens of GB of
    # *virtual* address space per context, so any sane-looking cap turns into a
    # spurious "OSError: [Errno 12] Cannot allocate memory" on every worker.
    rlimit_mb = 0 if info.cuda_available else (
        int(info.avail_ram_gb * 1024 * 1.6 / max(1, total_isl)) + 1200)
    on_device = plan.data_placement == "vram"
    placement = "ram" if plan.data_placement == "ram" else "mmap"
    headroom = info.avail_ram_gb * 1024 - dataset_mb() - 380 * (total_isl + 1)
    if placement == "ram" and not IS_DIST and n_cpu_isl and headroom > 0:
        load_cache("ram")            # preload so forked islands share via COW
        log(f"dataset preloaded into RAM, shared copy-on-write by "
            f"{n_cpu_isl} CPU island(s)")
    elif placement == "ram":
        placement = "mmap"           # too tight - fall back to the page cache
        log("RAM too tight for a resident copy -> memory-mapping instead")

    import multiprocessing as mp
    ctx_gpu = mp.get_context("spawn" if info.gpu_count else "fork")
    # forking a process that already holds a CUDA context is unsafe
    ctx_cpu = mp.get_context("spawn" if info.cuda_available else "fork")

    # ---- resume or cold start ---------------------------------------------- #
    st = None if args.fresh else load_state()
    if st:
        rng = random.Random()
        rng.setstate(pickle.loads(bytes.fromhex(st["rng"])))
        gen0, population = st["generation"], st["population"]
        history, best = st["history"], st["best"]
        elapsed_prev = st["elapsed_seconds"]
        log(f"RESUMING from generation {gen0} "
            f"({elapsed_prev/3600:.2f}h already spent, "
            f"best fitness {best['fitness']:.5f})\n")
    else:
        rng = random.Random(args.seed)
        gen0, history, best, elapsed_prev = 0, [], None, 0.0
        population = [random_genome(rng) for _ in range(args.pop_size)]
        log(f"COLD START: population {args.pop_size}\n")

    budget = args.max_hours * 3600
    t_start = time.time()

    def spent():
        return elapsed_prev + (time.time() - t_start)

    log(f"evolution: pop {args.pop_size} | {args.inner_steps} steps/individual | "
        f"elite {args.elite_frac:.0%} | mutate {args.mutate_rate:.0%}"
        f"@sigma {args.mutate_sigma} | immigrants {args.immigrant_frac:.0%}")
    log(f"islands  : {n_gpu_isl} GPU + {n_cpu_isl} CPU = {total_isl} concurrent"
        f" | budget {args.max_hours}h\n")

    gen = gen0
    dead_streak = 0
    while gen < args.generations:
        if spent() > budget:
            log(f"\n*** wall-clock budget reached ({args.max_hours}h) ***")
            break
        if _STOP["flag"]:
            break

        gt = time.time()
        remaining = budget - spent()
        # cap a generation so a slow CPU island can never stall the run
        if history:
            prev = max(h["seconds"] for h in history[-3:])
            cap = max(120.0, args.gen_timeout_mult * prev)
        else:
            cap = remaining
        deadline = time.time() + max(30.0, min(remaining, cap))

        tasks = []
        for i, g in enumerate(population):
            k = genome_key(g)
            init = CKPT / "elites" / f"{k}.pt"
            tasks.append({
                "genome": g,
                "seed": args.seed + gen * 1000 + i,
                "inner_steps": args.inner_steps,
                "parsimony": args.parsimony,
                "deadline": deadline,
                "init_state": str(init) if (args.lamarckian and init.exists()) else None,
                "save_state": str(CKPT / "elites" / f"{k}.pt"),
                "rlimit_mb": rlimit_mb,
                "placement": placement,
            })

        def bind_devices(subset):
            """Cheapest genomes go to CPU islands; the rest round-robin the GPUs."""
            n = len(subset)
            order = sorted(range(n), key=lambda i: cost_proxy(
                subset[i]["genome"], meta["n_features"]))
            if not gpu_devices:
                cpu_set = set(range(n))
            elif n_cpu_isl:
                cpu_set = set(order[:min(n_cpu_isl, n)])
            else:
                cpu_set = set()
            for i, t in enumerate(subset):
                to_cpu = i in cpu_set
                t["device"] = "cpu" if to_cpu else gpu_devices[i % len(gpu_devices)]
                t["threads"] = plan.threads_per_cpu_island if to_cpu else 1
                t["mem_budget_mb"] = cpu_budget if to_cpu else gpu_budget
                t["precision"] = "fp32" if to_cpu else plan.precision
                t["on_device"] = on_device and not to_cpu
            return subset

        if IS_DIST:
            import torch.distributed as dist
            if args.parallel == "ddp":
                for t in tasks:
                    t["device"] = gpu_devices[0] if gpu_devices else "cpu"
                    t["threads"] = 1
                    t["mem_budget_mb"] = gpu_budget
                    t["precision"] = plan.precision
                    t["on_device"] = on_device
                scored = [ddp_evaluate(t, RANK, WORLD) for t in tasks]
            else:
                mine = evaluate_population(bind_devices(tasks[RANK::WORLD]),
                                           n_gpu_isl, n_cpu_isl, ctx_gpu, ctx_cpu)
                bucket = [None] * WORLD
                dist.all_gather_object(bucket, mine)
                scored = [r for part in bucket for r in part]
        else:
            scored = evaluate_population(bind_devices(tasks), n_gpu_isl,
                                         n_cpu_isl, ctx_gpu, ctx_cpu)

        scored.sort(key=lambda d: d["fitness"])
        # a truncated individual trained on a short budget - don't crown it
        champ = next((s for s in scored if not s.get("truncated")), scored[0])

        if main_proc and champ.get("state_path") and (
                best is None or champ["fitness"] < best["fitness"]):
            best = dict(champ)
            src_p = Path(champ["state_path"])
            if src_p.exists():
                import shutil
                tmp = CKPT / "best_model.pt.tmp"
                shutil.copyfile(src_p, tmp)
                os.replace(tmp, CKPT / "best_model.pt")
                (CKPT / "best_meta.json").write_text(json.dumps({
                    "genome": best["genome"], "fitness": best["fitness"],
                    "mean_skill": best["mean_skill"],
                    "skill_per_h": best["skill_per_h"], "dir_acc": best["dir_acc"],
                    "params": best["params"],
                    "horizon_names": meta["horizon_names"],
                    "horizon_steps": meta["horizon_steps"],
                    "feature_names": meta["feature_names"],
                    "generation": gen,
                }, indent=2))

        ok = [s for s in scored if s["fitness"] < 1e5]
        n_fail = len(scored) - len(ok)
        n_trunc = sum(1 for s in scored if s.get("truncated") and not s.get("error"))
        history.append({
            "generation": gen, "best_fitness": champ["fitness"],
            "mean_fitness": float(np.mean([s["fitness"] for s in ok])) if ok
            else float("nan"),
            "best_dir_acc": champ["mean_dir_acc"],
            "best_backbone": champ["genome"]["backbone"],
            "n_failed": n_fail, "n_truncated": n_trunc,
            "seconds": round(time.time() - gt, 1),
        })

        log(f"gen {gen:>3} | best {champ['fitness']:.5f} "
            f"({champ['genome']['backbone']}, {champ['params']:,}p, "
            f"dir {champ['mean_dir_acc']*100:.2f}%) | "
            f"mean {history[-1]['mean_fitness']:.5f} | "
            f"fail {n_fail} trunc {n_trunc} | "
            f"{history[-1]['seconds']:.0f}s | "
            f"elapsed {spent()/3600:.2f}h / {args.max_hours}h")
        for s_ in scored:
            if s_.get("error"):
                log(f"        ! [{s_.get('device','?')}] "
                    f"{s_['genome']['backbone']:<5} {s_['error']}")
        if n_fail == len(scored):
            dead_streak += 1
            log(f"        !! entire generation failed "
                f"({dead_streak}/{args.max_dead_generations})")
            if dead_streak >= args.max_dead_generations:
                errs = {}
                for s_ in scored:
                    e = str(s_.get("error", "?")).split(":")[0]
                    errs[e] = errs.get(e, 0) + 1
                log("\n" + "!" * 70)
                log("ABORTING: every individual failed "
                    f"{dead_streak} generations in a row. Not burning the "
                    "remaining budget.")
                log(f"  dominant errors: {errs}")
                if any("Cannot allocate memory" in str(s_.get("error", ""))
                       for s_ in scored):
                    log("")
                    log("  ENOMEM with free RAM almost always means one of:")
                    log("    * the disk is full          -> df -h")
                    log("    * /dev/shm is tiny          -> df -h /dev/shm  "
                        "(needs GBs; docker --shm-size=16g)")
                    log("    * RLIMIT_AS / RLIMIT_DATA caps virtual memory")
                    log("    * vm.max_map_count or vm.overcommit_memory=2")
                log("")
                log("  Diagnose:  python3 scripts/diagnose.py")
                log("  Retry small: ./scripts/run_multigpu.sh "
                    "--islands-per-gpu 1 --no-cpu-islands")
                log("  Your last good checkpoint is intact and will resume.")
                log("!" * 70)
                break
        else:
            dead_streak = 0

        population = next_generation(
            [{"genome": s_["genome"], "fitness": s_["fitness"]} for s_ in scored],
            args.pop_size, rng, args.elite_frac, args.mutate_rate,
            args.immigrant_frac, args.mutate_sigma)
        gen += 1

        if IS_DIST:
            import torch.distributed as dist
            box = [population]
            dist.broadcast_object_list(box, src=0)
            population = box[0]

        if main_proc:
            save_state({
                "generation": gen, "population": population, "history": history,
                "best": best, "elapsed_seconds": spent(),
                "rng": pickle.dumps(rng.getstate()).hex(), "args": vars(args),
                "meta": meta, "strategy": info.strategy,
                "nvlink": info.nvlink_available, "world_size": WORLD,
                "plan": {"gpu_islands": n_gpu_isl, "cpu_islands": n_cpu_isl,
                         "precision": plan.precision,
                         "data_placement": plan.data_placement},
                "parallel": args.parallel if IS_DIST else "islands",
            })
            keep = {genome_key(g) for g in population}
            if best:
                keep.add(best["key"])
            for f in (CKPT / "elites").glob("*.pt"):
                if f.stem not in keep:
                    f.unlink(missing_ok=True)

    if main_proc:
        save_state({
            "generation": gen, "population": population, "history": history,
            "best": best, "elapsed_seconds": spent(),
            "rng": pickle.dumps(rng.getstate()).hex(), "args": vars(args),
            "meta": meta, "strategy": info.strategy,
            "nvlink": info.nvlink_available, "world_size": WORLD,
            "plan": {"gpu_islands": n_gpu_isl, "cpu_islands": n_cpu_isl,
                     "precision": plan.precision,
                     "data_placement": plan.data_placement},
            "parallel": args.parallel if IS_DIST else "islands",
        })
        log("\n" + "=" * 70)
        if best:
            log(f"BEST fitness   : {best['fitness']:.5f}  (1.0 = zero-return baseline)")
            log(f"     backbone  : {best['genome']['backbone']}  "
                f"{best['params']:,} params")
            log(f"     dir. acc  : {best['mean_dir_acc']*100:.2f}% mean")
            for nm, sk, da in zip(meta["horizon_names"], best["skill_per_h"],
                                  best["dir_acc"]):
                log(f"       {nm:<7} skill {sk:.4f}   dir {da*100:5.2f}%")
        log(f"generations    : {gen}")
        log(f"elapsed        : {spent()/3600:.3f} h")
        log(f"checkpoint     : {CKPT/'state.json'}")
        log(f"model          : {CKPT/'best_model.pt'}")
        log("=" * 70)

    if IS_DIST:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- budget / size ---
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--inner-steps", type=int, default=2000,
                   help="gradient steps per individual per generation")
    p.add_argument("--max-hours", type=float, default=5.0,
                   help="wall-clock budget; carried across resumes")
    # --- evolution rate (conservative by default) ---
    p.add_argument("--elite-frac", type=float, default=ELITE_FRAC)
    p.add_argument("--mutate-rate", type=float, default=MUTATE_RATE)
    p.add_argument("--mutate-sigma", type=float, default=MUTATE_SIGMA)
    p.add_argument("--immigrant-frac", type=float, default=IMMIGRANT_FRAC)
    p.add_argument("--parsimony", type=float, default=0.002,
                   help="fitness penalty per 1M params")
    # --- hardware ---
    p.add_argument("--islands-per-gpu", type=int, default=4,
                   help="max concurrent training instances per GPU (VRAM permitting)")
    p.add_argument("--cpu-islands", type=int, default=-1,
                   help="-1 = auto from spare cores, 0 = disable")
    p.add_argument("--no-cpu-islands", action="store_true")
    p.add_argument("--mem-budget-mb", type=int, default=0,
                   help="override the per-island activation budget")
    p.add_argument("--parallel", choices=["population", "ddp"], default="population")
    p.add_argument("--calibrate", action="store_true",
                   help="benchmark device throughput before starting")
    p.add_argument("--gen-timeout-mult", type=float, default=2.5,
                   help="cap a generation at N x the recent generation time")
    # --- misc ---
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lamarckian", action="store_true", default=True)
    p.add_argument("--no-lamarckian", dest="lamarckian", action="store_false")
    p.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-dead-generations", type=int, default=2,
                   help="abort after N consecutive all-failed generations")
    p.add_argument("--ignore-blockers", action="store_true",
                   help="start even if the environment looks unusable")
    return p.parse_args()


if __name__ == "__main__":
    run(cli())
