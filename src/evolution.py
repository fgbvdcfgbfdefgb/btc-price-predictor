"""Genome definition and genetic operators for the island-model GA.

Default rates are deliberately CONSERVATIVE ("low evolution rate"): strong
elitism, gentle mutation and few random immigrants, so compute is spent
refining good solutions instead of churning the population.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

# gene name -> list of allowed values (categorical) or (lo, hi, "log"|"lin")
GENE_SPACE = {
    "backbone":    ["mlp", "gru", "tcn", "attn"],
    "lookback":    [16, 32, 64, 96, 128],
    "hidden":      [32, 48, 64, 96, 128, 192, 256],
    "depth":       [1, 2, 3, 4],
    "dropout":     (0.0, 0.45, "lin"),
    "lr":          (3e-5, 8e-3, "log"),
    "weight_decay": (1e-7, 1e-2, "log"),
    "batch_size":  [128, 256, 512, 1024],
    "act":         ["relu", "gelu", "silu", "tanh"],
    "huber_delta": (0.25, 3.0, "log"),
    "feat_drop":   (0.0, 0.30, "lin"),
    "grad_clip":   (0.2, 5.0, "log"),
    # per-horizon loss weighting exponent: <0 favours short horizons
    "horizon_w":   (-0.6, 0.6, "lin"),
}

# --- conservative defaults ------------------------------------------------- #
MUTATE_RATE = 0.12       # fraction of genes touched per child
MUTATE_SIGMA = 0.15      # jitter size for continuous genes
ELITE_FRAC = 0.30        # survivors copied verbatim
IMMIGRANT_FRAC = 0.04    # fresh random blood


def random_genome(rng: random.Random) -> dict:
    g = {}
    for k, sp in GENE_SPACE.items():
        if isinstance(sp, list):
            g[k] = rng.choice(sp)
        else:
            lo, hi, mode = sp
            g[k] = (math.exp(rng.uniform(math.log(lo), math.log(hi)))
                    if mode == "log" else rng.uniform(lo, hi))
    return g


def mutate(genome: dict, rng: random.Random, rate: float = MUTATE_RATE,
           sigma: float = MUTATE_SIGMA) -> dict:
    """Per-gene mutation: categorical resample or log/linear Gaussian jitter."""
    g = copy.deepcopy(genome)
    for k, sp in GENE_SPACE.items():
        if rng.random() > rate:
            continue
        if isinstance(sp, list):
            if k in ("lookback", "hidden", "depth"):     # ordered -> step neighbour
                i = sp.index(g[k])
                g[k] = sp[max(0, min(len(sp) - 1, i + rng.choice([-1, 1])))]
            else:
                g[k] = rng.choice(sp)
        else:
            lo, hi, mode = sp
            v = (math.exp(math.log(g[k]) + rng.gauss(0, sigma)) if mode == "log"
                 else g[k] + rng.gauss(0, sigma * (hi - lo)))
            g[k] = max(lo, min(hi, v))
    return g


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    """Uniform crossover; BLX-alpha blend for continuous genes."""
    c = {}
    for k, sp in GENE_SPACE.items():
        if isinstance(sp, list):
            c[k] = a[k] if rng.random() < 0.5 else b[k]
        else:
            lo, hi, _ = sp
            w = rng.uniform(-0.1, 1.1)
            c[k] = max(lo, min(hi, a[k] * w + b[k] * (1 - w)))
    return c


def tournament(pop: list[dict], rng: random.Random, k: int = 3) -> dict:
    """pop entries are {'genome':..., 'fitness': float} - LOWER is better."""
    cands = rng.sample(pop, min(k, len(pop)))
    return min(cands, key=lambda d: d["fitness"])["genome"]


def next_generation(scored: list[dict], pop_size: int, rng: random.Random,
                    elite_frac: float = ELITE_FRAC,
                    mutate_rate: float = MUTATE_RATE,
                    immigrant_frac: float = IMMIGRANT_FRAC,
                    mutate_sigma: float = MUTATE_SIGMA) -> list[dict]:
    """scored: [{'genome':..., 'fitness':...}]. Returns the next population."""
    scored = sorted(scored, key=lambda d: d["fitness"])
    n_elite = max(1, int(round(pop_size * elite_frac)))
    n_immig = int(pop_size * immigrant_frac)

    children = [copy.deepcopy(s["genome"]) for s in scored[:n_elite]]   # elitism
    while len(children) < pop_size - n_immig:
        p1, p2 = tournament(scored, rng), tournament(scored, rng)
        children.append(mutate(crossover(p1, p2, rng), rng,
                               mutate_rate, mutate_sigma))
    while len(children) < pop_size:
        children.append(random_genome(rng))
    return children


def genome_key(g: dict) -> str:
    """Stable short signature, used for caching evaluations across resumes."""
    s = json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in sorted(g.items())}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def cost_proxy(g: dict, n_feat: int = 57) -> float:
    """Relative training cost, used to hand cheap genomes to CPU islands."""
    B, L, H, D = g["batch_size"], g["lookback"], g["hidden"], g["depth"]
    if g["backbone"] == "mlp":
        return B * (L * n_feat * H + H * H * max(0, D - 1))
    if g["backbone"] == "attn":
        return B * L * (H * H * D * 4 + L * H * D)
    return B * L * H * H * D            # gru / tcn
