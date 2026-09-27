"""
Held-out test-set evaluation for the evolved champion.

Reports, per horizon:
  skill      MSE relative to the zero-return baseline  (<1 = beats naive)
  dir_acc    directional accuracy
  IC         Spearman information coefficient (rank corr of pred vs actual)
  MAE_bps    mean absolute error in basis points of price
  price_MAPE error of the reconstructed USD price
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model                     # noqa: E402
from train import CACHE, CKPT, load_cache          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main(split: str = "test") -> dict:
    meta = json.loads((CKPT / "best_meta.json").read_text())
    g = meta["genome"]
    D = load_cache()
    X, Y = D["X"], D["Y"]
    idx = D[f"{split}_idx"]
    ystd = D["y_std"]
    n_feat, n_out = D["meta"]["n_features"], D["meta"]["n_horizons"]
    names = D["meta"]["horizon_names"]
    steps = D["meta"]["horizon_steps"]

    model = build_model(g, n_feat, n_out)
    model.load_state_dict(torch.load(CKPT / "best_model.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    L = g["lookback"]
    idx = idx[idx >= L - 1]
    offsets = np.arange(-L + 1, 1, dtype=np.int64)

    preds, acts = [], []
    with torch.no_grad():
        bs = 2048
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            win = b[:, None] + offsets[None, :]
            xb = torch.from_numpy(np.ascontiguousarray(X[win]))
            preds.append(model(xb).numpy())
            acts.append(Y[b])
    P = np.concatenate(preds) * ystd          # de-standardise -> log returns
    A = np.concatenate(acts)

    close = np.load(CACHE / "close.npy")
    c0 = close[idx]

    rows = []
    for j, (nm, st) in enumerate(zip(names, steps)):
        p, a = P[:, j], A[:, j]
        mse, base = float(np.mean((p - a) ** 2)), float(np.mean(a ** 2))
        mask = a != 0
        rows.append({
            "horizon": nm.replace("y_", ""),
            "minutes": st,
            "skill": mse / max(base, 1e-15),
            "r2_vs_zero": 1 - mse / max(base, 1e-15),
            "dir_acc": float(np.mean(np.sign(p[mask]) == np.sign(a[mask]))),
            "ic": _spearman(p, a),
            "mae_bps": float(np.mean(np.abs(p - a))) * 1e4,
            "price_mape": float(np.mean(np.abs(c0 * np.expm1(p) - c0 * np.expm1(a))
                                        / (c0 * np.exp(a)))) * 100,
        })

    print("=" * 84)
    print(f"  HELD-OUT {split.upper()} SET  -  {len(idx):,} samples, "
          f"{g['backbone']} champion")
    print("=" * 84)
    print(f"{'horizon':>8} {'skill':>9} {'R2_vs_0':>9} {'dir_acc':>9} "
          f"{'IC':>9} {'MAE_bps':>10} {'price_MAPE%':>12}")
    print("-" * 84)
    for r in rows:
        print(f"{r['horizon']:>8} {r['skill']:>9.4f} {r['r2_vs_zero']:>9.4f} "
              f"{r['dir_acc']*100:>8.2f}% {r['ic']:>9.4f} {r['mae_bps']:>10.2f} "
              f"{r['price_mape']:>11.4f}%")
    print("-" * 84)
    print(f"{'MEAN':>8} {np.mean([r['skill'] for r in rows]):>9.4f} "
          f"{np.mean([r['r2_vs_zero'] for r in rows]):>9.4f} "
          f"{np.mean([r['dir_acc'] for r in rows])*100:>8.2f}% "
          f"{np.mean([r['ic'] for r in rows]):>9.4f}")
    print("=" * 84)
    print("skill  < 1.0  and  R2_vs_0 > 0  mean the model beats 'predict no change'.")
    print("dir_acc materially > 50% is the signal that actually matters.")

    out = {"split": split, "n_samples": int(len(idx)), "genome": g, "rows": rows}
    (ROOT / "logs" / f"eval_{split}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
