"""
Inference: load the evolved champion and forecast BTC from 1 minute to 12 hours.

    python src/predict.py                # forecast from the newest bar on disk
    python src/predict.py --refresh      # pull the latest candles first
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import HORIZONS, build_features      # noqa: E402
from models import build_model                     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
CACHE = ROOT / "data" / "processed" / "cache"
DATA = ROOT / "data" / "processed" / "btc_1m.parquet"


def load_champion():
    meta = json.loads((CKPT / "best_meta.json").read_text())
    g = meta["genome"]
    n_feat = len(meta["feature_names"])
    n_out = len(meta["horizon_names"])
    model = build_model(g, n_feat, n_out)
    model.load_state_dict(torch.load(CKPT / "best_model.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    if not (CACHE / "scaler.pkl").exists():
        # regenerable, so not committed: build it on first use
        print("building the feature cache (one-off, ~20 s)...", flush=True)
        from train import build_cache                 # noqa: PLC0415
        build_cache()
    with open(CACHE / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    y_std = np.load(CACHE / "y_std.npy")
    return model, meta, scaler, y_std


def forecast(df: pd.DataFrame | None = None) -> dict:
    model, meta, scaler, y_std = load_champion()
    g = meta["genome"]
    L = g["lookback"]

    if df is None:
        df = pd.read_parquet(DATA)
    need = L + 1500                                   # feature warm-up history
    df = df.iloc[-need:].reset_index(drop=True)

    F = build_features(df)
    F = F[meta["feature_names"]]                      # exact training order
    Xn = np.clip((F.values.astype(np.float32) - scaler["median"]) / scaler["scale"],
                 -8, 8)
    Xn = np.nan_to_num(Xn, nan=0.0).astype(np.float32)

    win = torch.from_numpy(Xn[-L:][None, ...])        # (1, L, F)
    with torch.no_grad():
        logret = model(win).numpy()[0] * y_std

    spot = float(df.close.iloc[-1])
    ts = pd.Timestamp(df.ts.iloc[-1])

    # residual sigma per horizon, taken from the held-out evaluation if present
    sig = None
    ev = ROOT / "logs" / "eval_test.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        sig = {r["horizon"]: r["mae_bps"] / 1e4 * 1.2533 for r in e["rows"]}

    out = []
    for nm, lr in zip(meta["horizon_names"], logret):
        h = nm.replace("y_", "")
        price = spot * float(np.exp(lr))
        row = {
            "horizon": h,
            "minutes": HORIZONS[h],
            "valid_at": str(ts + pd.Timedelta(minutes=HORIZONS[h])),
            "pred_log_return": float(lr),
            "pred_return_pct": float(np.expm1(lr) * 100),
            "pred_price": price,
            "direction": "UP" if lr > 0 else "DOWN",
        }
        if sig and h in sig:
            s = sig[h]
            row["price_lo_68"] = spot * float(np.exp(lr - s))
            row["price_hi_68"] = spot * float(np.exp(lr + s))
        out.append(row)

    return {"as_of": str(ts), "spot": spot,
            "backbone": g["backbone"], "forecasts": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="download the newest candles before forecasting")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.refresh:
        import download_data
        download_data.main()

    r = forecast()
    if a.json:
        print(json.dumps(r, indent=2)); return

    print("=" * 78)
    print(f"  BTC/USDT forecast   as of {r['as_of']}   spot ${r['spot']:,.2f}")
    print(f"  model: evolved {r['backbone']}")
    print("=" * 78)
    print(f"{'horizon':>8} {'valid at (UTC)':>20} {'return':>9} "
          f"{'price':>13} {'68% band':>26}")
    print("-" * 78)
    for f in r["forecasts"]:
        band = (f"${f['price_lo_68']:>11,.0f} - ${f['price_hi_68']:>11,.0f}"
                if "price_lo_68" in f else "")
        arrow = "^" if f["direction"] == "UP" else "v"
        print(f"{f['horizon']:>8} {f['valid_at'][:16]:>20} "
              f"{arrow}{f['pred_return_pct']:>7.3f}% ${f['pred_price']:>12,.2f} {band:>26}")
    print("=" * 78)
    print("Point forecasts on a near-efficient market. Treat as a weak signal,")
    print("not a trading instruction.")


if __name__ == "__main__":
    main()
