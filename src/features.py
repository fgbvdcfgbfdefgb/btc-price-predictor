"""
Causal feature engineering + multi-horizon log-return targets for BTC 1m OHLCV.

Every feature at row t uses only information available at or before t.
Targets are forward log returns:  y_h(t) = log( close[t+h] / close[t] )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 1 minute -> 12 hours
HORIZONS: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "12h": 720,
}

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "btc_1m.parquet"


# --------------------------------------------------------------------------- #
# indicators (all causal)                                                      #
# --------------------------------------------------------------------------- #
def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low,
                    (df.high - pc).abs(),
                    (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, lc = df.close, np.log(df.close)
    r1 = lc.diff()

    # --- momentum: log return over increasing lookbacks -------------------- #
    for w in (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440):
        f[f"ret_{w}"] = lc.diff(w)

    # --- realised volatility ------------------------------------------------ #
    for w in (5, 15, 60, 240, 1440):
        f[f"vol_{w}"] = r1.rolling(w).std()
    f["vol_ratio_60_1440"] = f["vol_60"] / f["vol_1440"].replace(0, np.nan)

    # --- price position vs moving averages ---------------------------------- #
    for w in (5, 15, 60, 240, 1440):
        ma = c.rolling(w).mean()
        f[f"ma_dist_{w}"] = (c - ma) / ma
    for w in (60, 240, 1440):
        lo, hi = df.low.rolling(w).min(), df.high.rolling(w).max()
        f[f"chan_pos_{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)

    # --- Bollinger ----------------------------------------------------------- #
    for w in (60, 240):
        ma, sd = c.rolling(w).mean(), c.rolling(w).std()
        f[f"bb_{w}"] = (c - ma) / (2 * sd).replace(0, np.nan)

    # --- RSI / MACD / ATR ----------------------------------------------------- #
    for n in (14, 60, 240):
        f[f"rsi_{n}"] = _rsi(c, n) / 100 - 0.5
    ema12, ema26 = c.ewm(span=12 * 60).mean(), c.ewm(span=26 * 60).mean()
    macd = (ema12 - ema26) / c
    f["macd"] = macd
    f["macd_sig"] = macd.ewm(span=9 * 60).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]
    for n in (60, 1440):
        f[f"atr_{n}"] = _atr(df, n) / c

    # --- candle micro-structure ------------------------------------------------ #
    rng = (df.high - df.low).replace(0, np.nan)
    f["body"] = (df.close - df.open) / c
    f["upper_wick"] = (df.high - df[["open", "close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df.low) / rng
    f["hl_range"] = rng / c

    # --- volume / flow ----------------------------------------------------------- #
    lv = np.log1p(df.volume)
    for w in (15, 60, 240, 1440):
        m, s = lv.rolling(w).mean(), lv.rolling(w).std()
        f[f"volz_{w}"] = (lv - m) / s.replace(0, np.nan)
    f["taker_ratio"] = (df.taker_buy_base / df.volume.replace(0, np.nan)) - 0.5
    for w in (15, 60, 240):
        f[f"taker_ratio_{w}"] = f["taker_ratio"].rolling(w).mean()
    f["trades_z"] = ((np.log1p(df.trades) - np.log1p(df.trades).rolling(1440).mean())
                     / np.log1p(df.trades).rolling(1440).std().replace(0, np.nan))
    vwap = (df.quote_volume.rolling(60).sum()
            / df.volume.rolling(60).sum().replace(0, np.nan))
    f["vwap_dist_60"] = (c - vwap) / c

    # --- autocorrelation / mean-reversion signal (vectorised) ------------------ #
    for w in (60, 240):
        f[f"ac_{w}"] = r1.rolling(w).corr(r1.shift(1))

    # --- seasonality (cyclical encodings) ---------------------------------------- #
    ts = df.ts.dt
    mins = ts.hour * 60 + ts.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    f["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

    return f.replace([np.inf, -np.inf], np.nan)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    lc = np.log(df.close)
    return pd.DataFrame(
        {f"y_{k}": lc.shift(-h) - lc for k, h in HORIZONS.items()},
        index=df.index)


# --------------------------------------------------------------------------- #
# dataset assembly with chronological split + embargo                          #
# --------------------------------------------------------------------------- #
def prepare(path: Path = DATA,
            train_frac: float = 0.70,
            val_frac: float = 0.15,
            subsample: int | None = None,
            verbose: bool = True) -> dict:
    df = pd.read_parquet(path)
    if subsample:
        df = df.iloc[-subsample:].reset_index(drop=True)

    X = build_features(df)
    Y = build_targets(df)

    warmup = 1440                      # rows whose features need history
    max_h = max(HORIZONS.values())     # rows whose targets look forward
    valid = np.zeros(len(df), bool)
    valid[warmup:len(df) - max_h] = True
    valid &= ~X.isna().any(axis=1).values
    valid &= ~Y.isna().any(axis=1).values
    idx = np.flatnonzero(valid)

    n = len(idx)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    # embargo = longest horizon, so no training row can see a val/test outcome
    tr = idx[: max(0, i_tr - max_h)]
    va = idx[i_tr: max(i_tr, i_va - max_h)]
    te = idx[i_va:]

    Xv = X.values.astype(np.float32)
    Yv = Y.values.astype(np.float32)

    # standardise features on TRAIN ONLY (robust: median / IQR)
    med = np.nanmedian(Xv[tr], axis=0)
    q1, q3 = np.nanpercentile(Xv[tr], [25, 75], axis=0)
    scale = np.where((q3 - q1) > 1e-12, (q3 - q1) / 1.349, 1.0)
    Xn = np.clip((Xv - med) / scale, -8, 8)
    Xn = np.nan_to_num(Xn, nan=0.0).astype(np.float32)

    # scale targets to ~unit variance per horizon (train stats) so the loss
    # is not dominated by the 12h head
    ystd = Yv[tr].std(axis=0)
    ystd = np.where(ystd > 1e-12, ystd, 1.0).astype(np.float32)

    out = {
        "X": Xn, "Y": Yv, "y_std": ystd,
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "feature_names": list(X.columns),
        "horizon_names": [f"y_{k}" for k in HORIZONS],
        "horizon_steps": list(HORIZONS.values()),
        "close": df.close.values.astype(np.float32),
        "ts": df.ts.values,
        "scaler": {"median": med, "scale": scale},
    }
    if verbose:
        print(f"features   : {Xn.shape[1]}")
        print(f"usable rows: {n:,}")
        print(f"  train    : {len(tr):,}  {df.ts.iloc[tr[0]].date()} -> {df.ts.iloc[tr[-1]].date()}")
        print(f"  val      : {len(va):,}  {df.ts.iloc[va[0]].date()} -> {df.ts.iloc[va[-1]].date()}")
        print(f"  test     : {len(te):,}  {df.ts.iloc[te[0]].date()} -> {df.ts.iloc[te[-1]].date()}")
        print(f"  embargo  : {max_h} min between splits")
    return out


if __name__ == "__main__":
    d = prepare()
    print("\nOK ->", d["X"].shape, d["Y"].shape)
