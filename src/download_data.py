"""Download 1 year of BTCUSDT 1-minute OHLCV from Binance public data dumps.

Uses data.binance.vision (static S3 dumps, no API key, not geo-blocked)
instead of api.binance.com which is geo-restricted in this region.
Monthly archives for complete months + daily archives for the current month.
"""
import io, sys, zipfile, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error
import pandas as pd

SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
BASE     = "https://data.binance.vision/data/spot"
RAW      = Path("/home/user/btc_predictor/data/raw")
OUT      = Path("/home/user/btc_predictor/data/processed")
COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

def fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "btc-research/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

def load_zip(blob: bytes) -> pd.DataFrame:
    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = zf.namelist()[0]
    with zf.open(name) as fh:
        head = fh.read(200).decode("utf-8", "replace")
    hdr = 0 if head.lower().startswith("open_time") else None
    with zf.open(name) as fh:
        df = pd.read_csv(fh, header=hdr, names=None if hdr == 0 else COLS)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def grab(task):
    kind, label, url = task
    cache = RAW / f"{label}.zip"
    if cache.exists():
        blob = cache.read_bytes()
    else:
        blob = fetch(url)
        if blob is None:
            return label, None
        cache.write_bytes(blob)
    return label, load_zip(blob)

def main():
    today = dt.date.today()
    start = today - dt.timedelta(days=365)
    tasks = []
    # complete months
    y, m = start.year, start.month
    while (y, m) < (today.year, today.month):
        lab = f"{y:04d}-{m:02d}"
        tasks.append(("M", lab,
            f"{BASE}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{lab}.zip"))
        m += 1
        if m > 12: m, y = 1, y + 1
    # current (partial) month -> daily files
    d = dt.date(today.year, today.month, 1)
    while d < today:
        lab = d.isoformat()
        tasks.append(("D", lab,
            f"{BASE}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{lab}.zip"))
        d += dt.timedelta(days=1)

    print(f"Fetching {len(tasks)} archives ({start} -> {today})", flush=True)
    frames, missing = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for lab, df in ex.map(grab, tasks):
            if df is None:
                missing.append(lab); print(f"  miss {lab}", flush=True)
            else:
                frames.append(df); print(f"  ok   {lab}  {len(df):>7,} rows", flush=True)

    if not frames:
        sys.exit("No data downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df = df[["open_time","open","high","low","close","volume","quote_volume",
             "trades","taker_buy_base"]].copy()

    # Binance switched open_time from ms to microseconds during 2025 -> normalise
    ot = df["open_time"].astype("int64")
    df["open_time"] = ot.where(ot < 1_000_000_000_000_0, ot // 1000)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")

    df = (df.drop(columns=["open_time"])
            .dropna(subset=["close"])
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))

    cutoff = pd.Timestamp(start, tz="UTC")
    df = df[df["ts"] >= cutoff].reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "btc_1m.parquet", index=False, compression="zstd")
    df.to_csv(OUT / "btc_1m.csv.gz", index=False, compression="gzip")

    full = pd.date_range(df.ts.iloc[0], df.ts.iloc[-1], freq="1min")
    print("\n" + "="*62)
    print(f"rows        : {len(df):,}")
    print(f"range       : {df.ts.iloc[0]}  ->  {df.ts.iloc[-1]}")
    print(f"gaps        : {len(full) - len(df):,} missing minutes "
          f"({100*(1-len(df)/len(full)):.3f}%)")
    print(f"price range : ${df.close.min():,.2f} - ${df.close.max():,.2f}")
    print(f"last close  : ${df.close.iloc[-1]:,.2f}")
    print(f"missing arch: {missing or 'none'}")
    print("="*62)

if __name__ == "__main__":
    main()
