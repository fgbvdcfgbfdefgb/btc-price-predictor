#!/usr/bin/env bash
# One-shot environment setup.
#   ./setup.sh          CPU-only torch
#   ./setup.sh cu124    CUDA 12.4 build (needed for multi-GPU / NVLink)
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${1:-cpu}"
echo ">> installing torch (${VARIANT})"
pip install torch --index-url "https://download.pytorch.org/whl/${VARIANT}"

echo ">> installing the rest"
pip install numpy pandas pyarrow matplotlib jupyter ipykernel

if [ ! -f data/processed/btc_1m.parquet ]; then
  echo ">> downloading one year of 1-minute BTCUSDT candles"
  python3 src/download_data.py
else
  echo ">> dataset already present: $(du -h data/processed/btc_1m.parquet | cut -f1)"
fi

echo ">> hardware report"
python3 src/system_detect.py

cat <<'EOF'

Ready. Next:

  # 5-hour evolutionary run (resumable - re-run to continue)
  python3 src/train.py --max-hours 5

  # multi-GPU over NVLink
  ./scripts/run_multigpu.sh

  # notebook
  jupyter notebook notebooks/btc_price.ipynb
EOF
