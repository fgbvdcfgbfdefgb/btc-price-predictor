#!/usr/bin/env bash
#
# Complete launcher for the evolutionary BTC forecaster.
#
# Detects every usable resource - GPUs, NVLink topology, tensor-core
# precision, spare CPU cores, system RAM - and launches the training run that
# fits the machine. Resumes automatically; pass --fresh to start over.
#
#   ./scripts/run_multigpu.sh                     # 5 h, auto everything
#   ./scripts/run_multigpu.sh --hours 12          # longer budget
#   ./scripts/run_multigpu.sh --mode ddp          # all GPUs on one individual
#   ./scripts/run_multigpu.sh --background        # detach + tail the log
#   ./scripts/run_multigpu.sh --fresh             # ignore the checkpoint
#   ./scripts/run_multigpu.sh --pop 48 --steps 4000
#   ./scripts/run_multigpu.sh --no-cpu-islands    # GPUs only
#
# Anything after `--` is forwarded verbatim to src/train.py.
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ----------------------------------------------------------------- defaults --
HOURS=5
MODE=population          # population | ddp
POP=""                   # auto from device count
STEPS=2000
ISLANDS_PER_GPU=4
CPU_ISLANDS=-1           # -1 = auto from spare cores
BACKGROUND=0
FRESH=""
EVAL_AFTER=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours)            HOURS="$2"; shift 2 ;;
    --mode)             MODE="$2"; shift 2 ;;
    --pop)              POP="$2"; shift 2 ;;
    --steps)            STEPS="$2"; shift 2 ;;
    --islands-per-gpu)  ISLANDS_PER_GPU="$2"; shift 2 ;;
    --cpu-islands)      CPU_ISLANDS="$2"; shift 2 ;;
    --no-cpu-islands)   CPU_ISLANDS=0; shift ;;
    --background|-b)    BACKGROUND=1; shift ;;
    --fresh)            FRESH="--fresh"; shift ;;
    --no-eval)          EVAL_AFTER=0; shift ;;
    --)                 shift; EXTRA+=("$@"); break ;;
    -h|--help)          sed -n '2,20p' "$0"; exit 0 ;;
    *)                  EXTRA+=("$1"); shift ;;
  esac
done

log()  { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mXX\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- preflight --
log "preflight"
command -v python3 >/dev/null || die "python3 not found"
PY=python3
$PY - <<'EOF' || exit 1
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "torch")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"XX missing python packages: {', '.join(missing)}")
    print("   run ./setup.sh  (or ./setup.sh cu124 for CUDA)")
    sys.exit(1)
EOF

# Disk and /dev/shm are the two resources that make workers die with a
# misleading "Cannot allocate memory" on boxes with plenty of free RAM.
$PY - <<'EOF' || exit 1
import os, sys
s = os.statvfs(".")
free_gb = s.f_bavail * s.f_frsize / 1e9
if free_gb < 2.0:
    print(f"XX only {free_gb:.2f} GB of disk free here.")
    print("   Checkpoints, the feature cache and temp files cannot be written;")
    print("   workers then die with ENOSPC or ENOMEM. Free space and retry.")
    print("   Largest offenders:  du -xh . | sort -h | tail -20")
    sys.exit(1)
try:
    s = os.statvfs("/dev/shm")
    shm_mb = s.f_blocks * s.f_frsize / 1e6
    if shm_mb < 256:
        print(f"XX /dev/shm is only {shm_mb:.0f} MB.")
        print("   PyTorch worker processes need far more and will fail with")
        print("   'OSError: [Errno 12] Cannot allocate memory'.")
        print("   docker:     docker run --shm-size=16g ...")
        print("   kubernetes: emptyDir {medium: Memory} mounted at /dev/shm")
        print("   Diagnose:   python3 scripts/diagnose.py")
        sys.exit(1)
    if shm_mb < 2048:
        print(f"!! /dev/shm is {shm_mb:.0f} MB - tight for many workers; "
              "the planner will reduce the island count")
except FileNotFoundError:
    pass
EOF

if [ ! -f data/processed/btc_1m.parquet ]; then
  warn "dataset missing - downloading one year of 1-minute candles"
  $PY src/download_data.py || die "download failed"
fi

NGPU=$($PY -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
NCORE=$($PY -c "import os;print(os.cpu_count() or 1)")
log "GPUs: ${NGPU}   logical cores: ${NCORE}"

# ------------------------------------------------- full hardware inventory --
log "hardware inventory"
$PY src/system_detect.py || die "detection failed"

# ------------------------------------------------------- NCCL / NVLink env --
if [ "${NGPU}" -ge 2 ]; then
  if nvidia-smi nvlink --status 2>/dev/null | grep -q 'GB/s'; then
    log "NVLink active -> enabling P2P, NVLS and NVL-preferred transport"
    export NCCL_P2P_DISABLE=0
    export NCCL_P2P_LEVEL=NVL
    export NCCL_NVLS_ENABLE=1
  else
    warn "no NVLink -> NCCL will use PCIe (P2P disabled to avoid stalls)"
    export NCCL_P2P_DISABLE=1
  fi
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export CUDA_DEVICE_MAX_CONNECTIONS=1      # better NVLink/compute overlap
fi

# leave one core per rank for the data path, give the rest to math
if [ "${NGPU}" -ge 1 ]; then
  export OMP_NUM_THREADS=$(( NCORE / NGPU > 0 ? NCORE / NGPU : 1 ))
else
  export OMP_NUM_THREADS=1
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ------------------------------------------------------- population sizing --
if [ -z "${POP}" ]; then
  if [ "${NGPU}" -ge 1 ]; then
    POP=$(( NGPU * ISLANDS_PER_GPU * 4 ))
    [ "${POP}" -lt 16 ] && POP=16
  else
    POP=16
  fi
fi

mkdir -p logs checkpoints
STAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="logs/train-${STAMP}.log"

# low evolution rate: strong elitism, gentle mutation, few immigrants
COMMON=(
  --max-hours "${HOURS}"
  --pop-size "${POP}"
  --inner-steps "${STEPS}"
  --islands-per-gpu "${ISLANDS_PER_GPU}"
  --cpu-islands "${CPU_ISLANDS}"
  --elite-frac 0.30
  --mutate-rate 0.12
  --mutate-sigma 0.15
  --immigrant-frac 0.04
  --calibrate
)
[ -n "${FRESH}" ] && COMMON+=("${FRESH}")
[ ${#EXTRA[@]} -gt 0 ] && COMMON+=("${EXTRA[@]}")

if [ -f checkpoints/state.json ] && [ -z "${FRESH}" ]; then
  RESUME=$($PY - <<'EOF'
import json
s = json.load(open("checkpoints/state.json"))
print(f"resuming at generation {s['generation']} "
      f"({s['elapsed_seconds']/3600:.2f}h spent, best {s['best']['fitness']:.5f})")
EOF
)
  log "${RESUME}"
else
  log "starting a fresh evolution"
fi

# --------------------------------------------------------------- build cmd --
if [ "${NGPU}" -ge 2 ]; then
  log "launching torchrun: ${NGPU} ranks, mode=${MODE}, pop=${POP}, ${HOURS}h"
  CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}"
       src/train.py --parallel "${MODE}" "${COMMON[@]}")
else
  [ "${MODE}" = "ddp" ] && warn "ddp mode needs >=2 GPUs - using local islands"
  log "launching local island trainer: pop=${POP}, ${HOURS}h"
  CMD=("${PY}" src/train.py "${COMMON[@]}")
fi

# --------------------------------------------------------------------- run --
cleanup() { warn "signal received - train.py checkpoints before exiting"; }
trap cleanup INT TERM

if [ "${BACKGROUND}" -eq 1 ]; then
  nohup "${CMD[@]}" > "${LOGFILE}" 2>&1 &
  PID=$!
  echo "${PID}" > logs/train.pid
  log "running in background, pid ${PID}"
  log "log: ${LOGFILE}"
  log "stop with:  kill -TERM ${PID}   (checkpoints first)"
  sleep 3
  tail -f "${LOGFILE}"
else
  "${CMD[@]}" 2>&1 | tee "${LOGFILE}"
fi

# ---------------------------------------------------------------- evaluate --
if [ "${EVAL_AFTER}" -eq 1 ] && [ -f checkpoints/best_model.pt ]; then
  log "held-out evaluation"
  $PY src/evaluate.py test 2>&1 | tee -a "${LOGFILE}"
  log "forecast"
  $PY src/predict.py 2>&1 | tee -a "${LOGFILE}" || true
fi

log "done - log saved to ${LOGFILE}"
