"""
Exhaustive compute detection + execution planning.

Finds every usable resource - GPUs, NVLink/P2P topology, tensor-core
precision, physical CPU cores, NUMA layout, system RAM, swap, disk - and
turns it into a concrete ExecutionPlan: how many islands to run, on which
devices, with how many threads, at what precision, and where the dataset
should live (VRAM > RAM > mmap).

Strategy selection
------------------
  >=2 GPUs fully connected by NVLink -> DDP over NCCL with P2P enabled
  >=2 GPUs but only PCIe             -> DDP over NCCL, P2P disabled
  1 GPU                              -> concurrent islands sharing the device
  0 GPUs                             -> parallel CPU islands

CPU islands run *in addition to* GPU islands whenever spare physical cores
and RAM exist, so the box is never left half-idle.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


# --------------------------------------------------------------------------- #
# dataclasses                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class GPUInfo:
    index: int
    name: str
    memory_mb: int
    free_mb: int = 0
    compute_capability: str = "?"
    multi_processors: int = 0
    supports_tf32: bool = False
    supports_bf16: bool = False


@dataclass
class ExecutionPlan:
    gpu_islands: int = 0
    islands_per_gpu: int = 0
    cpu_islands: int = 0
    total_islands: int = 1
    threads_per_cpu_island: int = 1
    threads_per_gpu_island: int = 1
    precision: str = "fp32"          # bf16 | fp16 | fp32
    data_placement: str = "mmap"     # vram | ram | mmap
    pin_memory: bool = False
    gpu_mem_budget_mb: int = 0
    cpu_mem_budget_mb: int = 0
    use_tf32: bool = False
    dataset_mb: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class SystemInfo:
    # GPU
    gpu_count: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)
    nvlink_available: bool = False
    nvlink_active_links: int = 0
    nvlink_pairs: list[tuple[int, int]] = field(default_factory=list)
    nvlink_fully_connected: bool = False
    nvlink_bandwidth_gbs: float = 0.0
    p2p_matrix: list[list[bool]] = field(default_factory=list)
    topology_matrix: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    mps_available: bool = False
    # CPU / memory
    cpu_logical: int = 0
    cpu_physical: int = 0
    numa_nodes: int = 1
    cpu_model: str = ""
    cpu_flags: list[str] = field(default_factory=list)
    total_ram_gb: float = 0.0
    avail_ram_gb: float = 0.0
    swap_gb: float = 0.0
    disk_free_gb: float = 0.0
    shm_total_mb: float = 0.0
    shm_free_mb: float = 0.0
    blockers: list[str] = field(default_factory=list)
    # software
    torch_version: str = "not installed"
    cuda_available: bool = False
    nccl_available: bool = False
    # derived
    strategy: str = "cpu-islands"
    world_size: int = 1
    plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["gpus"] = [asdict(g) for g in self.gpus]
        d["plan"] = asdict(self.plan)
        return d


# --------------------------------------------------------------------------- #
# NVLink / P2P                                                                 #
# --------------------------------------------------------------------------- #
def detect_nvlink(gpu_count: int) -> dict:
    out = {"available": False, "active_links": 0, "pairs": [],
           "fully_connected": False, "topology": "", "bandwidth": 0.0}
    if gpu_count < 1:
        return out

    status = _run(["nvidia-smi", "nvlink", "--status"])
    if status:
        bw = [float(x) for x in re.findall(r"Link\s+\d+:\s+([\d.]+)\s*GB/s", status)]
        out["active_links"] = len(bw)
        out["available"] = len(bw) > 0
        out["bandwidth"] = round(sum(bw) / len(bw), 2) if bw else 0.0

    topo = _run(["nvidia-smi", "topo", "-m"])
    if topo:
        out["topology"] = topo.strip()
        pairs, nvl = [], 0
        for line in topo.splitlines():
            m = re.match(r"^\s*GPU(\d+)\s+(.*)$", line)
            if not m:
                continue
            i, cells = int(m.group(1)), m.group(2).split()[:gpu_count]
            for j, c in enumerate(cells):
                if re.fullmatch(r"NV\d+", c):
                    nvl += 1
                    if i < j:
                        pairs.append((i, j))
        out["pairs"] = sorted(pairs)
        if nvl:
            out["available"] = True
        out["fully_connected"] = (gpu_count > 1
                                  and len(pairs) == gpu_count * (gpu_count - 1) // 2)
    return out


def _p2p_matrix(n: int) -> list[list[bool]]:
    try:
        import torch
        return [[bool(i != j and torch.cuda.can_device_access_peer(i, j))
                 for j in range(n)] for i in range(n)]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# CPU / memory / disk                                                          #
# --------------------------------------------------------------------------- #
def _cpu_details(info: SystemInfo) -> None:
    info.cpu_logical = os.cpu_count() or 1
    info.cpu_physical = info.cpu_logical

    txt = _run(["lscpu"]) or ""
    def grab(pat, cast=str, default=None):
        m = re.search(pat, txt)
        return cast(m.group(1).strip()) if m else default

    cps = grab(r"Core\(s\) per socket:\s+(\d+)", int)
    sockets = grab(r"Socket\(s\):\s+(\d+)", int)
    if cps and sockets:
        info.cpu_physical = cps * sockets
    info.numa_nodes = grab(r"NUMA node\(s\):\s+(\d+)", int, 1) or 1
    info.cpu_model = grab(r"Model name:\s+(.+)", str, "") or ""

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    all_flags = set(line.split(":", 1)[1].split())
                    want = ["avx2", "avx512f", "avx512_bf16", "avx_vnni",
                            "amx_bf16", "amx_int8", "sha_ni", "f16c"]
                    info.cpu_flags = [w for w in want if w in all_flags]
                    break
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            mi = f.read()
        g = lambda k: int(re.search(rf"{k}:\s+(\d+)", mi).group(1)) / 1024 / 1024
        info.total_ram_gb = round(g("MemTotal"), 2)
        info.avail_ram_gb = round(g("MemAvailable"), 2)
        info.swap_gb = round(g("SwapTotal"), 2)
    except Exception:
        pass

    try:
        s = os.statvfs(os.path.dirname(os.path.abspath(__file__)))
        info.disk_free_gb = round(s.f_bavail * s.f_frsize / 1e9, 2)
    except Exception:
        pass
    try:
        s = os.statvfs("/dev/shm")
        info.shm_total_mb = round(s.f_blocks * s.f_frsize / 1e6, 1)
        info.shm_free_mb = round(s.f_bavail * s.f_frsize / 1e6, 1)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# planner                                                                      #
# --------------------------------------------------------------------------- #
GPU_RUNTIME_MB = 900        # CUDA context + torch per process
CPU_RUNTIME_MB = 380        # torch + numpy RSS for the first process
CUDA_HOST_MB = 1400         # host RSS per process holding a CUDA context
SHM_PER_PROC_MB = 256       # /dev/shm a torch worker wants
CPU_FORK_MARGINAL_MB = 260  # extra RSS per FORKED island (shares .so pages)
MIN_ACT_MB = 64             # smallest useful activation budget


def build_plan(info: SystemInfo, dataset_mb: float = 160.0,
               max_islands_per_gpu: int = 4,
               allow_cpu_islands: bool = True,
               force_cpu_islands: int | None = None) -> ExecutionPlan:
    p = ExecutionPlan(dataset_mb=dataset_mb)

    # ---- GPU islands: pack as many per card as VRAM allows ---------------- #
    if info.gpu_count:
        free = min((g.free_mb or g.memory_mb) for g in info.gpus)
        # each island needs: runtime + a resident copy of the dataset + activations
        per_island = GPU_RUNTIME_MB + dataset_mb + 400
        n = int(max(1, min(max_islands_per_gpu, free // max(per_island, 1))))
        # every CUDA worker is a process holding a context: it needs shared
        # memory and host RAM, and those run out long before VRAM does
        if info.shm_free_mb > 0:
            by_shm = max(1, int(info.shm_free_mb // SHM_PER_PROC_MB))
            if by_shm < n * info.gpu_count:
                n = max(1, by_shm // max(1, info.gpu_count))
                p.notes.append(
                    f"/dev/shm is only {info.shm_free_mb:.0f} MB -> capped to "
                    f"{n} island(s) per GPU")
        by_host = max(1, int(info.avail_ram_gb * 1024 * 0.5
                             / (CUDA_HOST_MB * max(1, info.gpu_count))))
        n = max(1, min(n, by_host))
        p.islands_per_gpu = n
        p.gpu_islands = n * info.gpu_count
        usable = free - n * (GPU_RUNTIME_MB + dataset_mb)
        p.gpu_mem_budget_mb = int(max(128, usable / n))

        cc = min(float(g.compute_capability) for g in info.gpus
                 if g.compute_capability != "?") if info.gpus else 0.0
        p.use_tf32 = cc >= 8.0
        p.precision = "bf16" if cc >= 8.0 else "fp16"
        p.pin_memory = True
        p.notes.append(
            f"{p.islands_per_gpu} island(s) per GPU x {info.gpu_count} GPU(s) "
            f"= {p.gpu_islands} GPU islands ({free} MB free VRAM each)")
        if p.use_tf32:
            p.notes.append("Ampere+ detected -> TF32 matmul + bf16 autocast")

    # ---- dataset placement ------------------------------------------------- #
    if info.gpu_count:
        free = min((g.free_mb or g.memory_mb) for g in info.gpus)
        if dataset_mb * p.islands_per_gpu < free * 0.45:
            p.data_placement = "vram"
            p.notes.append(f"dataset ({dataset_mb:.0f} MB) pinned in VRAM - "
                           "batches never cross PCIe")
        elif dataset_mb < info.avail_ram_gb * 1024 * 0.35:
            p.data_placement = "ram"
        else:
            p.data_placement = "mmap"
    else:
        p.data_placement = ("ram" if dataset_mb < info.avail_ram_gb * 1024 * 0.35
                            else "mmap")
    if p.data_placement == "ram":
        p.notes.append(f"dataset ({dataset_mb:.0f} MB) held in system RAM")
    elif p.data_placement == "mmap":
        p.notes.append(f"dataset ({dataset_mb:.0f} MB) memory-mapped "
                       "(page cache shared between islands)")

    # ---- CPU islands: use whatever physical cores the GPUs don't need ----- #
    reserve = p.gpu_islands                      # ~1 core feeding each GPU island
    spare = max(0, info.cpu_physical - reserve - (1 if info.gpu_count else 0))
    # the orchestrator holds torch + (maybe) the dataset; forked islands then
    # cost only their marginal private heap on top of the shared pages
    parent_mb = CPU_RUNTIME_MB + (dataset_mb if p.data_placement == "ram" else 0)
    ram_for_cpu = info.avail_ram_gb * 1024 * 0.85 - parent_mb
    by_ram = int(max(0, ram_for_cpu // (CPU_FORK_MARGINAL_MB + MIN_ACT_MB)))

    if force_cpu_islands is not None:
        p.cpu_islands = max(0, force_cpu_islands)
    elif not allow_cpu_islands:
        p.cpu_islands = 0
    elif info.gpu_count == 0:
        p.cpu_islands = max(1, min(info.cpu_logical, by_ram))
    else:
        p.cpu_islands = max(0, min(spare, by_ram, 8))

    if p.cpu_islands:
        p.threads_per_cpu_island = max(
            1, (info.cpu_physical - reserve) // max(1, p.cpu_islands)) \
            if info.gpu_count else max(1, info.cpu_physical // p.cpu_islands)
        pool = (info.avail_ram_gb * 1024 * 0.85 - CPU_RUNTIME_MB
                - CPU_FORK_MARGINAL_MB * p.cpu_islands)
        if p.data_placement == "ram":
            pool -= dataset_mb
        p.cpu_mem_budget_mb = int(max(64, pool / p.cpu_islands))
        if info.gpu_count:
            p.notes.append(
                f"{p.cpu_islands} spare-core CPU island(s) alongside the GPUs "
                "(deadline-capped so they can never stall a generation)")

    p.threads_per_gpu_island = 1
    p.total_islands = max(1, p.gpu_islands + p.cpu_islands)
    return p


# --------------------------------------------------------------------------- #
# throughput calibration                                                       #
# --------------------------------------------------------------------------- #
def calibrate(devices: list[str], steps: int = 12) -> dict[str, float]:
    """Steps/second on a representative tiny workload, per device kind."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {}
    out = {}
    for dev in dict.fromkeys(devices):
        try:
            d = torch.device(dev)
            m = nn.Sequential(nn.Flatten(), nn.Linear(64 * 57, 256), nn.GELU(),
                              nn.Linear(256, 9)).to(d)
            o = torch.optim.AdamW(m.parameters(), lr=1e-3)
            x = torch.randn(256, 64, 57, device=d)
            y = torch.randn(256, 9, device=d)
            for _ in range(3):                      # warm-up
                o.zero_grad(); nn.functional.mse_loss(m(x), y).backward(); o.step()
            if d.type == "cuda":
                torch.cuda.synchronize(d)
            t0 = time.time()
            for _ in range(steps):
                o.zero_grad(); nn.functional.mse_loss(m(x), y).backward(); o.step()
            if d.type == "cuda":
                torch.cuda.synchronize(d)
            out[dev] = round(steps / max(time.time() - t0, 1e-6), 1)
        except Exception:
            out[dev] = 0.0
    return out


# --------------------------------------------------------------------------- #
# main probe                                                                   #
# --------------------------------------------------------------------------- #
def detect(dataset_mb: float = 160.0, allow_cpu_islands: bool = True,
           force_cpu_islands: int | None = None,
           max_islands_per_gpu: int = 4) -> SystemInfo:
    info = SystemInfo()
    _cpu_details(info)

    try:
        import torch
        info.torch_version = torch.__version__
        info.cuda_available = torch.cuda.is_available()
        info.cuda_version = getattr(torch.version, "cuda", "") or ""
        info.mps_available = bool(getattr(getattr(torch.backends, "mps", None),
                                          "is_available", lambda: False)())
        if info.cuda_available:
            info.gpu_count = torch.cuda.device_count()
            for i in range(info.gpu_count):
                pr = torch.cuda.get_device_properties(i)
                try:
                    free, _ = torch.cuda.mem_get_info(i)
                    free_mb = free // (1024 ** 2)
                except Exception:
                    free_mb = pr.total_memory // (1024 ** 2)
                cc = f"{pr.major}.{pr.minor}"
                info.gpus.append(GPUInfo(
                    i, pr.name, pr.total_memory // (1024 ** 2), free_mb, cc,
                    getattr(pr, "multi_processor_count", 0),
                    pr.major >= 8, pr.major >= 8))
            try:
                info.nccl_available = torch.distributed.is_nccl_available()
            except Exception:
                pass
            info.p2p_matrix = _p2p_matrix(info.gpu_count)
    except ImportError:
        info.notes.append("PyTorch not installed - GPU probe via nvidia-smi only.")

    if info.gpu_count == 0:
        q = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                  "--format=csv,noheader,nounits"])
        if q:
            for line in q.strip().splitlines():
                c = [x.strip() for x in line.split(",")]
                if len(c) >= 4:
                    info.gpus.append(GPUInfo(int(c[0]), c[1], int(c[2]), int(c[3])))
            info.gpu_count = len(info.gpus)

    drv = _run(["nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"])
    if drv:
        info.driver_version = drv.strip().splitlines()[0]

    nv = detect_nvlink(info.gpu_count)
    info.nvlink_available = nv["available"]
    info.nvlink_active_links = nv["active_links"]
    info.nvlink_pairs = nv["pairs"]
    info.nvlink_fully_connected = nv["fully_connected"]
    info.nvlink_bandwidth_gbs = nv["bandwidth"]
    info.topology_matrix = nv["topology"]

    # ---- strategy ---------------------------------------------------------- #
    if info.gpu_count >= 2 and info.nvlink_available:
        info.strategy = "ddp-nvlink"
        info.world_size = info.gpu_count
        info.notes.append(
            f"NVLink: {info.nvlink_active_links} active links @ "
            f"{info.nvlink_bandwidth_gbs} GB/s, {len(info.nvlink_pairs)} peer pairs "
            "-> NCCL DDP with P2P.")
        if not info.nvlink_fully_connected and info.gpu_count > 2:
            info.notes.append("Partial NVLink mesh; some pairs fall back to PCIe.")
    elif info.gpu_count >= 2:
        info.strategy = "ddp-pcie"
        info.world_size = info.gpu_count
        info.notes.append("Multiple GPUs, no NVLink -> NCCL DDP over PCIe.")
    elif info.gpu_count == 1:
        info.strategy = "single-gpu-islands"
        info.world_size = 1
        info.notes.append("Single GPU -> concurrent islands share the device.")
    else:
        info.strategy = "cpu-islands"
        info.world_size = 1
        info.notes.append("No GPU -> parallel CPU islands.")

    if info.disk_free_gb < 1.0:
        info.blockers.append(
            f"Only {info.disk_free_gb:.2f} GB of disk free. Checkpoints, the "
            "feature cache and temp files cannot be written; workers will die "
            "with ENOSPC/ENOMEM. Free space, then retry.")
    if 0 < info.shm_total_mb < 256:
        info.blockers.append(
            f"/dev/shm is only {info.shm_total_mb:.0f} MB. PyTorch worker "
            "processes need far more and fail with 'OSError: [Errno 12] Cannot "
            "allocate memory'. Restart the container with --shm-size=16g "
            "(docker) or mount an emptyDir{medium: Memory} at /dev/shm (k8s).")

    info.plan = build_plan(info, dataset_mb, max_islands_per_gpu,
                           allow_cpu_islands, force_cpu_islands)
    if info.gpu_count >= 2:
        info.world_size = info.gpu_count
    return info


def apply_nvlink_env(info: SystemInfo) -> None:
    if info.strategy == "ddp-nvlink":
        os.environ["NCCL_P2P_DISABLE"] = "0"
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        os.environ.setdefault("NCCL_DEBUG", "WARN")
    elif info.strategy == "ddp-pcie":
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ.setdefault("NCCL_DEBUG", "WARN")


def apply_perf_env(info: SystemInfo) -> None:
    """Turn on every safe accelerator knob for the detected hardware."""
    try:
        import torch
    except ImportError:
        return
    p = info.plan
    if info.cuda_available:
        torch.backends.cudnn.benchmark = True
        if p.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, p.threads_per_cpu_island)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, p.threads_per_cpu_island)))


def render(info: SystemInfo) -> str:
    p = info.plan
    L = ["=" * 74, "  COMPUTE DETECTION", "=" * 74]
    L.append(f"  CPU              : {info.cpu_model or 'unknown'}")
    L.append(f"       cores       : {info.cpu_physical} physical / "
             f"{info.cpu_logical} logical, {info.numa_nodes} NUMA node(s)")
    if info.cpu_flags:
        L.append(f"       ISA         : {' '.join(info.cpu_flags)}")
    L.append(f"  RAM              : {info.total_ram_gb} GB total, "
             f"{info.avail_ram_gb} GB available, {info.swap_gb} GB swap")
    L.append(f"  Disk free        : {info.disk_free_gb} GB"
             + ("   <-- FULL" if info.disk_free_gb < 1.0 else ""))
    L.append(f"  /dev/shm         : {info.shm_free_mb:.0f} MB free "
             f"/ {info.shm_total_mb:.0f} MB"
             + ("   <-- TOO SMALL" if 0 < info.shm_total_mb < 256 else ""))
    L.append(f"  PyTorch          : {info.torch_version}"
             + (f"  (CUDA {info.cuda_version})" if info.cuda_version else ""))
    if info.driver_version:
        L.append(f"  NVIDIA driver    : {info.driver_version}")
    L.append(f"  CUDA / NCCL      : {info.cuda_available} / {info.nccl_available}")
    L.append(f"  GPUs detected    : {info.gpu_count}")
    for g in info.gpus:
        L.append(f"      [{g.index}] {g.name}")
        L.append(f"           {g.free_mb}/{g.memory_mb} MB free, sm_{g.compute_capability}"
                 f", {g.multi_processors} SMs"
                 f"{', TF32+BF16' if g.supports_bf16 else ''}")
    L.append(f"  NVLink           : {'YES' if info.nvlink_available else 'NO'}")
    if info.nvlink_available:
        L.append(f"      links        : {info.nvlink_active_links} @ "
                 f"{info.nvlink_bandwidth_gbs} GB/s")
        L.append(f"      peer pairs   : {info.nvlink_pairs}")
        L.append(f"      full mesh    : {info.nvlink_fully_connected}")
    if info.p2p_matrix:
        L.append(f"      P2P matrix   : "
                 + " ".join("".join("1" if c else "0" for c in row)
                            for row in info.p2p_matrix))
    L.append("-" * 74)
    L.append(f"  STRATEGY         : {info.strategy}")
    L.append(f"  EXECUTION PLAN")
    L.append(f"      GPU islands  : {p.gpu_islands}"
             + (f"  ({p.islands_per_gpu} per GPU)" if p.gpu_islands else ""))
    L.append(f"      CPU islands  : {p.cpu_islands}"
             + (f"  ({p.threads_per_cpu_island} thread(s) each)"
                if p.cpu_islands else ""))
    L.append(f"      TOTAL        : {p.total_islands} concurrent training instances")
    L.append(f"      precision    : {p.precision}"
             + ("  (TF32 matmul on)" if p.use_tf32 else ""))
    L.append(f"      dataset      : {p.data_placement}  ({p.dataset_mb:.0f} MB)")
    if p.gpu_mem_budget_mb:
        L.append(f"      VRAM/island  : {p.gpu_mem_budget_mb} MB activation budget")
    if p.cpu_mem_budget_mb:
        L.append(f"      RAM/island   : {p.cpu_mem_budget_mb} MB activation budget")
    for n in info.notes + p.notes:
        L.append(f"    * {n}")
    if info.blockers:
        L.append("-" * 74)
        L.append("  BLOCKERS")
        for b in info.blockers:
            L.append(f"    !! {b}")
    L.append("=" * 74)
    if info.topology_matrix:
        L.append("\nnvidia-smi topo -m:\n" + info.topology_matrix)
    return "\n".join(L)


if __name__ == "__main__":
    i = detect()
    print(render(i))
    devs = ([f"cuda:{g.index}" for g in i.gpus] or []) + ["cpu"]
    print("\ncalibrating device throughput (tiny MLP, steps/s)...")
    for d, s in calibrate(devs).items():
        print(f"    {d:<8} {s:>8.1f} steps/s")
    logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, "system_info.json"), "w") as f:
        json.dump(i.to_dict(), f, indent=2)
    print(f"\nwrote {os.path.normpath(os.path.join(logs, 'system_info.json'))}")
