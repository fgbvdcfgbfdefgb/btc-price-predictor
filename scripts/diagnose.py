"""
Pinpoint why workers die with OSError [Errno 12] Cannot allocate memory.

On a box with plenty of free RAM that error almost never means "out of RAM".
It means a process/mmap/shm resource ran out. This checks each candidate in
turn and tells you which one is broken.

    python3 scripts/diagnose.py

NOTE: every check lives inside main(). Module-level side effects would be
re-executed by every spawned child process - which is itself a common cause
of runaway resource use.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

OK, WARN, BAD = "  ok  ", " WARN ", " FAIL "


def _probe(_):
    """Runs in a spawned child: the thing that was failing for real."""
    import torch  # noqa: F401
    return os.getpid()


def main() -> int:
    issues: list[str] = []
    here = Path(__file__).resolve().parents[1]

    def line(status, label, detail=""):
        print(f"[{status}] {label:<34} {detail}")

    def mb(b):
        return b / 1024 / 1024

    print("=" * 78)
    print("  ENVIRONMENT DIAGNOSTIC")
    print("=" * 78)

    # ---- 1. disk ---------------------------------------------------------- #
    print("\n-- disk --")
    for path in [str(here), "/tmp", "/dev/shm", "/var/tmp",
                 os.path.expanduser("~")]:
        try:
            s = os.statvfs(path)
            free, total = s.f_bavail * s.f_frsize, s.f_blocks * s.f_frsize
            pct = 100 * (1 - free / total) if total else 100
            st = OK if free > 2e9 else (WARN if free > 2e8 else BAD)
            line(st, path, f"{mb(free):>9.0f} MB free / {mb(total):>9.0f} MB "
                           f"({pct:.1f}% used)")
            if free < 2e8 and path != "/dev/shm":
                issues.append(f"{path} is full ({mb(free):.0f} MB free) - "
                              "checkpoints and temp files cannot be written")
        except Exception as e:
            line(BAD, path, f"statvfs failed: {e}")

    # ---- 2. /dev/shm ------------------------------------------------------- #
    print("\n-- shared memory (/dev/shm) --")
    try:
        s = os.statvfs("/dev/shm")
        shm_total, shm_free = s.f_blocks * s.f_frsize, s.f_bavail * s.f_frsize
        st = OK if shm_total > 1e9 else (WARN if shm_total > 2.5e8 else BAD)
        line(st, "size", f"{mb(shm_total):.0f} MB total, {mb(shm_free):.0f} MB free")
        if shm_total < 2.5e8:
            issues.append(
                f"/dev/shm is only {mb(shm_total):.0f} MB. PyTorch worker "
                "processes need far more. This is THE classic cause of "
                "'OSError: [Errno 12] Cannot allocate memory' in containers.\n"
                "        docker:     docker run --shm-size=16g ...\n"
                "        kubernetes: emptyDir {medium: Memory} mounted at /dev/shm")
    except Exception as e:
        line(BAD, "/dev/shm", str(e))

    # ---- 3. kernel limits -------------------------------------------------- #
    print("\n-- kernel limits --")
    for f, name, want in [
        ("/proc/sys/vm/overcommit_memory", "vm.overcommit_memory", "0 or 1"),
        ("/proc/sys/vm/max_map_count", "vm.max_map_count", ">= 262144"),
        ("/proc/sys/kernel/threads-max", "kernel.threads-max", "large"),
        ("/proc/sys/kernel/pid_max", "kernel.pid_max", "large"),
    ]:
        try:
            v = Path(f).read_text().strip()
            bad = False
            if "overcommit" in name and v == "2":
                bad = True
                issues.append("vm.overcommit_memory=2 forbids over-committing, "
                              "so large fork/spawn fails with ENOMEM. "
                              "sysctl -w vm.overcommit_memory=0")
            if "max_map_count" in name and int(v) < 262144:
                bad = True
                issues.append(
                    f"vm.max_map_count={v} is low. Each CUDA process makes tens "
                    "of thousands of mappings, so many concurrent workers hit "
                    "ENOMEM. sysctl -w vm.max_map_count=262144")
            line(BAD if bad else OK, name, f"{v}   (want {want})")
        except Exception:
            line(WARN, name, "unreadable")

    # ---- 4. ulimits --------------------------------------------------------- #
    print("\n-- ulimits --")
    try:
        import resource
        fmt = lambda v: ("unlimited" if v == resource.RLIM_INFINITY else f"{v:,}")
        for res, name in [
            (resource.RLIMIT_AS, "RLIMIT_AS (virtual mem)"),
            (resource.RLIMIT_DATA, "RLIMIT_DATA"),
            (resource.RLIMIT_NPROC, "RLIMIT_NPROC (processes)"),
            (resource.RLIMIT_NOFILE, "RLIMIT_NOFILE (fds)"),
            (resource.RLIMIT_STACK, "RLIMIT_STACK"),
            (resource.RLIMIT_MEMLOCK, "RLIMIT_MEMLOCK"),
        ]:
            soft, hard = resource.getrlimit(res)
            bad = (res in (resource.RLIMIT_AS, resource.RLIMIT_DATA)
                   and soft != resource.RLIM_INFINITY and soft < 8 * 1024 ** 3)
            if bad:
                issues.append(f"{name} soft limit is {fmt(soft)} - far too small "
                              "for CUDA, which reserves tens of GB of address "
                              "space. ulimit -v unlimited")
            line(BAD if bad else OK, name, f"soft={fmt(soft)} hard={fmt(hard)}")
    except Exception as e:
        line(WARN, "ulimits", str(e))

    # ---- 5. memory ----------------------------------------------------------- #
    print("\n-- memory --")
    try:
        mi = Path("/proc/meminfo").read_text()
        g = lambda k: int([l for l in mi.splitlines()
                           if l.startswith(k + ":")][0].split()[1]) / 1024
        for k in ("MemTotal", "MemAvailable", "SwapTotal"):
            line(OK, k, f"{g(k):,.0f} MB")
        try:
            cg = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if cg != "max":
                lim = int(cg) / 1024 / 1024
                line(WARN if lim < g("MemTotal") * 0.9 else OK,
                     "cgroup memory.max", f"{lim:,.0f} MB")
                if lim < g("MemTotal") * 0.5:
                    issues.append(f"cgroup caps memory at {lim:,.0f} MB even "
                                  f"though the host shows {g('MemTotal'):,.0f} MB")
        except Exception:
            pass
    except Exception as e:
        line(WARN, "meminfo", str(e))

    # ---- 6. write tests ------------------------------------------------------ #
    print("\n-- write tests --")
    for d in [here / "checkpoints", here / "logs",
              here / "data" / "processed" / "cache",
              Path(tempfile.gettempdir())]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / ".diag_write_test"
            p.write_bytes(b"x" * (4 * 1024 * 1024))
            p.unlink()
            line(OK, f"write 4MB to {d.name}/", str(d))
        except Exception as e:
            line(BAD, f"write to {d.name}/", f"{type(e).__name__}: {e}")
            issues.append(f"cannot write to {d}: {e}")

    # ---- 7. spawning torch workers ------------------------------------------- #
    print("\n-- process spawning (the failing operation) --")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    worked = 0
    for n in (4, 8, 16, 23):
        try:
            with ProcessPoolExecutor(max_workers=n,
                                     mp_context=mp.get_context("spawn")) as ex:
                pids = list(ex.map(_probe, range(n)))
            worked = n
            line(OK, f"spawn {n} torch workers", f"{len(set(pids))} distinct pids")
        except Exception as e:
            line(BAD, f"spawn {n} torch workers", f"{type(e).__name__}: {e}")
            issues.append(
                f"cannot spawn {n} worker processes importing torch: "
                f"{type(e).__name__}: {e}\n"
                f"        {worked} workers did succeed -> run with at most that "
                "many islands")
            break

    # ---- 8. cuda -------------------------------------------------------------- #
    print("\n-- cuda --")
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            line(OK, "torch.cuda", f"{n} device(s), CUDA {torch.version.cuda}")
            for i in range(n):
                free, total = torch.cuda.mem_get_info(i)
                line(OK, f"gpu{i} memory",
                     f"{mb(free):,.0f} / {mb(total):,.0f} MB free")
        else:
            line(WARN, "torch.cuda", "not available (CPU-only build?)")
    except Exception as e:
        line(BAD, "torch import", f"{type(e).__name__}: {e}")
        issues.append(f"torch is not importable: {e}  ->  run ./setup.sh")

    # ---- verdict ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    if issues:
        print(f"  {len(issues)} PROBLEM(S) FOUND")
        print("=" * 78)
        for i, m in enumerate(issues, 1):
            print(f"\n  {i}. {m}")
        print("\n  Conservative retry that usually works:")
        print("     ./scripts/run_multigpu.sh --islands-per-gpu 1 --no-cpu-islands")
    else:
        print("  no environment problems detected")
    print("=" * 78)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
