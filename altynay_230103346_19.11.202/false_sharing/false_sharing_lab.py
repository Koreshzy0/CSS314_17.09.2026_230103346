import os
import re
import sys
import json
import time
import hashlib
import platform
import statistics
import threading
import subprocess
import numpy as np
import numba
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Git handle (can be overridden: python3 false_sharing_lab.py <git_username>)
GIT_USERNAME = "Koreshzy0"
if len(sys.argv) > 1:
    GIT_USERNAME = sys.argv[1]

NUM_THREADS = 4
# FS_ITERS env var is only for quick smoke tests; the lab requires 60,000,000.
ITERATIONS_PER_THREAD = int(os.environ.get("FS_ITERS", 60_000_000))
REPEATS = int(os.environ.get("FS_REPEATS", 3))   # median of REPEATS runs per layout
BUFFER_ALIGN = 256                                # bytes; covers 64 B and 128 B cache lines
STRIDES = {
    "A_contiguous": 1,
    "B_half_line": 4,
    "C_line_padded": 8,
    "D_prefetch_safe": 16,
}
MUL = 2654435761


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
            return int(out.strip())
        except Exception:
            return 64
    return 64


def get_cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def pick_physical_cores(n):
    """Linux only: choose n logical CPUs that sit on distinct physical cores."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    avail = sorted(os.sched_getaffinity(0))
    chosen, seen = [], set()
    for c in avail:
        key = c
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                key = f.read().strip()
        except OSError:
            pass
        if key in seen:
            continue
        seen.add(key)
        chosen.append(c)
        if len(chosen) == n:
            return chosen
    return avail[:n] if len(avail) >= n else None


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = get_cpu_model()
    cpu_sig = f"{cpu_model}::{platform.machine()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")
    affinity_cores = pick_physical_cores(NUM_THREADS) if affinity_supported else None

    return {
        "git_username": git_user.strip(),
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_sig": cpu_sig,
        "seed": instance_seed,
        "affinity_supported": affinity_supported,
        "affinity_cores": affinity_cores,
    }


def pin_thread(ctx, t):
    if ctx["affinity_cores"] is not None:
        try:
            os.sched_setaffinity(0, {ctx["affinity_cores"][t]})  # pid 0 = calling thread on Linux
        except OSError:
            pass


def thread_seed(instance_seed: int, t: int) -> np.uint64:
    return np.uint64((instance_seed ^ ((t + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF)


def aligned_uint64_array(n_elems: int, align: int = BUFFER_ALIGN) -> np.ndarray:
    raw = np.zeros(n_elems + align // 8, dtype=np.uint64)
    off = ((-raw.ctypes.data) % align) // 8
    arr = raw[off:off + n_elems]
    assert arr.ctypes.data % align == 0
    return arr


# ==============================================================================
# TASK 2: GIL-free native counter kernel (real RMW on buffer[slot] every iteration)
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    mul = np.uint64(2654435761)
    mask = np.uint64(0xFE)
    one = np.uint64(1)
    idx = slot
    for i in range(iters):
        v = buffer[idx]                                   # load from memory
        inc = ((np.uint64(i) * mul) ^ seed) & mask        # always even
        nv = v + inc
        buffer[idx] = nv                                  # store to memory
        # Loop-carried address dependency: the next address depends on the value just
        # computed. nv is always even at runtime, so idx == slot, but the compiler cannot
        # prove it -> it cannot promote buffer[slot] into a register.
        idx = slot + np.int64(nv & one)


# ==============================================================================
# TASK 4: Thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    mul = np.uint64(2654435761)
    mask = np.uint64(0xFE)
    acc = np.uint64(0)
    for i in range(iters):
        acc += ((np.uint64(i) * mul) ^ seed) & mask
    buffer[slot] = buffer[slot] + acc                     # single shared-memory write


def read_ctx_switches():
    if resource is None:
        return (0, 0)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (ru.ru_nvcsw, ru.ru_nivcsw)


def run_config(ctx, kernel, stride, iters):
    n = NUM_THREADS * stride + 16                         # +16: slack for the idx = slot + 0/1 access
    buf = aligned_uint64_array(n)
    seeds = [thread_seed(ctx["seed"], t) for t in range(NUM_THREADS)]
    barrier = threading.Barrier(NUM_THREADS)
    t_start = [0] * NUM_THREADS
    t_end = [0] * NUM_THREADS
    errors = []

    def worker(t):
        try:
            pin_thread(ctx, t)
            barrier.wait()
            a = time.perf_counter_ns()
            kernel(buf, t * stride, iters, seeds[t])
            b = time.perf_counter_ns()
            t_start[t], t_end[t] = a, b
        except Exception as exc:  # noqa
            errors.append(repr(exc))
            barrier.abort()

    cs0 = read_ctx_switches()
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(NUM_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    cs1 = read_ctx_switches()
    if errors:
        raise RuntimeError(f"worker failure: {errors}")

    slots = [t * stride for t in range(NUM_THREADS)]
    finals = [int(buf[s]) for s in slots]
    nonzero_total = int(np.count_nonzero(buf))
    nonzero_slots = sum(1 for s in slots if buf[s] != 0)
    return {
        "elapsed_ns": max(t_end) - min(t_start),
        "per_thread_ns": [t_end[t] - t_start[t] for t in range(NUM_THREADS)],
        "vol_ctx_switches": cs1[0] - cs0[0],
        "invol_ctx_switches": cs1[1] - cs0[1],
        "finals": finals,
        "stray_nonzero": nonzero_total - nonzero_slots,
    }


def lines_touched(stride, line_size):
    """Distinct cache lines used by the 4 counters (buffer base aligned)."""
    return len({(8 * t * stride) // line_size for t in range(NUM_THREADS)})


# ==============================================================================
# TASK 3: Multi-threaded stride sweep (+ privatized baseline)
# ==============================================================================
def run_stride_benchmark(ctx):
    # JIT warm-up (excluded from timing)
    warm = np.zeros(64, dtype=np.uint64)
    worker_strided_accumulate(warm, 0, 1000, np.uint64(1))
    worker_private_accumulate(warm, 0, 1000, np.uint64(1))

    configs = [(name, worker_strided_accumulate, s) for name, s in STRIDES.items()]
    configs.append(("private_baseline", worker_private_accumulate, 8))

    runs = {name: [] for name, _, _ in configs}
    for r in range(REPEATS):
        for name, kern, stride in configs:
            res = run_config(ctx, kern, stride, ITERATIONS_PER_THREAD)
            runs[name].append(res)
            print(f"  repeat {r + 1}/{REPEATS}  {name:18s} {res['elapsed_ns'] / 1e6:10.1f} ms", flush=True)

    summary = {}
    for name, kern, stride in configs:
        lst = runs[name]
        elapsed_list = [x["elapsed_ns"] for x in lst]
        med = int(statistics.median(elapsed_list))
        rep = min(lst, key=lambda x: abs(x["elapsed_ns"] - med))
        summary[name] = {
            "stride_elements": stride,
            "stride_bytes": stride * 8,
            "elapsed_ns_median": med,
            "elapsed_ns_all_runs": elapsed_list,
            "ns_per_update": med / ITERATIONS_PER_THREAD,
            "total_updates": NUM_THREADS * ITERATIONS_PER_THREAD,
            "vol_ctx_switches_median_run": rep["vol_ctx_switches"],
            "invol_ctx_switches_median_run": rep["invol_ctx_switches"],
            "lines_touched_64B": lines_touched(stride, 64),
            "lines_touched_detected": lines_touched(stride, ctx["cache_line_size"]),
            "finals": rep["finals"],
        }
    return summary, runs


# ==============================================================================
# TASK 5: Exact state parity & LLVM IR verification
# ==============================================================================
def audit_llvm_and_parity(ctx, summary, runs):
    # --- bit-exact parity across every run of every layout and the private baseline
    reference = runs["private_baseline"][0]["finals"]
    parity = True
    for name, lst in runs.items():
        for res in lst:
            if res["finals"] != reference or res["stray_nonzero"] != 0:
                parity = False

    # --- LLVM IR audit of the compiled kernel body (CPython wrapper excluded)
    ir = worker_strided_accumulate.inspect_llvm()
    if isinstance(ir, dict):
        ir = "\n".join(ir.values())
    funcs = re.split(r"(?m)^define ", ir)[1:]
    core = "\n".join(f for f in funcs if "cpython" not in f.split("{", 1)[0].lower())
    store_i64 = len(re.findall(r"\bstore i64\b", core))
    load_i64 = len(re.findall(r"\bload i64\b", core))
    py_fallback = any(tok in core for tok in ("PyObject", "PyEval", "Py_DecRef", "PyErr"))
    llvm_ok = store_i64 > 0 and load_i64 > 0 and not py_fallback

    return {
        "parity_bit_exact": parity,
        "reference_finals": reference,
        "llvm": {
            "store_i64": store_i64,
            "load_i64": load_i64,
            "python_object_fallback": py_fallback,
            "audit_passed": llvm_ok,
        },
        "verification_passed": bool(parity and llvm_ok),
    }


def gil_demo(total=8_000_000):
    """Pure-Python threads (no nogil): same total work, 1 thread vs 4 threads."""
    def py_worker(buf, slot, n):
        for _ in range(n):
            buf[slot] += 1

    def run(nthreads):
        buf = [0] * (nthreads * 8 + 8)
        per = total // nthreads
        ths = [threading.Thread(target=py_worker, args=(buf, t, per)) for t in range(nthreads)]
        t0 = time.perf_counter_ns()
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        return time.perf_counter_ns() - t0

    t1 = run(1)
    t4 = run(4)
    return {"total_updates": total, "one_thread_ns": t1, "four_threads_ns": t4,
            "ratio_4_over_1": t4 / t1}


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"GIT_USERNAME      : {ctx['git_username']}")
    print(f"CPU model         : {ctx['cpu_model']}")
    print(f"Logical cores     : {ctx['logical_cores']}")
    print(f"Cache line size   : {ctx['cache_line_size']} B (detected)")
    print(f"INSTANCE_SEED     : {ctx['seed']}")
    print(f"Affinity pinning  : {'supported ' + str(ctx['affinity_cores']) if ctx['affinity_supported'] else 'NOT supported on this OS'}")
    print(f"Iterations/thread : {ITERATIONS_PER_THREAD:,}  threads={NUM_THREADS}  repeats={REPEATS}\n")

    summary, runs = run_stride_benchmark(ctx)
    audit = audit_llvm_and_parity(ctx, summary, runs)
    gil = gil_demo()

    s1 = summary["A_contiguous"]["elapsed_ns_median"]
    s4 = summary["B_half_line"]["elapsed_ns_median"]
    s8 = summary["C_line_padded"]["elapsed_ns_median"]
    s16 = summary["D_prefetch_safe"]["elapsed_ns_median"]
    pv = summary["private_baseline"]["elapsed_ns_median"]

    print("\n--- Stride sweep (median of runs, wall-clock for 4 concurrent threads) ---")
    for name, d in summary.items():
        print(f"{name:18s} stride={d['stride_elements']:2d} ({d['stride_bytes']:3d} B)  "
              f"{d['elapsed_ns_median']:>14,d} ns  {d['ns_per_update']:8.3f} ns/update  "
              f"lines64={d['lines_touched_64B']} lines{ctx['cache_line_size']}={d['lines_touched_detected']}  "
              f"vcsw={d['vol_ctx_switches_median_run']} ivcsw={d['invol_ctx_switches_median_run']}")
    print(f"\nslowdown stride1 / stride8   = {s1 / s8:.3f}x")
    print(f"slowdown stride1 / stride16  = {s1 / s16:.3f}x")
    print(f"slowdown stride4 / stride8   = {s4 / s8:.3f}x")
    print(f"ratio    stride8 / stride16  = {s8 / s16:.3f}x")
    print(f"ratio    stride8 / private   = {s8 / pv:.3f}x")
    print(f"ratio    stride1 / private   = {s1 / pv:.3f}x")
    print(f"\nparity_bit_exact = {audit['parity_bit_exact']}  final counters = {audit['reference_finals']}")
    print(f"LLVM: store i64 = {audit['llvm']['store_i64']}, load i64 = {audit['llvm']['load_i64']}, "
          f"python_object_fallback = {audit['llvm']['python_object_fallback']}, "
          f"audit_passed = {audit['llvm']['audit_passed']}")
    print(f"GIL demo (pure Python, {gil['total_updates']:,} updates): 1 thread = {gil['one_thread_ns']:,} ns, "
          f"4 threads = {gil['four_threads_ns']:,} ns, ratio = {gil['ratio_4_over_1']:.3f}")
    print(f"verification_passed = {audit['verification_passed']}")

    manifest = {
        "git_username": ctx["git_username"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size_bytes": ctx["cache_line_size"],
        "instance_seed": ctx["seed"],
        "affinity_supported": ctx["affinity_supported"],
        "affinity_cores": ctx["affinity_cores"],
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numba": numba.__version__,
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "results": summary,
        "slowdown_stride1_vs_stride8": s1 / s8,
        "slowdown_stride1_vs_stride16": s1 / s16,
        "ratio_stride8_vs_stride16": s8 / s16,
        "ratio_stride8_vs_private": s8 / pv,
        "ratio_stride1_vs_private": s1 / pv,
        "parity_bit_exact": audit["parity_bit_exact"],
        "reference_finals": audit["reference_finals"],
        "llvm_audit": audit["llvm"],
        "gil_demo": gil,
        "verification_passed": audit["verification_passed"],
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nWrote cache_lab_manifest.json")
    if not audit["verification_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
