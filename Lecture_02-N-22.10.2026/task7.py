import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103346
N = 30_000_000
BLOCK = 65_536
REPEATS_A = 7
REPEATS_B = 3
STALL_EFFICIENCY = 0.50


def cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def copy_chunk(a: np.ndarray, b: np.ndarray, lo: int, hi: int) -> None:
    np.copyto(b[lo:hi], a[lo:hi])


def trig_chunk(a: np.ndarray, b: np.ndarray, lo: int, hi: int) -> None:
    t_sin = np.empty(BLOCK, dtype=np.float64)
    t_cos = np.empty(BLOCK, dtype=np.float64)
    t_tan = np.empty(BLOCK, dtype=np.float64)
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        m = e - s
        x = a[s:e]
        np.sin(x, out=t_sin[:m])
        np.cos(x, out=t_cos[:m])
        np.tan(x, out=t_tan[:m])
        np.multiply(t_sin[:m], t_cos[:m], out=t_sin[:m])
        np.add(t_sin[:m], t_tan[:m], out=b[s:e])


def run_parallel(kernel, a: np.ndarray, b: np.ndarray, pool: ThreadPoolExecutor, workers: int) -> None:
    bounds = np.linspace(0, a.size, workers + 1, dtype=np.int64)
    futures = [pool.submit(kernel, a, b, int(bounds[i]), int(bounds[i + 1]))
               for i in range(workers)]
    for f in futures:
        f.result()


def bench(kernel, a: np.ndarray, b: np.ndarray, workers: int, repeats: int) -> float:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run_parallel(kernel, a, b, pool, workers)
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            run_parallel(kernel, a, b, pool, workers)
            times.append(time.perf_counter() - t)
    return statistics.median(times)


def scaling_table(name, kernel, a, b, core_counts, repeats, reference):
    rows = []
    for p in core_counts:
        b.fill(0.0)
        t = bench(kernel, a, b, p, repeats)
        if not np.allclose(b, reference, rtol=1e-9, atol=1e-12):
            raise RuntimeError(f"{name}: result mismatch at {p} workers")
        rows.append((p, t))
    t1 = rows[0][1]
    return [(p, t, t1 / t, (t1 / t) / p) for p, t in rows]


def print_table(title, rows, bytes_moved=None):
    print(title)
    extra = f" | {'Bandwidth (GB/s)':>16}" if bytes_moved else ""
    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Efficiency':>10}{extra}"
    print(header)
    print("-" * len(header))
    for p, t, s, e in rows:
        bw = f" | {bytes_moved / t / 1e9:>16.2f}" if bytes_moved else ""
        print(f"{p:>6} | {t * 1000:>14.3f} | {s:>7.2f}x | {e:>9.1%}{bw}")
    print()


def main() -> None:
    max_cores = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)
    core_counts = sorted({1, 2, max_cores})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / CPU      : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Array         : {N:,} float64 elements ({N * 8 / 1e6:.0f} MB), seed={STUDENT_ID}")
    print(f"Core counts   : {core_counts}")
    print(f"Method        : ThreadPoolExecutor, median of {REPEATS_A}/{REPEATS_B} runs (A/B)")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random(N, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)

    ref_a = a.copy()
    ref_b = np.sin(a) * np.cos(a) + np.tan(a)

    rows_a = scaling_table("Kernel A", copy_chunk, a, b, core_counts, REPEATS_A, ref_a)
    rows_b = scaling_table("Kernel B", trig_chunk, a, b, core_counts, REPEATS_B, ref_b)

    print_table("Kernel A (memory-bound): B[i] = A[i]", rows_a, bytes_moved=2 * N * 8)
    print_table("Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])", rows_b)

    eff_a = rows_a[-1][3]
    eff_b = rows_b[-1][3]
    top = core_counts[-1]
    stalled_a = eff_a < STALL_EFFICIENCY
    stalled_b = eff_b < STALL_EFFICIENCY

    print("Verdict")
    print("-------")
    print(f"Kernel A efficiency at {top} cores: {eff_a:.1%}")
    print(f"Kernel B efficiency at {top} cores: {eff_b:.1%}")
    if stalled_a and not stalled_b:
        print("RESULT: Kernel A (vector copy) STALLED in scaling due to memory channel saturation.")
        print("        Kernel B (sin/cos/tan) kept scaling because it is limited by CPU computation.")
    elif stalled_a and stalled_b:
        print("RESULT: Both kernels fell below the efficiency threshold; "
              f"Kernel A is the more memory-bound (efficiency {eff_a:.1%} vs {eff_b:.1%})."
              if eff_a < eff_b else
              "RESULT: Both kernels fell below the efficiency threshold; "
              f"Kernel B scaled worse ({eff_b:.1%} vs {eff_a:.1%}), so memory saturation is not the only limit here.")
    elif not stalled_a and not stalled_b:
        print("RESULT: Neither kernel dropped below the stall threshold "
              f"({STALL_EFFICIENCY:.0%} efficiency) on this machine.")
    else:
        print("RESULT: Kernel B scaled worse than Kernel A on this machine; "
              "memory channel saturation was not the limiting factor for Kernel A.")
    print(f"(stall threshold: efficiency < {STALL_EFFICIENCY:.0%} at the maximum core count)")


if __name__ == "__main__":
    main()
