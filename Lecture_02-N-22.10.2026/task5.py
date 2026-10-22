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
A = 2.5
B = 1.5
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
FINE_POINTS = 16


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


def serial(x: np.ndarray, y: np.ndarray) -> None:
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def kernel(x: np.ndarray, y: np.ndarray, lo: int, hi: int) -> None:
    xs = x[lo:hi]
    ys = y[lo:hi]
    np.multiply(xs, A, out=ys)
    np.add(ys, B, out=ys)


def parallel(x: np.ndarray, y: np.ndarray, pool: ThreadPoolExecutor, workers: int) -> None:
    bounds = np.linspace(0, x.size, workers + 1, dtype=np.int64)
    futures = [pool.submit(kernel, x, y, int(bounds[i]), int(bounds[i + 1]))
               for i in range(workers)]
    for f in futures:
        f.result()


def reps_for(n: int) -> int:
    return max(7, min(3000, 3_000_000 // n))


def median_time(fn, reps: int) -> float:
    for _ in range(max(2, reps // 10)):
        fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def measure(n: int, pool: ThreadPoolExecutor, workers: int):
    rng = np.random.default_rng(STUDENT_ID)
    x = rng.random(n, dtype=np.float64)
    y_s = np.empty(n, dtype=np.float64)
    y_p = np.empty(n, dtype=np.float64)
    reps = reps_for(n)
    t_serial = median_time(lambda: serial(x, y_s), reps)
    t_par = median_time(lambda: parallel(x, y_p, pool, workers), reps)
    if not np.allclose(y_s, y_p):
        raise RuntimeError(f"result mismatch at N={n}")
    return t_serial, t_par


def find_crossover(points):
    """points: sorted list of (n, speedup). Returns (slower_n, faster_n) or a status string."""
    first_faster = None
    for idx in range(len(points)):
        if all(s > 1.0 for _, s in points[idx:]):
            first_faster = idx
            break
    if first_faster is None:
        return "never"
    if first_faster == 0:
        return "always"
    return points[first_faster - 1][0], points[first_faster][0]


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / CPU      : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Workers       : {workers} (ThreadPoolExecutor, pool created once and reused)")
    print(f"Operation     : y[i] = {A} * x[i] + {B}, float64")
    print("Timing        : median of many repetitions per size")
    print()

    header = (f"{'N':>12} | {'Serial (us)':>12} | {'Parallel (us)':>13} | "
              f"{'Speedup':>8} | Verdict")
    print(header)
    print("-" * (len(header) + 10))

    coarse = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n in SIZES:
            ts, tp = measure(n, pool, workers)
            s = ts / tp
            coarse.append((n, s))
            verdict = "parallel FASTER" if s > 1.0 else "parallel SLOWER"
            print(f"{n:>12,} | {ts * 1e6:>12.2f} | {tp * 1e6:>13.2f} | {s:>7.2f}x | {verdict}")

        result = find_crossover(coarse)
        print()
        if result == "never":
            print("Coarse result : parallel was never consistently faster than serial "
                  "for the tested sizes.")
            return
        if result == "always":
            print("Coarse result : parallel was already faster at the smallest tested N.")
            return

        lo, hi = result
        print(f"Coarse range  : crossover lies between N = {lo:,} (slower) and N = {hi:,} (faster)")
        print()
        print(f"Refinement between {lo:,} and {hi:,} ({FINE_POINTS} log-spaced points):")
        fine_header = f"{'N':>12} | {'Serial (us)':>12} | {'Parallel (us)':>13} | {'Speedup':>8}"
        print(fine_header)
        print("-" * len(fine_header))

        fine_ns = sorted({int(v) for v in np.geomspace(lo, hi, FINE_POINTS)})
        fine = []
        for n in fine_ns:
            ts, tp = measure(n, pool, workers)
            s = ts / tp
            fine.append((n, s))
            print(f"{n:>12,} | {ts * 1e6:>12.2f} | {tp * 1e6:>13.2f} | {s:>7.2f}x")

    fine_result = find_crossover(fine)
    print()
    if isinstance(fine_result, tuple):
        a, b = fine_result
        print(f"CROSSOVER RANGE: parallel becomes faster than serial for N in ({a:,}, {b:,}]")
        print(f"Below ~{a:,} elements parallel overhead dominates; from ~{b:,} upward "
              f"parallel wins on this CPU.")
    else:
        print(f"CROSSOVER RANGE: between {lo:,} and {hi:,} (refinement was noisy: {fine_result})")


if __name__ == "__main__":
    main()
