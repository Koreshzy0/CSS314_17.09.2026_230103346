import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103346
N = 20_000_000
P_PARALLEL = 0.95       
REPEATS = 7            
WARMUP = 2              


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


def amdahl(p: int, parallel_fraction: float = P_PARALLEL) -> float:
    """Theoretical max speedup: S(p) = 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p)


def parallel_sum(data: np.ndarray, workers: int, pool: ThreadPoolExecutor) -> float:
    """Split the array into `workers` chunks, sum each in a thread, combine."""
    bounds = np.linspace(0, data.size, workers + 1, dtype=np.int64)
    chunks = [data[bounds[i]:bounds[i + 1]] for i in range(workers)]
    # np.sum releases the GIL, so threads run truly in parallel
    partials = pool.map(np.sum, chunks)
    return float(sum(partials))


def benchmark(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Return (median wall time in ms, computed sum) for a worker count."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(WARMUP):
            parallel_sum(data, workers, pool)
        times, result = [], 0.0
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = parallel_sum(data, workers, pool)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), result


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / CPU      : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores : {max_cores}")
    print(f"Array         : {N:,} float64 elements, seed={STUDENT_ID}")
    print(f"Method        : ThreadPoolExecutor + np.sum per chunk, "
          f"median of {REPEATS} runs, P = {P_PARALLEL}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    reference = float(np.sum(data))

    results = []
    for p in worker_counts:
        t_ms, total = benchmark(data, p)
        assert abs(total - reference) < 1e-6 * abs(reference), "sum mismatch"
        results.append((p, t_ms))

    t1 = results[0][1]   

    print()
    header = f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for p, t_ms in results:
        print(f"{p:>6} | {t_ms:>15.3f} | {t1 / t_ms:>7.2f}x | {amdahl(p):>17.2f}x")

    print()
    print(f"Sum (verified for all worker counts): {reference:.6f}")
    print(f"Amdahl limit as p -> infinity       : {1 / (1 - P_PARALLEL):.2f}x")


if __name__ == "__main__":
    main()