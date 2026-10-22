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
ROWS = COLS = 8_000
REPEATS = 5


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


def physical_cores() -> int:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        pass
    return os.cpu_count() or 1


def sum_rows(m: np.ndarray, r0: int, r1: int) -> float:
    total = 0.0
    for i in range(r0, r1):
        total += float(np.sum(m[i, :]))
    return total


def sum_cols(m: np.ndarray, c0: int, c1: int) -> float:
    total = 0.0
    for j in range(c0, c1):
        total += float(np.sum(m[:, j]))
    return total


def run_mode(fn, m: np.ndarray, pool: ThreadPoolExecutor, workers: int, extent: int) -> float:
    bounds = np.linspace(0, extent, workers + 1, dtype=np.int64)
    futures = [pool.submit(fn, m, int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
    return sum(f.result() for f in futures)


def bench(fn, m: np.ndarray, pool: ThreadPoolExecutor, workers: int, extent: int):
    run_mode(fn, m, pool, workers, extent)
    times, result = [], 0.0
    for _ in range(REPEATS):
        t = time.perf_counter()
        result = run_mode(fn, m, pool, workers, extent)
        times.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(times), result


def main() -> None:
    logical = os.cpu_count() or 1
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else physical_cores()

    print(f"Student ID      : {STUDENT_ID}")
    print(f"OS / CPU        : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores   : {logical}")
    print(f"Physical cores  : {physical_cores()}")
    print(f"Workers used    : {workers}")
    print(f"Matrix          : {ROWS:,} x {COLS:,} float64 ({ROWS * COLS * 8 / 1e6:.0f} MB), seed={STUDENT_ID}")
    print(f"Method          : ThreadPoolExecutor, median of {REPEATS} runs")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    m = rng.random((ROWS, COLS), dtype=np.float64)
    reference = float(np.sum(m))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        t_a, sum_a = bench(sum_rows, m, pool, workers, ROWS)
        t_b, sum_b = bench(sum_cols, m, pool, workers, COLS)

    if not (np.isclose(sum_a, reference, rtol=1e-9) and np.isclose(sum_b, reference, rtol=1e-9)):
        raise RuntimeError("sum mismatch between modes")

    print(f"Mode A (Row-Major, linear)    : {t_a:>10.3f} ms   sum = {sum_a:.4f}")
    print(f"Mode B (Column-Major, strided): {t_b:>10.3f} ms   sum = {sum_b:.4f}")
    print()
    print(f"Cache penalty factor (Mode B / Mode A): {t_b / t_a:.2f}x")


if __name__ == "__main__":
    main()
