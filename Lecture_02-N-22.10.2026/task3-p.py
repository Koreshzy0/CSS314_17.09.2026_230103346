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
N = 2_000_000
REPEATS = 9


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


def serial_scan(data: np.ndarray, out: np.ndarray) -> None:
    np.cumsum(data, out=out)


def local_scan(data: np.ndarray, out: np.ndarray, lo: int, hi: int) -> int:
    np.cumsum(data[lo:hi], out=out[lo:hi])
    return int(out[hi - 1])


def apply_offset(out: np.ndarray, lo: int, hi: int, offset: int) -> None:
    if offset:
        out[lo:hi] += offset


def parallel_scan(data: np.ndarray, out: np.ndarray, pool: ThreadPoolExecutor, workers: int) -> None:
    bounds = np.linspace(0, data.size, workers + 1, dtype=np.int64)
    ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]

    futures = [pool.submit(local_scan, data, out, lo, hi) for lo, hi in ranges]
    chunk_totals = [f.result() for f in futures]

    offsets = [0] * workers
    for i in range(1, workers):
        offsets[i] = offsets[i - 1] + chunk_totals[i - 1]

    futures = [pool.submit(apply_offset, out, lo, hi, offsets[i])
               for i, (lo, hi) in enumerate(ranges)]
    for f in futures:
        f.result()


def median_ms(fn) -> float:
    fn()
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(times)


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else physical_cores()

    print(f"Student ID      : {STUDENT_ID}")
    print(f"OS / CPU        : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores   : {os.cpu_count()}")
    print(f"Workers used    : {workers}")
    print(f"Array           : {N:,} int64 values in [1, 10], seed={STUDENT_ID}")
    print(f"Method          : 2-pass chunked scan (local scans, offset reduction, apply offsets)")
    print(f"Timing          : median of {REPEATS} runs")
    print()

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=N, dtype=np.int64)
    out_serial = np.empty(N, dtype=np.int64)
    out_parallel = np.empty(N, dtype=np.int64)

    t_serial = median_ms(lambda: serial_scan(data, out_serial))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        t_parallel = median_ms(lambda: parallel_scan(data, out_parallel, pool, workers))

    last_ok = int(out_parallel[-1]) == int(out_serial[-1])
    full_ok = bool(np.array_equal(out_parallel, out_serial))
    passed = last_ok and full_ok

    print(f"Serial last element   : {int(out_serial[-1]):,}")
    print(f"Parallel last element : {int(out_parallel[-1]):,}")
    print()
    header = (f"{'Serial Time (ms)':>17} | {'Parallel Time (ms)':>19} | "
              f"{'Speedup':>8} | Verification Status")
    print(header)
    print("-" * (len(header) + 4))
    print(f"{t_serial:>17.3f} | {t_parallel:>19.3f} | {t_serial / t_parallel:>7.2f}x | "
          f"{'PASS' if passed else 'FAIL'}")

    if not passed:
        print("ERROR: parallel_output[-1] != serial_output[-1] or arrays differ", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
