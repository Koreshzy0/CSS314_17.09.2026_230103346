import hashlib
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103346
TASKS = 10_000
BLOCK_BYTES = 1024
HASH_ROUNDS = 500
REPEATS = 3
TOLERANCE = 0.03

PAYLOAD = (bytes(range(256)) * (BLOCK_BYTES // 256)) * HASH_ROUNDS


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


def task(i: int) -> int:
    h = hashlib.sha256()
    h.update((STUDENT_ID + i).to_bytes(8, "little"))
    h.update(PAYLOAD)
    return int.from_bytes(h.digest()[:8], "little")


def run_batch(pool_size: int, tasks: int):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        checksum = 0
        for value in pool.map(task, range(tasks), chunksize=1):
            checksum = (checksum + value) & 0xFFFFFFFFFFFFFFFF
    return (time.perf_counter() - start) * 1000.0, checksum


def median_run(pool_size: int):
    times, checksum = [], 0
    for _ in range(REPEATS):
        ms, checksum = run_batch(pool_size, TASKS)
        times.append(ms)
    return statistics.median(times), checksum


def pool_sizes(cores: int):
    raw = [1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256]
    seen, ordered = set(), []
    for p in raw:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return sorted(ordered)


def main() -> None:
    cores = os.cpu_count() or 1
    sizes = pool_sizes(cores)

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / CPU      : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores : {cores}")
    print(f"Batch         : {TASKS:,} tasks, each hashes {HASH_ROUNDS} x {BLOCK_BYTES} B "
          f"({len(PAYLOAD) // 1024} KB) with SHA-256")
    print(f"Pool sizes    : {sizes}")
    print(f"Timing        : median of {REPEATS} runs, pool creation and shutdown included")
    print("Note          : each task hashes one large buffer per call, so hashlib releases the GIL")
    print()

    run_batch(max(sizes[:3]), 500)

    results = []
    reference = None
    for p in sizes:
        ms, checksum = median_run(p)
        if reference is None:
            reference = checksum
        elif checksum != reference:
            raise RuntimeError(f"checksum mismatch at pool size {p}")
        results.append((p, ms))

    t1 = results[0][1]
    header = f"{'Pool Size':>10} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status"
    print(header)
    print("-" * (len(header) + 40))

    degradation_at = None
    best_p, best_t = min(results, key=lambda r: r[1])
    prev_t = None
    for p, ms in results:
        if prev_t is None:
            status = "baseline"
        elif ms > prev_t * (1 + TOLERANCE):
            if degradation_at is None:
                degradation_at = p
                status = "<-- WORSENING STARTS (oversubscription / context switching)"
            else:
                status = "worse"
        elif ms < prev_t * (1 - TOLERANCE):
            status = "improving"
        else:
            status = "plateau"
        if p == best_p:
            status = "BEST | " + status
        print(f"{p:>10} | {ms:>19.3f} | {t1 / ms:>18.2f}x | {status}")
        prev_t = ms

    print()
    print(f"Fastest pool size       : {best_p} threads ({best_t:.3f} ms, {t1 / best_t:.2f}x vs 1 thread)")
    if degradation_at is not None:
        print(f"Execution time starts worsening at pool size {degradation_at} "
              f"(more than {TOLERANCE:.0%} slower than the previous size).")
    else:
        print(f"No pool size was more than {TOLERANCE:.0%} slower than the previous one; "
              f"oversubscription up to {sizes[-1]} threads did not measurably degrade this workload.")
    print(f"Checksums identical for all pool sizes: {reference:#018x}")


if __name__ == "__main__":
    main()
