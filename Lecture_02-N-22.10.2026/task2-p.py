import math
import os
import platform
import random
import threading
import time

STUDENT_ID = 230103346
THREADS = 4
TOTAL_POINTS = 40_000_000
WARMUP_POINTS = 1_000_000


def run_shared(total_points: int):
    rng = random.Random(STUDENT_ID * 10)
    lock = threading.Lock()
    per_thread = total_points // THREADS
    inside = [0] * THREADS
    go = threading.Event()

    def worker(idx: int) -> None:
        rand = rng.random
        go.wait()
        hits = 0
        for _ in range(per_thread):
            with lock:
                x = rand()
                y = rand()
            if x * x + y * y <= 1.0:
                hits += 1
        inside[idx] = hits

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
    for t in threads:
        t.start()
    start = time.perf_counter()
    go.set()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - start) * 1000.0
    return 4.0 * sum(inside) / (per_thread * THREADS), ms


def run_independent(total_points: int):
    per_thread = total_points // THREADS
    inside = [0] * THREADS
    go = threading.Event()

    def worker(idx: int) -> None:
        rand = random.Random(STUDENT_ID * 10 + idx).random
        go.wait()
        hits = 0
        for _ in range(per_thread):
            x = rand()
            y = rand()
            if x * x + y * y <= 1.0:
                hits += 1
        inside[idx] = hits

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
    for t in threads:
        t.start()
    start = time.perf_counter()
    go.set()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - start) * 1000.0
    return 4.0 * sum(inside) / (per_thread * THREADS), ms


def main() -> None:
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Python           : {platform.python_version()}")
    print(f"Logical cores    : {os.cpu_count()}")
    print(f"Threads          : {THREADS}")
    print(f"Sample points    : {TOTAL_POINTS:,} ({TOTAL_POINTS // THREADS:,} per thread)")
    seeds = ", ".join(str(STUDENT_ID * 10 + t) for t in range(THREADS))
    print(f"Independent seeds: {seeds}")
    print()

    run_shared(WARMUP_POINTS)
    run_independent(WARMUP_POINTS)

    pi_s, ms_s = run_shared(TOTAL_POINTS)
    pi_i, ms_i = run_independent(TOTAL_POINTS)

    print("Shared PRNG (one random.Random guarded by threading.Lock):")
    print(f"  Computed pi    : {pi_s:.8f}")
    print(f"  Absolute error : {abs(pi_s - math.pi):.8f}")
    print(f"  Wall time      : {ms_s:.3f} ms")
    print()
    print("Independent PRNG (one random.Random per thread, no locking):")
    print(f"  Computed pi    : {pi_i:.8f}")
    print(f"  Absolute error : {abs(pi_i - math.pi):.8f}")
    print(f"  Wall time      : {ms_i:.3f} ms")
    print()
    print(f"Speedup from eliminating generator locking: {ms_s / ms_i:.2f}x")


if __name__ == "__main__":
    main()
