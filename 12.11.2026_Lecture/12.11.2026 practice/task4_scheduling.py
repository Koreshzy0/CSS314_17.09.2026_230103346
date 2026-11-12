
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103346
P = 8

N = 1600
CHUNK_SIZES = [1, 8, 32, 64]

WARMUP = 1
RUNS = 3


def calculate_work(index):
    """
    Skewed workload:
    larger indexes require more iterations.
    """
    iterations = 40 + index // 2

    x = 0.5 + index * 0.0001

    for _ in range(iterations):
        x = math.sin(x + 0.00001) + math.sqrt(abs(x) + 1.0)

    return x


def worker(
    worker_id,
    mode,
    shm_name,
    n,
    p,
    chunk_size,
    counter,
    active_times
):
    shm = shared_memory.SharedMemory(name=shm_name)
    output = shm.buf.cast("d")

    start_time = time.perf_counter()

    def process_item(index):
        output[index] = calculate_work(index)

    if mode == "contiguous":
        start = worker_id * n // p
        end = (worker_id + 1) * n // p

        for index in range(start, end):
            process_item(index)

    elif mode == "cyclic":
        for index in range(worker_id, n, p):
            process_item(index)

    elif mode == "dynamic":
        while True:
            with counter.get_lock():
                start = counter.value
                counter.value += chunk_size

            if start >= n:
                break

            end = min(start + chunk_size, n)

            for index in range(start, end):
                process_item(index)

    active_times[worker_id] = time.perf_counter() - start_time

    output.release()
    shm.close()


def run_trial(mode, chunk_size=1):
    shm = shared_memory.SharedMemory(
        create=True,
        size=N * 8
    )

    counter = mp.Value("i", 0)
    active_times = mp.Array("d", P, lock=False)

    processes = []

    try:
        for worker_id in range(P):
            process = mp.Process(
                target=worker,
                args=(
                    worker_id,
                    mode,
                    shm.name,
                    N,
                    P,
                    chunk_size,
                    counter,
                    active_times
                )
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker process failed: exit code {process.exitcode}"
                )

        times = list(active_times)

        return times

    finally:
        shm.close()
        shm.unlink()


def benchmark_strategy(mode, chunk_size=1):
    last_times = []

    def benchmark_function():
        nonlocal last_times
        last_times = run_trial(mode, chunk_size)

    stats = run_benchmark(
        benchmark_function,
        warmup=WARMUP,
        runs=RUNS
    )

    max_active = max(last_times)
    min_active = min(last_times)

    imbalance = (
        max_active / min_active
        if min_active > 0
        else float("inf")
    )

    return stats, last_times, imbalance


def verify_results():
    """
    Run all strategies once and compare their output arrays.
    """
    results = {}

    for mode, chunk_size in [
        ("contiguous", 1),
        ("cyclic", 1),
        ("dynamic", 8)
    ]:
        shm = shared_memory.SharedMemory(
            create=True,
            size=N * 8
        )

        counter = mp.Value("i", 0)
        active_times = mp.Array("d", P, lock=False)

        processes = []

        try:
            for worker_id in range(P):
                process = mp.Process(
                    target=worker,
                    args=(
                        worker_id,
                        mode,
                        shm.name,
                        N,
                        P,
                        chunk_size,
                        counter,
                        active_times
                    )
                )
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            for process in processes:
                if process.exitcode != 0:
                    raise RuntimeError("Verification worker failed")

            output = shm.buf.cast("d")
            results[mode] = list(output)
            output.release()

        finally:
            shm.close()
            shm.unlink()

    reference = results["contiguous"]

    max_error = 0.0

    for mode in ("cyclic", "dynamic"):
        for i in range(N):
            error = abs(reference[i] - results[mode][i])
            max_error = max(max_error, error)

    print("\n========== NUMERICAL VERIFICATION ==========")
    print(f"Maximum absolute error: {max_error:.12e}")

    if max_error > 1e-12:
        raise RuntimeError("Numerical verification FAILED")

    print("Numerical verification PASSED")


def main():
    print("=" * 65)
    print("PHASE 4: SCHEDULING AND LOAD BALANCING")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Processes: {P}")
    print(f"Number of tasks: {N}")
    print("Workload: increasing cost with task index")

    results = {}

    print("\n========== CONTIGUOUS STATIC SCHEDULING ==========")

    stats, worker_times, imbalance = benchmark_strategy(
        "contiguous"
    )

    results["contiguous"] = stats.median_ms

    print(f"Median: {stats.median_ms:.3f} ms")
    print(f"IQR: {stats.iqr_ms:.3f} ms")
    print("Worker active times (seconds):")
    print([round(t, 4) for t in worker_times])
    print(f"Load imbalance ratio: {imbalance:.3f}")

    print("\n========== CYCLIC STATIC SCHEDULING ==========")

    stats, worker_times, imbalance = benchmark_strategy(
        "cyclic"
    )

    results["cyclic"] = stats.median_ms

    print(f"Median: {stats.median_ms:.3f} ms")
    print(f"IQR: {stats.iqr_ms:.3f} ms")
    print("Worker active times (seconds):")
    print([round(t, 4) for t in worker_times])
    print(f"Load imbalance ratio: {imbalance:.3f}")

    print("\n========== DYNAMIC CHUNK SCHEDULING ==========")

    dynamic_results = {}

    for chunk in CHUNK_SIZES:
        print(f"\nChunk size: {chunk}")

        stats, worker_times, imbalance = benchmark_strategy(
            "dynamic",
            chunk
        )

        dynamic_results[chunk] = stats.median_ms

        print(f"Median: {stats.median_ms:.3f} ms")
        print(f"IQR: {stats.iqr_ms:.3f} ms")
        print("Worker active times (seconds):")
        print([round(t, 4) for t in worker_times])
        print(f"Load imbalance ratio: {imbalance:.3f}")

    best_chunk = min(
        dynamic_results,
        key=dynamic_results.get
    )

    best_dynamic = dynamic_results[best_chunk]

    print("\n========== PERFORMANCE SUMMARY ==========")

    print(f"Contiguous static: {results['contiguous']:.3f} ms")
    print(f"Cyclic static:     {results['cyclic']:.3f} ms")

    for chunk in CHUNK_SIZES:
        print(
            f"Dynamic chunk {chunk:2d}: "
            f"{dynamic_results[chunk]:.3f} ms"
        )

    print(f"\nBest dynamic chunk size: {best_chunk}")
    print(f"Best dynamic time: {best_dynamic:.3f} ms")

    verify_results()

    print("\n========== VERIFICATION HASH ==========")

    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            results["contiguous"],
            best_dynamic
        )
    )

    print("\nPHASE 4 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
