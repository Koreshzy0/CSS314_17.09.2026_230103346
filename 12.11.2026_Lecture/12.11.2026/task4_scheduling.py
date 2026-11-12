"""
Phase 4: Scheduling and load balancing.

Implements:
1. Contiguous static scheduling
2. Cyclic static scheduling
3. Dynamic chunk scheduling with a synchronized shared counter

The per-item output array is stored in multiprocessing.shared_memory.
Per-worker active times are stored in mp.Array('d', p).
The dynamic queue uses mp.Value('i', 0) and its lock.
"""

import array
import json
import math
import multiprocessing as mp
import subprocess
import time
from multiprocessing import shared_memory

from harness import emit_verify_tag


STUDENT_ID = 230103346
N = 12000
CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]


def physical_cores():
    """Return the number of physical CPU cores on macOS."""
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.physicalcpu"],
            text=True
        )
        return int(output.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return mp.cpu_count()


def work_item(i):
    """
    Deterministic, non-uniform workload.

    K(i) = 1 + floor((i / 60)^2).
    Integer arithmetic below is exactly equivalent:
    floor((i / 60)^2) = floor(i^2 / 3600).
    """
    k = 1 + (i * i) // 3600
    x = (i + 1) * 0.001

    for _ in range(k):
        x = (
            math.sin(x)
            + math.cos(x)
            + math.sqrt(abs(x) + 1.0)
        )

    return x


def worker(
    worker_id,
    worker_count,
    strategy,
    chunk_size,
    counter,
    worker_times,
    shared_memory_name
):
    """Worker process: attach to shared output memory and process items."""
    shm = shared_memory.SharedMemory(name=shared_memory_name)
    output = shm.buf.cast("d")

    start_time = time.perf_counter()

    try:
        if strategy == "static":
            start = worker_id * N // worker_count
            end = (worker_id + 1) * N // worker_count

            for i in range(start, end):
                output[i] = work_item(i)

        elif strategy == "cyclic":
            for i in range(worker_id, N, worker_count):
                output[i] = work_item(i)

        elif strategy == "dynamic":
            while True:
                # Atomically reserve the next chunk of indices.
                with counter.get_lock():
                    start = counter.value
                    end = min(start + chunk_size, N)
                    counter.value = end

                if start >= N:
                    break

                for i in range(start, end):
                    output[i] = work_item(i)

        else:
            raise ValueError(f"Unknown scheduling strategy: {strategy}")

    finally:
        worker_times[worker_id] = (
            time.perf_counter() - start_time
        ) * 1000.0

        del output
        shm.close()


def run_strategy(ctx, strategy, worker_count, chunk_size=1):
    """Run one scheduling strategy and return timing and verification data."""
    # Initialize every output slot to NaN. Any unprocessed item remains NaN.
    initial_data = array.array("d", [float("nan")] * N)
    shm = shared_memory.SharedMemory(create=True, size=initial_data.itemsize * N)
    shm.buf.cast("B")[:] = initial_data.tobytes()

    counter = ctx.Value("i", 0)
    worker_times = ctx.Array("d", worker_count, lock=False)
    processes = []

    wall_start = time.perf_counter()

    try:
        for worker_id in range(worker_count):
            process = ctx.Process(
                target=worker,
                args=(
                    worker_id,
                    worker_count,
                    strategy,
                    chunk_size,
                    counter,
                    worker_times,
                    shm.name
                )
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        wall_ms = (time.perf_counter() - wall_start) * 1000.0

        failed = [
            (index, process.exitcode)
            for index, process in enumerate(processes)
            if process.exitcode != 0
        ]
        if failed:
            raise RuntimeError(f"Worker process failure(s): {failed}")

        # Read all per-item outputs in their original index order.
        values = array.array("d")
        values.frombytes(bytes(shm.buf))

        missing = [
            index for index, value in enumerate(values)
            if math.isnan(value)
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} items were not processed; "
                f"first missing indices: {missing[:10]}"
            )

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Non-finite value detected in shared results.")

        times = list(worker_times)
        average_time = sum(times) / worker_count
        imbalance = max(times) / average_time if average_time > 0 else 0.0

        # The values are read in the same item order for every strategy.
        checksum = math.fsum(values)

        return {
            "strategy": strategy,
            "chunk_size": chunk_size if strategy == "dynamic" else None,
            "wall_ms": wall_ms,
            "worker_times_ms": times,
            "imbalance": imbalance,
            "checksum": checksum,
            "processed_items": len(values)
        }

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


def print_result(label, result):
    print(f"\n{label}")
    print(f"Wall time: {result['wall_ms']:.3f} ms")
    print(
        "Worker active times (ms):",
        [round(value, 3) for value in result["worker_times_ms"]]
    )
    print(f"Load imbalance lambda: {result['imbalance']:.4f}")
    print(f"Checksum: {result['checksum']:.12f}")
    print(f"Processed items: {result['processed_items']}")


def main():
    ctx = mp.get_context()
    p = min(physical_cores(), N)

    print("===== PHASE 4: SCHEDULING AND LOAD BALANCING =====")
    print("Student ID:", STUDENT_ID)
    print("Logical CPUs:", mp.cpu_count())
    print("Physical CPUs:", p)
    print("Worker processes:", p)
    print("Number of items N:", N)
    print("Dynamic chunk sizes:", CHUNK_SIZES)
    print("K(0):", 1 + (0 * 0) // 3600)
    print("K(11999):", 1 + (11999 * 11999) // 3600)

    all_results = {}

    # Strategy 1: contiguous static scheduling.
    static = run_strategy(ctx, "static", p)
    all_results["static"] = static
    print_result("STRATEGY 1: CONTIGUOUS STATIC", static)

    # Strategy 2: cyclic static scheduling.
    cyclic = run_strategy(ctx, "cyclic", p)
    all_results["cyclic"] = cyclic
    print_result("STRATEGY 2: CYCLIC STATIC", cyclic)

    # Strategy 3: dynamic scheduling for all required chunk sizes.
    dynamic_results = {}
    for chunk_size in CHUNK_SIZES:
        result = run_strategy(
            ctx,
            "dynamic",
            p,
            chunk_size=chunk_size
        )
        dynamic_results[str(chunk_size)] = result
        all_results[f"dynamic_{chunk_size}"] = result
        print_result(
            f"STRATEGY 3: DYNAMIC QUEUE, C={chunk_size}",
            result
        )

    # Verify that every scheduling strategy produced the same item results.
    reference_checksum = static["checksum"]
    for label, result in all_results.items():
        if not math.isclose(
            result["checksum"],
            reference_checksum,
            rel_tol=1e-12,
            abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Checksum mismatch for {label}: "
                f"{result['checksum']} != {reference_checksum}"
            )

    # Select C* using the lowest measured dynamic wall-clock time.
    best_chunk = min(
        CHUNK_SIZES,
        key=lambda c: dynamic_results[str(c)]["wall_ms"]
    )
    best_dynamic = dynamic_results[str(best_chunk)]

    print("\n===== PHASE 4 SUMMARY =====")
    print(f"Contiguous static wall time: {static['wall_ms']:.3f} ms")
    print(f"Cyclic static wall time: {cyclic['wall_ms']:.3f} ms")
    print(f"Best dynamic chunk C*: {best_chunk}")
    print(f"Best dynamic wall time: {best_dynamic['wall_ms']:.3f} ms")
    print("All scheduling strategies passed checksum verification.")

    print("\n===== PHASE 4 VERIFICATION HASH =====")
    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            static["wall_ms"],
            best_dynamic["wall_ms"]
        )
    )

    output_data = {
        "student_id": STUDENT_ID,
        "N": N,
        "physical_cores": p,
        "logical_cpus": mp.cpu_count(),
        "workers": p,
        "chunk_sizes": CHUNK_SIZES,
        "static": static,
        "cyclic": cyclic,
        "dynamic": dynamic_results,
        "best_chunk": best_chunk,
        "best_dynamic": best_dynamic,
        "checksum_verification": "passed"
    }

    with open("phase4_results.json", "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4)

    print("\nSaved results to phase4_results.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
