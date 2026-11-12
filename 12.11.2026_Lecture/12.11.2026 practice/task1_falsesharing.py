
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


M = 16_000_000
STRIDES = [1, 2, 4, 8, 16, 32]

PHYSICAL_CORES = 8
STUDENT_ID = 230103346


# Listing 2: Core Worker
def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        acc = arr[slot_idx]

        for i in range(iters):
            acc = (
                acc * 6364136223846793005 + (i | 1)
            ) & 0xFFFFFFFFFFFFFFFF

            arr[slot_idx] = acc

        del arr

    finally:
        shm.close()


# Execute one parallel trial
def run_trial(p, stride):

    shm = shared_memory.SharedMemory(create=True, size=4096)

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 1

        del arr

        processes = []

        iters = M // p

        for worker_id in range(p):

            slot_idx = worker_id * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(process)

        # Start workers
        for process in processes:
            process.start()

        # Wait for workers
        for process in processes:
            process.join()

        # Check worker exit codes
        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

    finally:
        shm.close()
        shm.unlink()


# Benchmark one stride and process count
def benchmark_stride(p, stride):

    stats = run_benchmark(
        lambda: run_trial(p, stride)
    )

    throughput = M / (stats.median_ms * 1000)

    return {
        "median": stats.median_ms,
        "iqr": stats.iqr_ms,
        "min": stats.min_ms,
        "throughput": throughput
    }


# Karp-Flatt experimental serial fraction
def karp_flatt(speedup, p):

    if p <= 1:
        return None

    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


# Print stride sweep results
def print_stride_results(results):

    print("\n========== STRIDE SWEEP ==========")
    print("Processes:", PHYSICAL_CORES)

    print(
        f"{'Stride':>8} {'Bytes':>8} "
        f"{'Median ms':>14} {'IQR ms':>12} "
        f"{'Mops/s':>12}"
    )

    for stride in STRIDES:

        r = results[stride]

        print(
            f"{stride:>8} "
            f"{stride * 8:>8} "
            f"{r['median']:>14.3f} "
            f"{r['iqr']:>12.3f} "
            f"{r['throughput']:>12.3f}"
        )


# Process scaling experiment
def run_scaling(process_counts):

    print("\n========== PROCESS SCALING ==========")

    scaling_results = {
        1: {},
        16: {}
    }

    for stride in [1, 16]:

        print(f"\n--- STRIDE = {stride} ---")

        baseline = None

        print(
            f"{'p':>5} {'Median ms':>14} "
            f"{'Speedup':>12} {'Efficiency':>12} "
            f"{'Karp-Flatt':>14}"
        )

        for p in process_counts:

            result = benchmark_stride(p, stride)

            if p == 1:
                baseline = result["median"]

            speedup = baseline / result["median"]

            efficiency = speedup / p

            kf = karp_flatt(speedup, p)

            scaling_results[stride][p] = {
                "median": result["median"],
                "speedup": speedup,
                "efficiency": efficiency,
                "karp_flatt": kf
            }

            kf_text = (
                "N/A" if kf is None else f"{kf:.6f}"
            )

            print(
                f"{p:>5} "
                f"{result['median']:>14.3f} "
                f"{speedup:>12.3f} "
                f"{efficiency:>12.3f} "
                f"{kf_text:>14}"
            )

    return scaling_results


def main():

    logical_cpus = mp.cpu_count()

    physical_cores = PHYSICAL_CORES

    print("=" * 65)
    print("PHASE 1: FALSE SHARING STRIDE SWEEP")
    print("AND KARР-FLATT DIAGNOSTICS")
    print("=" * 65)

    print("Student ID:", STUDENT_ID)
    print("Physical cores:", physical_cores)
    print("Logical CPUs:", logical_cpus)
    print("Total operations:", M)

    # 1. Stride sweep at physical core count
    stride_results = {}

    for stride in STRIDES:

        print(f"\nRunning stride={stride}, p={physical_cores}...")

        stride_results[stride] = benchmark_stride(
            physical_cores,
            stride
        )

    print_stride_results(stride_results)

    # 2. Process scaling
    process_counts = []
    p = 1

    while p <= logical_cpus:
        process_counts.append(p)
        p *= 2

    if logical_cpus not in process_counts:
        process_counts.append(logical_cpus)

    scaling_results = run_scaling(process_counts)

    # 3. Identify best stride
    print("\n========== ANALYSIS ==========")

    best_stride = min(
        STRIDES,
        key=lambda s: stride_results[s]["median"]
    )

    print("Best measured stride:", best_stride)
    print("Best byte offset:", best_stride * 8)

    print("\nStride comparison:")

    for stride in STRIDES:

        median = stride_results[stride]["median"]

        print(
            f"Stride {stride:>2} "
            f"({stride * 8:>3} bytes): "
            f"{median:.3f} ms"
        )

    print("\nKarp-Flatt comparison at p =", physical_cores)

    for stride in [1, 16]:

        kf = scaling_results[stride][physical_cores][
            "karp_flatt"
        ]

        print(f"Stride {stride}: e(p) = {kf:.6f}")

    # 4. Verification tag
    median_stride_1 = scaling_results[1][physical_cores][
        "median"
    ]

    median_stride_16 = scaling_results[16][physical_cores][
        "median"
    ]

    print("\n========== VERIFICATION ==========")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            median_stride_1,
            median_stride_16
        )
    )

    print("\nPHASE 1 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
