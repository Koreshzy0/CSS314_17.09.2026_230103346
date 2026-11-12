import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import json
import csv
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103346
M = 24_000_000
STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8]


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc

    del arr
    shm.close()


def execute(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    processes = []

    try:
        for t in range(p):
            proc = mp.Process(
                target=fs_worker,
                args=(shm.name, t * stride, M // p)
            )
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()

        for proc in processes:
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {proc.exitcode}"
                )
    finally:
        shm.close()
        shm.unlink()


def benchmark(p, stride):
    return run_benchmark(
        lambda: execute(p, stride),
        warmup_runs=2,
        timed_runs=7
    )


def karp_flatt(speedup, p):
    if p == 1:
        return 0.0
    return (1 / speedup - 1 / p) / (1 - 1 / p)


def main():
    print("===== PHASE 1: FALSE SHARING =====")
    print("Student ID:", STUDENT_ID)
    print("Total operations:", M)

    results_1a = {}
    results_1b = {}

    # Table 1A
    print("\n===== TABLE 1A: STRIDE SWEEP =====")

    for stride in STRIDES:
        stats = benchmark(8, stride)

        throughput = M / (stats.median_ms * 1000)

        results_1a[str(stride)] = {
            "median_ms": stats.median_ms,
            "iqr_ms": stats.iqr_ms,
            "throughput_mops": throughput
        }

        print(
            f"Stride={stride:2d} | "
            f"Bytes={stride * 8:3d} | "
            f"Median={stats.median_ms:.4f} ms | "
            f"IQR={stats.iqr_ms:.4f} ms | "
            f"Throughput={throughput:.4f} Mops/s"
        )

    # Table 1B
    print("\n===== TABLE 1B: PROCESS SCALING =====")

    for stride in [1, 16]:
        results_1b[str(stride)] = {}

        print(f"\n--- STRIDE {stride} ---")

        t1 = benchmark(1, stride)
        baseline = t1.median_ms

        for p in PROCESS_COUNTS:
            stats = t1 if p == 1 else benchmark(p, stride)

            speedup = baseline / stats.median_ms
            efficiency = speedup / p
            e = karp_flatt(speedup, p)

            results_1b[str(stride)][str(p)] = {
                "median_ms": stats.median_ms,
                "speedup": speedup,
                "efficiency": efficiency,
                "karp_flatt": e
            }

            print(
                f"p={p} | "
                f"Median={stats.median_ms:.4f} ms | "
                f"Speedup={speedup:.4f} | "
                f"Efficiency={efficiency:.4f} | "
                f"e(p)={e:.6f}"
            )

    # Save results
    with open("phase1_results.json", "w") as f:
        json.dump({
            "table_1a": results_1a,
            "table_1b": results_1b
        }, f, indent=4)

    # Save Table 1A CSV
    with open("table1a.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Stride", "Bytes", "Median_ms",
            "IQR_ms", "Throughput_Mops"
        ])

        for stride, data in results_1a.items():
            writer.writerow([
                stride,
                int(stride) * 8,
                data["median_ms"],
                data["iqr_ms"],
                data["throughput_mops"]
            ])

    # Save Table 1B CSV
    with open("table1b.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Stride", "Processes", "Median_ms",
            "Speedup", "Efficiency", "Karp_Flatt"
        ])

        for stride, rows in results_1b.items():
            for p, data in rows.items():
                writer.writerow([
                    stride, p,
                    data["median_ms"],
                    data["speedup"],
                    data["efficiency"],
                    data["karp_flatt"]
                ])

    m1 = results_1a["1"]["median_ms"]
    m16 = results_1a["16"]["median_ms"]

    print("\n===== PHASE 1 VERIFICATION =====")
    print(emit_verify_tag("PHASE1", STUDENT_ID, m1, m16))

    print("\nSaved:")
    print("phase1_results.json")
    print("table1a.csv")
    print("table1b.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
