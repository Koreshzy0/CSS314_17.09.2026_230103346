
import csv
import ctypes
import multiprocessing as mp
import time
from pathlib import Path


K = 5_000_000
WORKERS = [1, 2, 4, 8]


def worker(shared, worker_id, iterations, stride, mode):
    if mode == "privatized":
        local = 0
        for _ in range(iterations):
            local += 1
        shared[worker_id * stride] = local
    else:
        index = worker_id * stride
        for _ in range(iterations):
            shared[index] += 1


def run_test(P, mode):
    stride = 8 if mode == "padded" else 1
    shared = mp.RawArray(ctypes.c_int64, P * stride)

    processes = [
        mp.Process(
            target=worker,
            args=(shared, p, K, stride, mode),
        )
        for p in range(P)
    ]

    start = time.perf_counter()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    elapsed = time.perf_counter() - start

    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("A worker process failed")

    total = sum(shared[p * stride] for p in range(P))
    assert total == P * K, (mode, P, total)

    return elapsed, total


def karp_flatt(T1, TP, P):
    if P == 1 or TP <= 0:
        return 0.0
    speedup = T1 / TP
    if speedup <= 0:
        return float("nan")
    return (1 / speedup - 1 / P) / (1 - 1 / P)


def main():
    results = []

    for mode in ["unpadded", "padded", "privatized"]:
        for P in WORKERS:
            elapsed, total = run_test(P, mode)
            results.append({
                "mode": mode,
                "processes": P,
                "iterations_per_process": K,
                "runtime_seconds": elapsed,
                "total_count": total,
            })
            print(mode, P, elapsed, total)

    for mode in ["unpadded", "padded", "privatized"]:
        subset = [r for r in results if r["mode"] == mode]
        T1 = next(r["runtime_seconds"] for r in subset
                  if r["processes"] == 1)
        for row in subset:
            P = row["processes"]
            TP = row["runtime_seconds"]
            row["speedup"] = T1 / TP
            row["karp_flatt"] = karp_flatt(T1, TP, P)

    Path("results").mkdir(exist_ok=True)
    with open("results/false_sharing_karp_flatt.tsv", "w",
              newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys(), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    mp.freeze_support()
    main()
