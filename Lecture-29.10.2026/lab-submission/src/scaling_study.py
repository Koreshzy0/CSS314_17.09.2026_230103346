
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def stencil_chunk(args):
    start, end, N, seed = args
    rng = np.random.default_rng(seed)
    # Each worker generates its own deterministic chunk.
    rows = end - start
    A = rng.random((rows + 2, N + 2))
    C = (
        A[:-2, 1:-1] + A[2:, 1:-1]
        + A[1:-1, :-2] + A[1:-1, 2:]
        - 4.0 * A[1:-1, 1:-1]
    )
    return float(np.sum(C)), rows * N


def run_stencil(N, P, seed):
    boundaries = np.linspace(0, N, P + 1, dtype=int)
    chunks = [
        (int(boundaries[i]), int(boundaries[i + 1]), N, seed + i)
        for i in range(P)
        if boundaries[i] < boundaries[i + 1]
    ]

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=P) as pool:
        results = list(pool.map(stencil_chunk, chunks))
    elapsed = time.perf_counter() - start

    checksum = sum(item[0] for item in results)
    cells = sum(item[1] for item in results)
    return elapsed, checksum, cells


def karp_flatt(T1, TP, P):
    if P == 1:
        return 0.0
    S = T1 / TP
    return (1 / S - 1 / P) / (1 - 1 / P)


def main():
    params = json.loads(
        Path("results/salted_params.json").read_text()
    )
    profile = json.loads(
        Path("results/host_profile.json").read_text()
    )

    N = params["N"]
    seed = params["seed"]
    physical = profile.get("physical_cores") or os.cpu_count() or 1
    max_p = min(physical, 8)
    workers = sorted(set([1] + [
        p for p in [2, 4, 8] if p <= max_p
    ]))

    # Warm up each worker count before recording timings.
    for p in workers:
        run_stencil(min(N, 64), p, seed)

    strong = []
    reference_checksum = None

    for P in workers:
        timings = []
        checksums = []
        for _ in range(3):
            elapsed, checksum, cells = run_stencil(N, P, seed)
            timings.append(elapsed)
            checksums.append(checksum)

        median = statistics.median(timings)
        checksum = checksums[-1]
        if P == 1:
            reference_checksum = checksum

        strong.append({
            "study": "strong",
            "processes": P,
            "N": N,
            "runtime_seconds": median,
            "checksum": checksum,
            "cells": cells,
        })

    T1 = next(r["runtime_seconds"] for r in strong
              if r["processes"] == 1)

    for row in strong:
        P = row["processes"]
        S = T1 / row["runtime_seconds"]
        row["speedup"] = S
        row["efficiency_percent"] = S / P * 100
        row["karp_flatt"] = karp_flatt(
            T1, row["runtime_seconds"], P
        )

    serial_values = [
        r["karp_flatt"] for r in strong if r["processes"] > 1
    ]
    serial_fraction = max(
        0.0, min(1.0, statistics.mean(serial_values))
    ) if serial_values else 1.0

    # Weak scaling: N grows approximately with sqrt(P).
    weak = []
    weak_T1 = next(r["runtime_seconds"] for r in strong
                   if r["processes"] == 1)

    for P in workers:
        NP = math.ceil(N * math.sqrt(P))
        timings = []
        checksums = []

        for _ in range(3):
            elapsed, checksum, cells = run_stencil(NP, P, seed)
            timings.append(elapsed)
            checksums.append(checksum)

        median = statistics.median(timings)
        weak.append({
            "study": "weak",
            "processes": P,
            "N": NP,
            "runtime_seconds": median,
            "checksum": checksums[-1],
            "cells": cells,
            "scaled_speedup": weak_T1 / median,
        })

    all_rows = strong + weak
    with open("results/scaling_metrics.tsv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({
                key for row in all_rows for key in row
            }), delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    cache = profile.get("cache", {})
    # Five-point stencil: approximately 5 FLOPs and 6 doubles
    # transferred per output cell under a conservative DRAM model.
    flops_per_cell = 5
    bytes_per_cell = 6 * 8
    arithmetic_intensity = flops_per_cell / bytes_per_cell

    roofline = {
        "kernel": "five_point_stencil",
        "flops_per_cell_estimate": flops_per_cell,
        "dram_bytes_per_cell_estimate": bytes_per_cell,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "cache_sizes_bytes": cache,
        "peak_memory_bandwidth_bytes_per_second": None,
        "peak_compute_flops_per_second": None,
        "bottleneck_classification": (
            "Undetermined: provide measured or documented "
            "peak bandwidth and peak FLOP/s to classify Roofline."
        ),
        "serial_fraction_estimate": serial_fraction,
        "amdahl_speedup_ceiling": (
            1 / serial_fraction if serial_fraction > 0 else None
        ),
        "strong_scaling_reference_checksum": reference_checksum,
        "note": (
            "Checksum comparisons across process counts require "
            "identical input data and decomposition-independent "
            "numerical behavior."
        ),
    }

    Path("results/roofline_summary.json").write_text(
        json.dumps(roofline, indent=2)
    )
    print(json.dumps(roofline, indent=2))


if __name__ == "__main__":
    main()
