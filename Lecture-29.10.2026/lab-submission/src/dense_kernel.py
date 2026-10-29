
import json
import statistics
import time
from pathlib import Path

import numpy as np
from numba import njit


@njit
def naive(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit
def interchanged(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            aik = A[i, k]
            for j in range(n):
                C[i, j] += aik * B[k, j]
    return C


def benchmark(fn, A, B):
    times = []
    result = None
    for _ in range(3):
        start = time.perf_counter()
        result = fn(A, B)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def main():
    params = json.loads(
        Path("results/salted_params.json").read_text()
    )
    M = params["N"] // 2
    rng = np.random.default_rng(params["seed"])

    A = rng.random((M, M))
    B = rng.random((M, M))

    # Compile before collecting benchmark timings.
    naive(A[:2, :2], B[:2, :2])
    interchanged(A[:2, :2], B[:2, :2])

    t_naive, C1 = benchmark(naive, A, B)
    t_interchanged, C2 = benchmark(interchanged, A, B)

    result = {
        "N": params["N"],
        "M": M,
        "naive_median_seconds": t_naive,
        "interchanged_median_seconds": t_interchanged,
        "speedup": t_naive / t_interchanged,
        "max_absolute_error": float(np.max(np.abs(C1 - C2))),
        "numerically_correct": bool(
            np.allclose(C1, C2, rtol=1e-10, atol=1e-12)
        ),
    }

    Path("results/loop_interchange.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
