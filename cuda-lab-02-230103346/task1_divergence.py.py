import time
import numpy as np
from numba import cuda

N = 2**20
ITERATIONS = 1000
THREADS_PER_BLOCK = 256


@cuda.jit
def kernel_a(d_y, n):
    idx = cuda.grid(1)
    if idx < n:
        y = d_y[idx]
        for _ in range(ITERATIONS):
            y = y * 1.0001 + 0.0001
        d_y[idx] = y


@cuda.jit
def kernel_b(d_y, n):
    idx = cuda.grid(1)
    if idx < n:
        y = d_y[idx]
        if idx % 2 == 0:
            for _ in range(ITERATIONS):
                y = y * 1.0001 + 0.0001
        else:
            for _ in range(ITERATIONS):
                # Distinct subtract-divide path for odd threads.
                y = y - 0.0001 / (y + 1.0)
        d_y[idx] = y


@cuda.jit
def kernel_c(d_y, n):
    idx = cuda.grid(1)
    if idx < n:
        y = d_y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(ITERATIONS):
                y = y * 1.0001 + 0.0001
        else:
            for _ in range(ITERATIONS):
                y = y - 0.0001 / (y + 1.0)
        d_y[idx] = y


def _benchmark_kernel(kernel, d_y, n, blocks_per_grid):
    # Warm-up
    kernel[blocks_per_grid, THREADS_PER_BLOCK](d_y, n)
    cuda.synchronize()

    times_ms = []
    for _ in range(10):
        start = time.perf_counter()
        kernel[blocks_per_grid, THREADS_PER_BLOCK](d_y, n)
        cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(times_ms)), times_ms


def main():
    h_y = np.ones(N, dtype=np.float32)
    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    d_y = cuda.to_device(h_y)

    results = {}
    for name, kernel in [
        ("A - Uniform", kernel_a),
        ("B - Full Divergence", kernel_b),
        ("C - Warp-Aligned", kernel_c),
    ]:
        d_y.copy_to_device(h_y)
        avg_ms, trials = _benchmark_kernel(kernel, d_y, N, blocks)
        results[name] = (avg_ms, trials)

    print(f"N = {N:,}")
    print("Kernel-only timing, average of 10 trials (ms):")
    for name, (avg_ms, trials) in results.items():
        print(f"{name:24s}: {avg_ms:10.4f} ms")
        print("  trials:", ", ".join(f"{x:.4f}" for x in trials))


if __name__ == "__main__":
    main()
