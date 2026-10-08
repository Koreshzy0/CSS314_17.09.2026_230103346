import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]

        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]

        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.asarray(h_in, dtype=np.float32)
    N = h_in.size
    if N == 0:
        return np.empty(0, dtype=np.float32)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(h_in)

    blocks = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    stencil_1d[blocks, THREADS_PER_BLOCK](d_in, d_out, N)
    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_stencil(arr):
    arr = np.asarray(arr)
    padded = np.pad(arr, (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def main():
    N = 100_007
    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)

    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)

    max_delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)

    print(f"TASK 2 PASSED: MAX DELTA = {max_delta:.8g}")


if __name__ == "__main__":
    main()
