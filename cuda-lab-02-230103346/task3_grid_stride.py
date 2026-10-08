import numpy as np
from numba import cuda

N = 2**24
THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.asarray(h_arr, dtype=np.float32)
    N = h_arr.size

    d_arr = cuda.to_device(h_arr)
    grid_stride_scale_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](d_arr, factor, N)
    cuda.synchronize()

    return d_arr.copy_to_host()


def main():
    h_arr = np.ones(N, dtype=np.float32)
    factor = np.float32(4.25)

    result = run_grid_stride(h_arr, factor)
    assert np.allclose(result, factor)

    print(f"TASK 3 PASSED: all {N:,} elements equal {factor}")


if __name__ == "__main__":
    main()
