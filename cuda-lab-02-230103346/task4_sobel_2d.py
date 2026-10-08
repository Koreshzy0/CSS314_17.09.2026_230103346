import numpy as np
from numba import cuda

THREADS_2D = (16, 16)


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    if row < rows and col < cols:
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = 0.0
        else:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1]
                + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1]
                + 1.0 * d_in[row + 1, col + 1]
            )


def run_sobel(h_img):
    h_img = np.asarray(h_img, dtype=np.float32)
    if h_img.ndim != 2:
        raise ValueError("h_img must be a 2D matrix")

    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(h_img)

    blocks_2d = (
        (cols + THREADS_2D[0] - 1) // THREADS_2D[0],
        (rows + THREADS_2D[1] - 1) // THREADS_2D[1],
    )

    sobel_x_kernel[blocks_2d, THREADS_2D](d_in, d_out, rows, cols)
    cuda.synchronize()

    return d_out.copy_to_host()


def main():
    rows = cols = 2048
    h_img = np.random.default_rng(42).random((rows, cols), dtype=np.float32)

    result = run_sobel(h_img)

    # Border must be zero.
    assert np.all(result[0, :] == 0.0)
    assert np.all(result[-1, :] == 0.0)
    assert np.all(result[:, 0] == 0.0)
    assert np.all(result[:, -1] == 0.0)

    print(f"TASK 4 PASSED: Sobel-X completed for {rows}x{cols} image")


if __name__ == "__main__":
    main()
