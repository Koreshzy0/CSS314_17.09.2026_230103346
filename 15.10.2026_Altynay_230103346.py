import numpy as np
import pytest
import scipy.ndimage
from numba import cuda

from lab_kernels.module1_execution_model import *
from lab_kernels.module2_memory_hierarchy import *

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
rng = np.random.default_rng(0)


def cdiv(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("N", [1, 255, 256, 1024, 1000003])
def test_01_vector_add(N):
    a, b = rng.random(N, dtype=np.float32), rng.random(N, dtype=np.float32)
    out = np.zeros(N, np.float32)
    vector_add_1d[cdiv(N, 256), 256](a, b, out)
    np.testing.assert_allclose(out, a + b, atol=1e-5)


def test_02_grid_stride():
    N = 10**7
    x = rng.random(N, dtype=np.float32)
    out = np.zeros(N, np.float32)
    vector_scale_grid_stride[64, 256](x, np.float32(2.5), out)
    np.testing.assert_allclose(out, x * 2.5, atol=1e-5)


def test_03_matrix_add():
    M, N = 513, 1021
    A, B = rng.random((M, N), dtype=np.float32), rng.random((M, N), dtype=np.float32)
    C = np.zeros_like(A)
    matrix_add_2d[(cdiv(N, 16), cdiv(M, 16)), (16, 16)](A, B, C)
    np.testing.assert_allclose(C, A + B, atol=1e-5)


def test_04_transpose_naive():
    M, N = 513, 1021
    A = rng.random((M, N), dtype=np.float32)
    B = np.zeros((N, M), np.float32)
    matrix_transpose_naive[(cdiv(N, 16), cdiv(M, 16)), (16, 16)](A, B)
    np.testing.assert_allclose(B, A.T, atol=1e-5)


def test_05_volume_init():
    D, H, W = 33, 47, 65
    vol = np.zeros((D, H, W), np.int32)
    volume_init_3d[(cdiv(W, 8), cdiv(H, 8), cdiv(D, 8)), (8, 8, 8)](vol, D, H, W)
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    np.testing.assert_equal(vol, (z * 10000 + y * 100 + x).astype(np.int32))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_06_grayscale(dtype):
    H, W = 130, 257
    img = (rng.random((H, W, 3)) * 255).astype(dtype)
    gray = np.zeros((H, W), np.float32)
    rgb_to_grayscale[(cdiv(W, 16), cdiv(H, 16)), (16, 16)](img, gray)
    ref = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    np.testing.assert_allclose(gray, ref, atol=1e-4, rtol=1e-5)


def test_07_lerp():
    N = 100001
    a, b, t = (rng.random(N, dtype=np.float32) for _ in range(3))
    out = np.zeros(N, np.float32)
    vector_lerp[cdiv(N, 256), 256](a, b, t, out)
    np.testing.assert_allclose(out, (1 - t) * a + t * b, atol=1e-5)


def test_08_threshold():
    N = 500001
    x = rng.random(N, dtype=np.float32)
    mask = np.zeros(N, np.uint8)
    vector_threshold[128, 256](x, np.float32(0.5), mask)
    np.testing.assert_equal(mask, (x >= 0.5).astype(np.uint8))


def test_09_strided_benchmark():
    n = 1 << 24
    d_in = cuda.to_device(rng.random(n, dtype=np.float32))
    d_out = cuda.device_array(n, np.float32)
    _, bw1 = benchmark_strided_access(d_in, d_out, 1)
    np.testing.assert_array_equal(d_out.copy_to_host(), d_in.copy_to_host())
    _, bw32 = benchmark_strided_access(d_in, d_out, 32)
    assert bw1 > bw32


@pytest.mark.parametrize("radius", [1, 3, 8])
def test_10_stencil(radius):
    N = 10007
    x = rng.random(N, dtype=np.float32)
    out = np.zeros(N, np.float32)
    stencil_1d_shared(x, out, radius)
    ref = scipy.ndimage.uniform_filter1d(x, size=2 * radius + 1, mode="nearest")
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize("K", [3, 5, 9, 16])
def test_11_conv1d(K):
    N = 5003
    x = rng.random(N, dtype=np.float32)
    k = rng.random(K, dtype=np.float32)
    y = np.zeros(N, np.float32)
    conv_1d_shared_tiled(x, k, y)
    np.testing.assert_allclose(y, np.convolve(x, k, mode="same"), atol=1e-4)


@pytest.mark.parametrize("n", [64, 500, 1024])
def test_12_matmul(n):
    A, B = rng.random((n, n), dtype=np.float32), rng.random((n, n), dtype=np.float32)
    C = np.zeros((n, n), np.float32)
    matmul_tiled_shared[(cdiv(n, 16), cdiv(n, 16)), (16, 16)](A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-3, rtol=1e-4)


def test_13_transpose_bcf():
    M, N = 513, 1021
    A = rng.random((M, N), dtype=np.float32)
    B = np.zeros((N, M), np.float32)
    transpose_bank_conflict_free[(cdiv(N, 32), cdiv(M, 32)), (32, 32)](A, B)
    np.testing.assert_allclose(B, A.T, atol=1e-5)


def test_14_reduction_sum():
    N = 1000003
    x = rng.random(N, dtype=np.float32)
    g = cdiv(N, 256)
    sums = np.zeros(g, np.float32)
    reduction_sum_shared[g, 256](x, sums)
    np.testing.assert_allclose(sums.sum(), x.sum(), rtol=1e-4)


def test_15_argmax():
    N = 100000
    x = rng.random(N, dtype=np.float32)
    x[777] = x[50000] = 5.0            # tie -> earliest index must win
    g = cdiv(N, 256)
    vals, idx = np.zeros(g, np.float32), np.zeros(g, np.int64)
    reduction_argmax_shared[g, 256](x, vals, idx)
    best = np.argmax(vals)
    assert vals[best] == x.max()
    assert idx[best] == 777 == np.argmax(x)
