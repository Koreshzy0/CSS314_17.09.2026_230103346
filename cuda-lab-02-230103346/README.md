# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** `<230103346>`  
**Allocated GPU Node:** `<Tesla T4>`  
**CUDA Compute Capability:** `<7.5>`  
**Official Verification Token:** `<80EE7EA7EE8F1AD9BDD5>`

## Repository structure

cuda-lab-02-<230103346>/
├── README.md
├── task1_divergence.py
├── task2_stencil_1d.py
├── task3_grid_stride.py
├── task4_sobel_2d.py
└── verify_submission.py
```

## Task 1 — Warp Divergence Microbenchmark

- Array size: `N = 2^20 = 1,048,576`
- Data type: `np.float32`
- Iterations per element: `1,000`
- Kernel A: uniform arithmetic path.
- Kernel B: interleaved `idx % 2` branching, causing intra-warp divergence.
- Kernel C: `warp_id = idx // 32`, so every warp follows one path.
- Benchmark excludes host/device transfers.
- A warm-up launch is performed before 10 measured trials.

Run:

```bash
python task1_divergence.py
```

### Benchmark results

Fill these values with the actual output from the CUDA GPU:

| Kernel | Average time (ms) |
|---|---:|
| A — Uniform | `<RUN ON GPU>` |
| B — Full Divergence | `<RUN ON GPU>` |
| C — Warp-Aligned | `<RUN ON GPU>` |

### Interpretation

Kernel B introduces divergence inside each warp because adjacent threads take different branches. The GPU therefore serializes the different paths within a warp. Kernel C avoids intra-warp divergence because all 32 threads in a warp follow the same branch.

## Task 2 — 1D Boundary Stencil

The 3-point smoothing filter is:

`y[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1]`

At the boundaries, halo replication is used:

- `i == 0`: left neighbor is `x[0]`
- `i == N-1`: right neighbor is `x[N-1]`
- otherwise: normal neighboring elements are used.

Test size:

`N = 100,007`

The GPU result is checked against the NumPy reference with `atol=1e-4`.

Run:

```bash
python task2_stencil_1d.py
```

Expected output format:

```text
TASK 2 PASSED: MAX DELTA = <small floating-point value>
```

## Task 3 — Grid-Stride Scaling

- Vector size: `N = 2^24 = 16,777,216`
- Threads per block: `256`
- Blocks per grid: `64`
- Total launched hardware threads: `16,384`
- The grid-stride loop processes all `16,777,216` elements.

The kernel uses:

```python
start = cuda.grid(1)
stride = cuda.gridsize(1)

for i in range(start, N, stride):
    d_arr[i] = d_arr[i] * factor
```

Run:

```bash
python task3_grid_stride.py
```

## Task 4 — Sobel Horizontal Filter

Input matrix:

`2048 x 2048`, `float32`

Sobel-X kernel:

```text
[-1  0 +1]
[-2  0 +2]
[-1  0 +1]
```

The outermost border pixels are set to `0.0`.

CUDA launch configuration:

```text
threads_2d = (16, 16)
```

Grid dimensions are calculated dynamically from the image dimensions.

Run:

```bash
python task4_sobel_2d.py
```

## Automated verification

Run:

```bash
python verify_submission.py
```

The verification script checks Tasks 2–4 and generates the official integrity token from:

1. Student ID
2. CUDA device name
3. Task 2 output bytes
4. Task 4 output bytes

After running it successfully, replace `<PASTE_TOKEN_HERE>` above with the printed token.

## Answers / key concepts

### Why does Kernel B diverge?

Threads in the same warp use alternating branches (`idx % 2`), so neighboring threads do not follow the same execution path. The GPU must execute the different paths serially for the warp.

### Why is Kernel C better aligned with the hardware?

A warp contains 32 threads. Using `warp_id = idx // 32` makes all threads in one warp choose the same branch, avoiding intra-warp branch divergence.

### Why is boundary clamping needed in Task 2?

The first element has no real left neighbor and the last element has no real right neighbor. Replicating the boundary value prevents out-of-bounds memory accesses.

### Why use a grid-stride loop in Task 3?

The data vector is much larger than the number of launched threads. Each thread therefore processes multiple elements separated by the total grid stride.

### Why are border pixels zero in Task 4?

A 3x3 Sobel operation needs neighboring pixels on all sides. The outer border does not have a complete 3x3 neighborhood, so the assignment explicitly sets those pixels to zero.

## Integrity note

The benchmark numbers and official verification token are machine-specific and must be generated on the CUDA environment used for submission. They should not be fabricated.
