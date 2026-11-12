
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103346

PHYSICAL_CORES = 8

BUFFER_SIZE = 64 * 1024 * 1024
N = 256

TILE_SIZES = [8, 16, 32, 64, 96, 128]

WARMUP = 1
RUNS = 3


# ============================================================
# PART 3A: DRAM BANDWIDTH
# ============================================================

def memory_copy_worker(args):

    name_a, name_b, start, end = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    try:

        mv_a = memoryview(shm_a.buf)
        mv_b = memoryview(shm_b.buf)

        mv_a[start:end] = mv_b[start:end]

        mv_a.release()
        mv_b.release()

    finally:

        shm_a.close()
        shm_b.close()


def measure_bandwidth():

    print("\n========== DRAM BANDWIDTH ==========")

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    try:

        chunk = BUFFER_SIZE // PHYSICAL_CORES

        tasks = []

        for p in range(PHYSICAL_CORES):

            start = p * chunk

            end = (
                BUFFER_SIZE
                if p == PHYSICAL_CORES - 1
                else start + chunk
            )

            tasks.append(
                (shm_a.name, shm_b.name, start, end)
            )

        with mp.Pool(processes=PHYSICAL_CORES) as pool:

            stats = run_benchmark(
                lambda: pool.map(memory_copy_worker, tasks),
                warmup=WARMUP,
                runs=RUNS
            )

        # Two buffers: one read and one write
        bandwidth = (
            2 * BUFFER_SIZE
            / (stats.median_ms * 1_000_000)
        )

        print(f"Median: {stats.median_ms:.3f} ms")
        print(f"IQR: {stats.iqr_ms:.3f} ms")
        print(f"Sustained bandwidth: {bandwidth:.3f} GB/s")

        return bandwidth

    finally:

        shm_a.close()
        shm_b.close()

        shm_a.unlink()
        shm_b.unlink()


# ============================================================
# COMPUTE CEILING
# ============================================================

def compute_worker(iterations):

    # 16 independent scalar accumulators.
    # Each update performs one multiplication and one addition.

    a0 = 0.1
    a1 = 0.2
    a2 = 0.3
    a3 = 0.4
    a4 = 0.5
    a5 = 0.6
    a6 = 0.7
    a7 = 0.8
    a8 = 0.9
    a9 = 1.0
    a10 = 1.1
    a11 = 1.2
    a12 = 1.3
    a13 = 1.4
    a14 = 1.5
    a15 = 1.6

    b = 1.000000001
    c = 0.000000001

    for _ in range(iterations):

        a0 = a0 * b + c
        a1 = a1 * b + c
        a2 = a2 * b + c
        a3 = a3 * b + c
        a4 = a4 * b + c
        a5 = a5 * b + c
        a6 = a6 * b + c
        a7 = a7 * b + c

        a8 = a8 * b + c
        a9 = a9 * b + c
        a10 = a10 * b + c
        a11 = a11 * b + c
        a12 = a12 * b + c
        a13 = a13 * b + c
        a14 = a14 * b + c
        a15 = a15 * b + c

    return (
        a0 + a1 + a2 + a3 +
        a4 + a5 + a6 + a7 +
        a8 + a9 + a10 + a11 +
        a12 + a13 + a14 + a15
    )


def measure_compute():

    print("\n========== COMPUTE CEILING ==========")

    total_flops_per_worker = 60_000_000

    iterations = total_flops_per_worker // 32

    with mp.Pool(processes=PHYSICAL_CORES) as pool:

        stats = run_benchmark(
            lambda: pool.map(
                compute_worker,
                [iterations] * PHYSICAL_CORES
            ),
            warmup=WARMUP,
            runs=RUNS
        )

    total_flops = (
        total_flops_per_worker * PHYSICAL_CORES
    )

    gflops = (
        total_flops / (stats.median_ms * 1_000_000)
    )

    print(f"Median: {stats.median_ms:.3f} ms")
    print(f"IQR: {stats.iqr_ms:.3f} ms")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Compute ceiling: {gflops:.6f} GFLOP/s")

    return gflops


# ============================================================
# PART 3B: GEMM HELPERS
# ============================================================

def initialize_matrices(shm_a, shm_b, shm_c):

    count = N * N

    A = (ctypes.c_double * count).from_buffer(shm_a.buf)
    B = (ctypes.c_double * count).from_buffer(shm_b.buf)
    C = (ctypes.c_double * count).from_buffer(shm_c.buf)

    for i in range(N):

        for j in range(N):

            A[i * N + j] = (
                ((i * 7 + j * 3) % 100) / 100.0
            )

            B[i * N + j] = (
                ((i * 5 + j * 11) % 100) / 100.0
            )

            C[i * N + j] = 0.0

    del A
    del B
    del C


def make_row_ranges():

    ranges = []

    rows_per_worker = N // PHYSICAL_CORES

    for p in range(PHYSICAL_CORES):

        start = p * rows_per_worker

        end = (
            N
            if p == PHYSICAL_CORES - 1
            else start + rows_per_worker
        )

        ranges.append((start, end))

    return ranges


def clear_matrix(shm_c):

    C = (ctypes.c_double * (N * N)).from_buffer(shm_c.buf)

    for i in range(N * N):
        C[i] = 0.0

    del C


# ============================================================
# GEMM VARIANT WORKER
# ============================================================

def gemm_worker(args):

    name_a, name_b, name_c, start, end, variant, bs = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    try:

        count = N * N

        A = (ctypes.c_double * count).from_buffer(shm_a.buf)
        B = (ctypes.c_double * count).from_buffer(shm_b.buf)
        C = (ctypes.c_double * count).from_buffer(shm_c.buf)

        # ----------------------------------------
        # Variant 1: Naive i-j-k
        # ----------------------------------------

        if variant == 1:

            for i in range(start, end):

                for j in range(N):

                    total = 0.0

                    for k in range(N):

                        total += (
                            A[i * N + k] *
                            B[k * N + j]
                        )

                    C[i * N + j] = total

        # ----------------------------------------
        # Variant 2: Reordered i-k-j
        # ----------------------------------------

        elif variant == 2:

            for i in range(start, end):

                for k in range(N):

                    a_ik = A[i * N + k]

                    for j in range(N):

                        C[i * N + j] += (
                            a_ik * B[k * N + j]
                        )

        # ----------------------------------------
        # Variant 3: Cache-blocked i-k-j
        # ----------------------------------------

        elif variant == 3:

            for ii in range(start, end, bs):

                i_end = min(ii + bs, end)

                for jj in range(0, N, bs):

                    j_end = min(jj + bs, N)

                    for kk in range(0, N, bs):

                        k_end = min(kk + bs, N)

                        for i in range(ii, i_end):

                            for k in range(kk, k_end):

                                a_ik = A[i * N + k]

                                for j in range(jj, j_end):

                                    C[i * N + j] += (
                                        a_ik * B[k * N + j]
                                    )

        del A
        del B
        del C

    finally:

        shm_a.close()
        shm_b.close()
        shm_c.close()


# ============================================================
# GEMM BENCHMARK
# ============================================================

def gemm_tasks(shm_a, shm_b, shm_c, variant, bs):

    tasks = []

    for start, end in make_row_ranges():

        tasks.append(
            (
                shm_a.name,
                shm_b.name,
                shm_c.name,
                start,
                end,
                variant,
                bs
            )
        )

    return tasks


def run_gemm_benchmark(pool, shm_a, shm_b, shm_c, variant, bs=32):

    def execute():

        clear_matrix(shm_c)

        tasks = gemm_tasks(
            shm_a,
            shm_b,
            shm_c,
            variant,
            bs
        )

        pool.map(gemm_worker, tasks)

    stats = run_benchmark(
        execute,
        warmup=WARMUP,
        runs=RUNS
    )

    return stats


def read_matrix(shm_c):

    C = (
        ctypes.c_double * (N * N)
    ).from_buffer(shm_c.buf)

    values = list(C)

    del C

    return values


def matrix_gflops(median_ms):

    operations = 2 * (N ** 3)

    return operations / (median_ms * 1_000_000)


# ============================================================
# COMPLETE GEMM EXPERIMENT
# ============================================================

def measure_gemm():

    print("\n========== GEMM EXPERIMENT ==========")

    size = N * N * ctypes.sizeof(ctypes.c_double)

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    shm_c = shared_memory.SharedMemory(
        create=True,
        size=size
    )

    try:

        initialize_matrices(shm_a, shm_b, shm_c)

        with mp.Pool(processes=PHYSICAL_CORES) as pool:

            # Variant 1
            print("\nRunning Variant 1: Naive i-j-k")

            stats1 = run_gemm_benchmark(
                pool, shm_a, shm_b, shm_c, 1
            )

            gflops1 = matrix_gflops(stats1.median_ms)

            print(f"Median: {stats1.median_ms:.3f} ms")
            print(f"IQR: {stats1.iqr_ms:.3f} ms")
            print(f"GFLOP/s: {gflops1:.6f}")

            # Variant 2
            print("\nRunning Variant 2: Reordered i-k-j")

            stats2 = run_gemm_benchmark(
                pool, shm_a, shm_b, shm_c, 2
            )

            gflops2 = matrix_gflops(stats2.median_ms)

            print(f"Median: {stats2.median_ms:.3f} ms")
            print(f"IQR: {stats2.iqr_ms:.3f} ms")
            print(f"GFLOP/s: {gflops2:.6f}")

            # Save Variant 2 result for verification
            reference = read_matrix(shm_c)

            # Variant 3: Tile sweep
            print("\n========== CACHE TILE SWEEP ==========")

            tile_results = {}

            print(
                f"{'BS':>8} {'Footprint KB':>16} "
                f"{'Intensity':>14} {'Median ms':>14} "
                f"{'GFLOP/s':>14}"
            )

            for bs in TILE_SIZES:

                print(f"Running tile size BS={bs}...")

                stats3 = run_gemm_benchmark(
                    pool,
                    shm_a,
                    shm_b,
                    shm_c,
                    3,
                    bs
                )

                gflops3 = matrix_gflops(stats3.median_ms)

                footprint = 3 * bs * bs * 8

                footprint_kb = footprint / 1024

                intensity = bs / 12.0

                tile_results[bs] = {
                    "median": stats3.median_ms,
                    "iqr": stats3.iqr_ms,
                    "gflops": gflops3,
                    "footprint": footprint,
                    "intensity": intensity
                }

                print(
                    f"{bs:>8} "
                    f"{footprint_kb:>16.2f} "
                    f"{intensity:>14.3f} "
                    f"{stats3.median_ms:>14.3f} "
                    f"{gflops3:>14.6f}"
                )

            # Select fastest measured tile
            best_bs = min(
                TILE_SIZES,
                key=lambda bs: tile_results[bs]["median"]
            )

            best_result = tile_results[best_bs]

            # Read Variant 3 output
            blocked_result = read_matrix(shm_c)

            # Numerical verification
            max_error = max(
                abs(a - b)
                for a, b in zip(reference, blocked_result)
            )

            print("\n========== NUMERICAL VERIFICATION ==========")

            print(f"Best tile size: {best_bs}")
            print(f"Best tile footprint: {best_result['footprint']} bytes")
            print(f"Best operational intensity: {best_result['intensity']:.3f}")
            print(f"Maximum absolute error: {max_error:.12e}")

            if max_error < 1e-9:
                print("Numerical verification PASSED")
            else:
                print("Numerical verification FAILED")

            return (
                gflops1,
                gflops2,
                best_result["gflops"],
                best_bs,
                max_error
            )

    finally:

        shm_a.close()
        shm_b.close()
        shm_c.close()

        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3: EMPIRICAL ROOFLINE AND CACHE-BLOCKED GEMM")
    print("=" * 65)

    print("Student ID:", STUDENT_ID)
    print("Physical cores:", PHYSICAL_CORES)
    print("Logical CPUs:", mp.cpu_count())
    print("GEMM matrix size:", N, "x", N)

    # Part A: Bandwidth
    bandwidth = measure_bandwidth()

    # Part A: Compute ceiling
    compute = measure_compute()

    # Roofline ridge point
    ridge_point = compute / bandwidth

    print("\n========== ROOFLINE MODEL ==========")

    print(f"Bandwidth ceiling: {bandwidth:.3f} GB/s")
    print(f"Compute ceiling: {compute:.6f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.6f} FLOP/byte")

    # Part B: GEMM
    gflops1, gflops2, gflops3, best_bs, max_error = (
        measure_gemm()
    )

    print("\n========== ROOFLINE COMPARISON ==========")

    print(f"Variant 1: {gflops1:.6f} GFLOP/s")
    print(f"Variant 2: {gflops2:.6f} GFLOP/s")
    print(f"Variant 3: {gflops3:.6f} GFLOP/s")
    print(f"Optimal measured BS: {best_bs}")

    print("\n========== VERIFICATION ==========")

    if max_error >= 1e-9:

        raise RuntimeError(
            "GEMM numerical verification failed."
        )

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            gflops1,
            gflops3
        )
    )

    print("\nPHASE 3 COMPLETED")


if __name__ == "__main__":

    mp.freeze_support()
    main()
