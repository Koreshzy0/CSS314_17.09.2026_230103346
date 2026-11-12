import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import json
from harness import emit_verify_tag

N = 384
P = 8
TILES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

_SHM = []

def init_worker(a_name, b_name, c_name, n):
    global A, B, C, SHM
    SHM = [
        shared_memory.SharedMemory(name=a_name),
        shared_memory.SharedMemory(name=b_name),
        shared_memory.SharedMemory(name=c_name)
    ]
    A = SHM[0].buf.cast("d")
    B = SHM[1].buf.cast("d")
    C = SHM[2].buf.cast("d")

def gemm_worker(args):
    mode, row_start, row_end, bs = args
    n = N

    if mode == 1:
        # Variant 1: naive i-j-k
        for i in range(row_start, row_end):
            ib = i * n
            for j in range(n):
                total = 0.0
                for k in range(n):
                    total += A[ib + k] * B[k * n + j]
                C[ib + j] = total

    elif mode == 2:
        # Variant 2: reordered i-k-j
        for i in range(row_start, row_end):
            ib = i * n
            for k in range(n):
                aik = A[ib + k]
                kb = k * n
                for j in range(n):
                    C[ib + j] += aik * B[kb + j]

    else:
        # Variant 3: cache-blocked i-k-j
        for ii in range(row_start, row_end, bs):
            i_end = min(ii + bs, row_end)
            for kk in range(0, n, bs):
                k_end = min(kk + bs, n)
                for jj in range(0, n, bs):
                    j_end = min(jj + bs, n)
                    for i in range(ii, i_end):
                        ib = i * n
                        for k in range(kk, k_end):
                            aik = A[ib + k]
                            kb = k * n
                            for j in range(jj, j_end):
                                C[ib + j] += aik * B[kb + j]

    return row_end - row_start

def zero_c(c_view):
    for i in range(N * N):
        c_view[i] = 0.0

def run_variant(pool, c_view, mode, bs=0):
    zero_c(c_view)
    rows_per_worker = (N + P - 1) // P
    tasks = []
    for w in range(P):
        start = w * rows_per_worker
        end = min(start + rows_per_worker, N)
        if start < end:
            tasks.append((mode, start, end, bs))

    t0 = time.perf_counter()
    pool.map(gemm_worker, tasks)
    return time.perf_counter() - t0

def snapshot(c_view):
    return [c_view[i] for i in range(N * N)]

def max_error(x, y):
    return max(abs(a - b) for a, b in zip(x, y))

def benchmark(pool, c_view, mode, bs=0, repeats=3):
    run_variant(pool, c_view, mode, bs)
    times = [
        run_variant(pool, c_view, mode, bs)
        for _ in range(repeats)
    ]
    return statistics.median(times), times

def main():
    print("===== PHASE 3B: GEMM OPTIMIZATION =====")
    print(f"Matrix: {N} x {N}; workers: {P}")

    size = N * N * 8
    shm_a = shared_memory.SharedMemory(create=True, size=size)
    shm_b = shared_memory.SharedMemory(create=True, size=size)
    shm_c = shared_memory.SharedMemory(create=True, size=size)

    a = shm_a.buf.cast("d")
    b = shm_b.buf.cast("d")
    c = shm_c.buf.cast("d")

    try:
        # Deterministic input matrices
        for i in range(N):
            for j in range(N):
                a[i * N + j] = ((i * 7 + j * 3) % 101) / 101.0
                b[i * N + j] = ((i * 5 + j * 11) % 97) / 97.0

        with mp.Pool(
            processes=P,
            initializer=init_worker,
            initargs=(shm_a.name, shm_b.name, shm_c.name, N)
        ) as pool:

            results = {}

            print("\nVariant 1: Naive i-j-k")
            t1, samples1 = benchmark(pool, c, 1)
            result1 = snapshot(c)
            g1 = (2 * N**3) / t1 / 1e9
            results["variant1"] = {
                "median_s": t1,
                "samples_s": samples1,
                "GFLOP_s": g1
            }
            print(f"Median: {t1:.4f} s; GFLOP/s: {g1:.6f}")

            print("\nVariant 2: Reordered i-k-j")
            t2, samples2 = benchmark(pool, c, 2)
            result2 = snapshot(c)
            g2 = (2 * N**3) / t2 / 1e9
            err12 = max_error(result1, result2)
            results["variant2"] = {
                "median_s": t2,
                "samples_s": samples2,
                "GFLOP_s": g2,
                "max_error_vs_variant1": err12
            }
            print(f"Median: {t2:.4f} s; GFLOP/s: {g2:.6f}")
            print(f"Max error vs Variant 1: {err12:.3e}")

            print("\nVariant 3: Blocked i-k-j tile sweep")
            tile_results = {}

            for bs in TILES:
                t, samples = benchmark(pool, c, 3, bs)
                result3 = snapshot(c)
                err = max_error(result2, result3)
                g = (2 * N**3) / t / 1e9

                tile_results[str(bs)] = {
                    "median_s": t,
                    "samples_s": samples,
                    "GFLOP_s": g,
                    "max_error_vs_variant2": err,
                    "working_set_bytes": 3 * bs * bs * 8
                }

                print(
                    f"BS={bs:3d}: {t:.4f} s; "
                    f"{g:.6f} GFLOP/s; error={err:.3e}; "
                    f"working set={3 * bs * bs * 8} bytes"
                )

            best_bs = max(
                tile_results,
                key=lambda x: tile_results[x]["GFLOP_s"]
            )
            best_g = tile_results[best_bs]["GFLOP_s"]

            print("\n===== PHASE 3B SUMMARY =====")
            print("Best tile size:", best_bs)
            print("Best tiled GFLOP/s:", f"{best_g:.6f}")
            print("Variant 1 GFLOP/s:", f"{g1:.6f}")
            print("Variant 2 GFLOP/s:", f"{g2:.6f}")
            print("Variant 3 max numerical error:",
                  tile_results[best_bs]["max_error_vs_variant2"])

            result = {
                "N": N,
                "workers": P,
                "variant1": results["variant1"],
                "variant2": results["variant2"],
                "variant1_vs_variant2_max_error": err12,
                "tiles": tile_results,
                "best_tile": int(best_bs),
                "best_tiled_GFLOP_s": best_g
            }

            with open("phase3b_results.json", "w") as f:
                json.dump(result, f, indent=4)

            print("\n===== PHASE 3 VERIFICATION =====")
            print(emit_verify_tag("PHASE3", 230103346, g1, best_g))
            print("\nSaved phase3b_results.json")

    finally:
        del a, b, c
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_c.close()
        shm_c.unlink()

if __name__ == "__main__":
    mp.freeze_support()
    main()
