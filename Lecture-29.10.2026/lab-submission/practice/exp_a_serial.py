import time, math
from multiprocessing import Pool
from karp_flatt import speedup, karp_flatt

def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))

def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  # Intentional sequential bottleneck
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0

if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {p: run_experiment(p) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in (2, 4):
        tp = runtimes[p]
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={speedup(t1, tp):.2f}x | KarpFlatt e={karp_flatt(t1, tp, p):.3f}")
    print(f"Serial fraction (0.40/T1) = {0.40 / t1:.3f}")
    e2, e4 = karp_flatt(t1, runtimes[2], 2), karp_flatt(t1, runtimes[4], 4)
    print(f"e(4)/e(2) = {e4 / e2:.2f}  (rubric: 0.80-1.25)")
