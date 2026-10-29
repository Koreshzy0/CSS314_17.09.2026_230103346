import time
from multiprocessing import Pool
from karp_flatt import speedup, karp_flatt

def private_worker(iterations):
    local_sum = 0  # private state, zero locks
    for _ in range(iterations):
        local_sum += 1
    return local_sum

def run_privatized(p, total_increments=20_000_000):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # synchronize exactly ONCE
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0

if __name__ == "__main__":
    cores = [1, 2, 4]
    runtimes = {p: min(run_privatized(p) for _ in range(3)) for p in cores}  # best of 3
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = speedup(t1, tp)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Efficiency={sp/p:.0%} | KarpFlatt e={karp_flatt(t1, tp, p):.3f}")
