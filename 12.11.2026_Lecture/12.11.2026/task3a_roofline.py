import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import json
import os

P = 8
SIZE = 96 * 1024 * 1024
FLOPS_PER_WORKER = 80_000_000

def copy_worker(args):
    name_a, name_b, start, end = args
    a = shared_memory.SharedMemory(name=name_a)
    b = shared_memory.SharedMemory(name=name_b)
    mv_a = a.buf
    mv_b = b.buf
    mv_a[start:end] = mv_b[start:end]
    del mv_a, mv_b
    a.close()
    b.close()
    return end - start

def compute_worker(flops):
    loops = flops // 32
    a = 1.0000001
    b = 0.0000001
    x0=x1=x2=x3=x4=x5=x6=x7=1.0
    x8=x9=x10=x11=x12=x13=x14=x15=1.0

    for _ in range(loops):
        x0=x0*a+b; x1=x1*a+b
        x2=x2*a+b; x3=x3*a+b
        x4=x4*a+b; x5=x5*a+b
        x6=x6*a+b; x7=x7*a+b
        x8=x8*a+b; x9=x9*a+b
        x10=x10*a+b; x11=x11*a+b
        x12=x12*a+b; x13=x13*a+b
        x14=x14*a+b; x15=x15*a+b

    return x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15

def median_time(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e9)
    return statistics.median(times), times

def main():
    print("===== PHASE 3A: ROOFLINE =====")
    print("Workers:", P)
    print("Buffer size per array:", SIZE / 1024 / 1024, "MiB")

    shm_a = shared_memory.SharedMemory(create=True, size=SIZE)
    shm_b = shared_memory.SharedMemory(create=True, size=SIZE)

    try:
        shm_b.buf[:] = b'\x01' * SIZE

        chunk = SIZE // P
        tasks = []
        for i in range(P):
            start = i * chunk
            end = SIZE if i == P - 1 else (i + 1) * chunk
            tasks.append((shm_a.name, shm_b.name, start, end))

        with mp.Pool(P) as pool:
            def copy_all():
                pool.map(copy_worker, tasks)

            copy_median, copy_samples = median_time(copy_all)
        
        bandwidth = (2 * SIZE) / copy_median / 1e9

        print("\nDRAM bandwidth:")
        print("Times (seconds):", copy_samples)
        print(f"Median: {copy_median:.6f} s")
        print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()

    print("\nMeasuring Python compute ceiling...")
    with mp.Pool(P) as pool:
        def compute_all():
            return pool.map(compute_worker, [FLOPS_PER_WORKER] * P)

        compute_median, compute_samples = median_time(compute_all)

    total_flops = FLOPS_PER_WORKER * P
    compute_gflops = total_flops / compute_median / 1e9

    print("\nPython compute ceiling:")
    print("Times (seconds):", compute_samples)
    print(f"Median: {compute_median:.6f} s")
    print(f"Total FLOPs: {total_flops}")
    print(f"Compute ceiling: {compute_gflops:.4f} GFLOP/s")

    ridge = compute_gflops / bandwidth

    result = {
        "workers": P,
        "buffer_bytes_each": SIZE,
        "bandwidth_GB_s": bandwidth,
        "bandwidth_median_s": copy_median,
        "bandwidth_samples_s": copy_samples,
        "compute_FLOPs_per_worker": FLOPS_PER_WORKER,
        "compute_median_s": compute_median,
        "compute_samples_s": compute_samples,
        "compute_GFLOP_s": compute_gflops,
        "ridge_point_FLOP_per_byte": ridge
    }

    with open("phase3a_results.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nEmpirical ridge point:", f"{ridge:.4f} FLOP/Byte")
    print("\nSaved phase3a_results.json")

if __name__ == "__main__":
    mp.freeze_support()
    main()
