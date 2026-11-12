
import multiprocessing as mp
import math

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103346

STRONG_N = 1_200_000
WEAK_N0 = 200_000

WARMUP = 1
RUNS = 3


# Stage A: serial computation in the parent process
# 5% of N elements, 10 math operations per element
def stage_a(n):

    count = int(0.05 * n)
    acc = 0.5

    for i in range(count):

        x = math.sin(acc + i * 0.000001)
        y = math.sqrt(abs(x) + 1.0)

        acc = math.sin(y) + math.sqrt(abs(x) + 1.0)

        x = math.sin(acc)
        y = math.sqrt(abs(x) + 1.0)

        acc = math.sin(y) + math.sqrt(abs(x) + 1.0)

        x = math.sin(acc)
        y = math.sqrt(abs(x) + 1.0)

        acc = math.sin(y) + math.sqrt(abs(x) + 1.0)

        x = math.sin(acc)
        y = math.sqrt(abs(x) + 1.0)

        acc = math.sin(y) + math.sqrt(abs(x) + 1.0)

    return acc


# Stage B worker: 30 math operations per element
def stage_b_worker(bounds):

    start, end = bounds
    acc = 0.5

    for i in range(start, end):

        x = (i % 1000) * 0.0001 + acc * 0.01

        for j in range(10):

            x = math.sin(x + 0.1)
            y = math.sqrt(abs(x) + 1.0)
            x = math.cos(y + x)

        acc += x * 0.000001

    return acc


# Divide N elements into p approximately equal partitions
def make_partitions(n, p):

    partitions = []

    base = n // p
    remainder = n % p

    start = 0

    for worker in range(p):

        size = base + (1 if worker < remainder else 0)

        end = start + size

        partitions.append((start, end))

        start = end

    return partitions


# Complete two-stage pipeline
def run_pipeline(pool, n, p):

    # Stage A executes serially in the parent
    result_a = stage_a(n)

    # Stage B executes in parallel using the persistent pool
    partitions = make_partitions(n, p)

    results_b = pool.map(stage_b_worker, partitions)

    return result_a + sum(results_b)


def measure_stage_a(n):

    stats = run_benchmark(
        lambda: stage_a(n),
        warmup=WARMUP,
        runs=RUNS
    )

    return stats


# Strong scaling: fixed N
def strong_scaling(logical_cpus):

    print("\n========== STRONG SCALING ==========")
    print("Fixed N:", STRONG_N)

    stage_a_stats = measure_stage_a(STRONG_N)

    stage_a_ms = stage_a_stats.median_ms

    results = {}

    print(
        f"{'p':>5} {'Median ms':>14} "
        f"{'Speedup':>12} {'Efficiency':>12}"
    )

    for p in range(1, logical_cpus + 1):

        print(f"Running strong scaling p={p}...")

        # Pool is created outside the timed benchmark
        with mp.Pool(processes=p) as pool:

            stats = run_benchmark(
                lambda: run_pipeline(pool, STRONG_N, p),
                warmup=WARMUP,
                runs=RUNS
            )

        median = stats.median_ms

        results[p] = median

        if p == 1:
            baseline = median

        speedup = baseline / median
        efficiency = speedup / p

        print(
            f"{p:>5} "
            f"{median:>14.3f} "
            f"{speedup:>12.3f} "
            f"{efficiency:>12.3f}"
        )

    # Amdahl's Law
    serial_fraction = stage_a_ms / results[1]

    amdahl_cap = 1.0 / serial_fraction

    print("\n--- AMDAHL'S LAW ---")
    print(f"Stage A median: {stage_a_ms:.3f} ms")
    print(f"Strong p=1 median: {results[1]:.3f} ms")
    print(f"Serial fraction s: {serial_fraction:.6f}")
    print(f"Amdahl theoretical cap: {amdahl_cap:.3f}")

    final_speedup = results[logical_cpus] / results[logical_cpus]

    final_speedup = results[1] / results[logical_cpus]

    print(
        f"Observed speedup at p={logical_cpus}: "
        f"{final_speedup:.3f}"
    )

    print(
        f"Amdahl cap comparison: "
        f"{final_speedup:.3f} vs {amdahl_cap:.3f}"
    )

    return results, final_speedup, serial_fraction


# Weak scaling: fixed workload per process
def weak_scaling(logical_cpus):

    print("\n========== WEAK SCALING ==========")
    print("Per-worker workload N0:", WEAK_N0)

    results = {}

    print(
        f"{'p':>5} {'N':>12} {'Median ms':>14} "
        f"{'Efficiency':>12} {'Alpha':>12} "
        f"{'Gustafson':>12}"
    )

    baseline = None

    for p in range(1, logical_cpus + 1):

        n = p * WEAK_N0

        print(f"Running weak scaling p={p}, N={n}...")

        # Measure serial Stage A for this weak-scaling workload
        stage_a_stats = measure_stage_a(n)

        stage_a_ms = stage_a_stats.median_ms

        # Persistent pool outside timed loop
        with mp.Pool(processes=p) as pool:

            stats = run_benchmark(
                lambda: run_pipeline(pool, n, p),
                warmup=WARMUP,
                runs=RUNS
            )

        median = stats.median_ms

        if p == 1:
            baseline = median

        weak_efficiency = baseline / median

        alpha = stage_a_ms / median

        gustafson = p - alpha * (p - 1)

        results[p] = {
            "n": n,
            "median": median,
            "stage_a": stage_a_ms,
            "efficiency": weak_efficiency,
            "alpha": alpha,
            "gustafson": gustafson
        }

        print(
            f"{p:>5} "
            f"{n:>12} "
            f"{median:>14.3f} "
            f"{weak_efficiency:>12.3f} "
            f"{alpha:>12.6f} "
            f"{gustafson:>12.3f}"
        )

    final_gustafson = results[logical_cpus]["gustafson"]

    print("\nGustafson scaled speedup at p =", logical_cpus)
    print(f"{final_gustafson:.3f}")

    return results, final_gustafson


def main():

    logical_cpus = mp.cpu_count()

    physical_cores = 8

    print("=" * 65)
    print("PHASE 2: STRONG SCALING VS WEAK SCALING")
    print("=" * 65)

    print("Student ID:", STUDENT_ID)
    print("Physical cores:", physical_cores)
    print("Logical CPUs:", logical_cpus)
    print("Strong N:", STRONG_N)
    print("Weak N0:", WEAK_N0)
    print("Warmup runs:", WARMUP)
    print("Measured runs:", RUNS)

    # 1. Strong scaling
    strong_results, strong_speedup, serial_fraction = (
        strong_scaling(logical_cpus)
    )

    # 2. Weak scaling
    weak_results, gustafson_speedup = (
        weak_scaling(logical_cpus)
    )

    # 3. Architectural inflection
    print("\n========== ARCHITECTURAL INFLECTION ==========")

    if physical_cores + 1 <= logical_cpus:

        t_physical = strong_results[physical_cores]
        t_next = strong_results[physical_cores + 1]

        marginal_speedup = t_physical / t_next

        print(
            f"Marginal speedup from p={physical_cores} "
            f"to p={physical_cores + 1}: "
            f"{marginal_speedup:.3f}"
        )

    else:

        print(
            f"Cannot measure p={physical_cores + 1}: "
            f"logical CPU limit is {logical_cpus}."
        )

        print(
            "Apple M2 reports 8 physical cores and "
            "8 logical CPUs, so no additional logical "
            "CPU is available for this comparison."
        )

    # 4. Verification
    print("\n========== VERIFICATION ==========")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup,
            gustafson_speedup
        )
    )

    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":

    mp.freeze_support()
    main()
