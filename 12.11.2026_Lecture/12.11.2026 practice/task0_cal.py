import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    # 1. Get logical CPU count
    logical_cpus = mp.cpu_count()
    print("Logical CPUs:", logical_cpus)

    # 2. Measure timer latency
    n = 10**6

    start = time.perf_counter_ns()

    for _ in range(n):
        time.perf_counter_ns()

    elapsed = time.perf_counter_ns() - start
    average_latency = elapsed / n

    print("Number of calls:", n)
    print("Total time (ns):", elapsed)
    print("Average latency (ns/call):", average_latency)

    # 3. Generate verification hash
    verify_tag = emit_verify_tag(
        "PHASE0",
        230103346,
        average_latency,
        elapsed / 1e6
    )

    print(verify_tag)


if __name__ == "__main__":
    main()