import time
import multiprocessing as mp
from harness import emit_verify_tag

STUDENT_ID = 230103346
CLOCK_READS = 2_000_000

def main():
    cpu_count = mp.cpu_count()

    # Warm-up
    for _ in range(10000):
        time.perf_counter_ns()

    start = time.perf_counter_ns()

    for _ in range(CLOCK_READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    elapsed_ns = end - start
    average_ns = elapsed_ns / CLOCK_READS

    print("===== PHASE 0 CALIBRATION =====")
    print("Student ID:", STUDENT_ID)
    print("Logical CPUs:", cpu_count)
    print("Clock reads:", CLOCK_READS)
    print("Total time (ns):", elapsed_ns)
    print("Average clock call latency (ns):", average_ns)

    verification = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_ns,
        float(CLOCK_READS)
    )

    print(verification)

if __name__ == "__main__":
    main()
