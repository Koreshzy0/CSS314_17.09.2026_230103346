import multiprocessing as mp
import math
import time
import json
from harness import emit_verify_tag

STUDENT_ID = 230103346
P_VALUES = list(range(1, mp.cpu_count() + 1))
N_STRONG = 1_600_000
N0_WEAK = 250_000
CHUNK = 10_000

def work_range(args):
    start, end = args
    total = 0.0
    for i in range(start, end):
        x = (i % 1000 + 1) * 0.001
        for j in range(35):
            x = x * 1.000001 + math.sin(x) * 0.00001
        total += x
    return total

def serial_stage(n):
    total = 0.0
    count = max(1, int(n * 0.05))
    for i in range(count):
        x = (i % 1000 + 1) * 0.001
        for _ in range(12):
            x = math.sin(x) + math.sqrt(abs(x) + 1.0)
        total += x
    return total

def parallel_stage(pool, n):
    ranges = [
        (start, min(start + CHUNK, n))
        for start in range(0, n, CHUNK)
    ]
    return sum(pool.map(work_range, ranges))

def run_trial(pool, n):
    start = time.perf_counter_ns()
    serial_stage(n)
    serial_end = time.perf_counter_ns()
    parallel_stage(pool, n)
    end = time.perf_counter_ns()

    return (
        (end - start) / 1e6,
        (serial_end - start) / 1e6
    )

def median(values):
    values = sorted(values)
    n = len(values)
    if n % 2:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2

def measure(p, n):
    with mp.Pool(processes=p) as pool:
        run_trial(pool, n)  # warm-up
        samples = [run_trial(pool, n) for _ in range(3)]

    return (
        median([x[0] for x in samples]),
        median([x[1] for x in samples])
    )

def main():
    strong = {}
    weak = {}

    print("===== PHASE 2: STRONG SCALING =====")
    print("N =", N_STRONG)

    for p in P_VALUES:
        total, serial = measure(p, N_STRONG)
        strong[str(p)] = {
            "time_ms": total,
            "serial_ms": serial
        }
        print(f"p={p}: time={total:.3f} ms, serial={serial:.3f} ms")

    t1 = strong["1"]["time_ms"]
    s_fraction = strong["1"]["serial_ms"] / t1
    s_infinity = 1 / s_fraction

    print("\n===== STRONG SCALING TABLE =====")
    for p in P_VALUES:
        t = strong[str(p)]["time_ms"]
        speedup = t1 / t
        efficiency = speedup / p
        strong[str(p)].update({
            "speedup": speedup,
            "efficiency": efficiency
        })
        print(
            f"p={p}: T={t:.3f}, "
            f"Speedup={speedup:.4f}, Efficiency={efficiency:.4f}"
        )

    print("\nSerial fraction:", s_fraction)
    print("Amdahl maximum speedup:", s_infinity)

    print("\n===== PHASE 2: WEAK SCALING =====")

    for p in P_VALUES:
        n = p * N0_WEAK
        total, serial = measure(p, n)
        alpha = serial / total
        gustafson = p - alpha * (p - 1)

        weak[str(p)] = {
            "n": n,
            "time_ms": total,
            "serial_ms": serial,
            "alpha": alpha,
            "gustafson_speedup": gustafson
        }

        print(
            f"p={p}: N={n}, T={total:.3f} ms, "
            f"alpha={alpha:.4f}, Gustafson={gustafson:.4f}"
        )

    weak_t1 = weak["1"]["time_ms"]

    for p in P_VALUES:
        weak[str(p)]["efficiency"] = (
            weak_t1 / weak[str(p)]["time_ms"]
        )

    observed = strong[str(P_VALUES[-1])]["speedup"]
    gustafson_final = weak[str(P_VALUES[-1])]["gustafson_speedup"]

    with open("phase2_results.json", "w") as f:
        json.dump({
            "strong": strong,
            "weak": weak,
            "serial_fraction": s_fraction,
            "amdahl_maximum": s_infinity
        }, f, indent=4)

    print("\n===== PHASE 2 VERIFICATION =====")
    print(emit_verify_tag(
        "PHASE2", STUDENT_ID, observed, gustafson_final
    ))
    print("\nSaved phase2_results.json")

if __name__ == "__main__":
    mp.freeze_support()
    main()
