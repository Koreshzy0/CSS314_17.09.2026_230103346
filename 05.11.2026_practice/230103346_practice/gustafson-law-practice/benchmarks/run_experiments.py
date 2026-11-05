#!/usr/bin/env python3
"""Task 23 driver: sweep thread counts with the OpenMP benchmark, 5 independent runs each -> benchmarks/results.csv

  python3 benchmarks/run_experiments.py                    # thread counts up to os.cpu_count()
  python3 benchmarks/run_experiments.py --threads 1 2 4 6 8
  python3 benchmarks/run_experiments.py --allow-oversubscribe   # smoke test on machines with < 5 usable counts
"""
import argparse, os, platform, subprocess, sys, pathlib, csv
HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE / "weak_scale_bench"

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, nargs="+")
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--k", type=int, default=50_000_000, help="steps per thread")
ap.add_argument("--serial", type=int, default=2_000_000, help="serial steps")
ap.add_argument("--out", default=str(HERE / "results.csv"))
ap.add_argument("--allow-oversubscribe", action="store_true")
a = ap.parse_args()

subprocess.run(["make", "-C", str(BENCH)], check=True)
hw = os.cpu_count() or 1
counts = a.threads or sorted({t for t in (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64) if t <= hw} | {hw})
if len(counts) < 5:
    if not a.allow_oversubscribe:
        sys.exit(f"Only {len(counts)} distinct thread counts <= {hw} hardware threads; pass --threads or --allow-oversubscribe")
    counts = sorted(set(counts) | {1, 2, 4, 8, 16}); print("WARNING: oversubscribed runs - NOT valid for real scaling analysis", file=sys.stderr)
host = f"{platform.machine()}|{platform.system()}|hw_threads={hw}" + ("|OVERSUBSCRIBED" if max(counts) > hw else "")
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["host", "threads", "run", "time_s", "pi_estimate"])
    for P in counts:
        res = subprocess.run([str(BENCH / "weak_scale"), str(P), str(a.k), str(a.serial), str(a.runs), "0"], capture_output=True, text=True, check=True)
        for line in res.stdout.strip().splitlines():
            t, r, ts, pi = line.split(","); w.writerow([host, t, r, ts, pi])
        print(f"P={P}: done", flush=True)
print("wrote", a.out)
