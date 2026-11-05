#!/usr/bin/env python3
"""Fallback (no compiler needed): same weak-scaling experiment with Python multiprocessing + numpy.
usage: python3 weak_scale.py THREADS [K_per_worker=20000000] [serial_steps=1000000] [runs=5] [header=0]"""
import sys, time, multiprocessing as mp
import numpy as np

def work(args):
    first, last, h = args
    CH = 2_000_000; s = 0.0
    for a in range(first, last, CH):
        i = np.arange(a, min(a + CH, last), dtype=np.float64); x = (i + 0.5) * h; s += float(np.sum(4.0 / (1.0 + x * x)))
    return s

if __name__ == "__main__":
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    K = int(float(sys.argv[2])) if len(sys.argv) > 2 else 20_000_000
    SER = int(float(sys.argv[3])) if len(sys.argv) > 3 else 1_000_000
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    if len(sys.argv) > 5 and sys.argv[5] == "1": print("threads,run,time_s,pi_estimate")
    h = 1.0 / (K * P)
    with mp.Pool(P) as pool:
        for r in range(runs):
            t0 = time.perf_counter()
            work((0, SER, 1.0 / SER))                                   # serial section
            pi = sum(pool.map(work, [(t * K, (t + 1) * K, h) for t in range(P)])) * h
            print(f"{P},{r},{time.perf_counter()-t0:.6f},{pi:.10f}", flush=True)
