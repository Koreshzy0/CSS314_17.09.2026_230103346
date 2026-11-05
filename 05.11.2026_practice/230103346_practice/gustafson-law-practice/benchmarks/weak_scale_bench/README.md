# Weak-scaling benchmark (Task 22)

**What it does.** Integrates 4/(1+x²) over [0,1] (= π). Every thread gets the same number of steps `K`
(problem size per thread constant), total work grows ∝ P. A constant *serial* section of `SERIAL_STEPS` steps
runs on one thread; it plays the role of Gustafson's non-scaling part (default: 2·10⁶ steps vs. 5·10⁷ per thread ⇒ ideal α ≈ 3.8%).

## Build & run
```bash
# Linux
make
# macOS (MacBook Air, Apple clang has no OpenMP built in)
brew install libomp && make

./weak_scale THREADS [K=50000000] [SERIAL_STEPS=2000000] [RUNS=5] [PRINT_HEADER=0]
./weak_scale 4 50000000 2000000 5 1        # example: 4 threads, 5 runs, with header

# full experiment (compiles, sweeps thread counts, 5 runs each, writes benchmarks/results.csv)
python3 ../run_experiments.py
# no-compiler fallback:  python3 weak_scale.py 4
```
Output columns: `threads,run,time_s,pi_estimate`. The pi estimate is a correctness check (should be ≈ 3.14159265).
Pin/limit threads explicitly if needed: `OMP_PROC_BIND=close OMP_PLACES=cores`.
