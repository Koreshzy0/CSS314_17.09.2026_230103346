# Task 24 — Hardware Audit and Empirical-vs-Theoretical Discrepancy

## 1. Machine used for the committed `benchmarks/results.csv` (pipeline smoke test only)
Collected with `lscpu` / `free` in the sandbox where the repo was assembled:

| Item | Value |
|---|---|
| CPU | Intel Xeon (Cascade Lake class, family 6 model 85) @ 2.80 GHz, AVX-512 |
| Visible hardware threads | **1** vCPU (KVM guest, 1 socket, 1 core, no SMT) |
| Cache hierarchy | L1d 32 KiB, L1i 32 KiB, L2 1 MiB, L3 33 MiB (shared with other tenants of the host) |
| NUMA nodes | 1 |
| RAM | ~3 GiB |

**This data is NOT valid for a Gustafson analysis.** With a single vCPU, `P` threads time-share one core, so the
weak-scaled run takes ≈ `P × T(1)` and the scaled speedup is ≈ 1 for every P; the fit gives α ≈ 1.01 (≈ 1 means
"no parallelism at all"). The data only shows that the harness, regression and plotting scripts work. **Re-run on
your own machine** (`python3 benchmarks/run_experiments.py`, then `python3 benchmarks/fit_gustafson.py`) and replace
`results.csv`, `fit_summary.csv`, `plots/scaling_comparison.*`.

## 2. Your machine — fill in after running the commands (MacBook Air, Apple silicon)
```bash
sysctl -n machdep.cpu.brand_string                 # chip name (e.g. Apple M-series)
sysctl hw.physicalcpu hw.logicalcpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
sysctl hw.l1dcachesize hw.l2cachesize hw.cachelinesize hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType | head -30
pmset -g thermlog                                  # shows thermal throttling events during the run
```
| Item | Value (fill in) |
|---|---|
| Chip / P-cores + E-cores | |
| L1d / L2 per cluster / SLC | |
| Memory type, channels, bandwidth | LPDDR, unified memory shared by CPU+GPU |
| NUMA nodes | none (single unified memory pool) |
| Cooling | fanless (Air) → passive heat sink |

## 3. Expected sources of deviation from the Gustafson curve
| Effect | Mechanism | Signature in the data |
|---|---|---|
| **Thermal throttling** | Fanless chassis: sustained all-core load heats the SoC and the clock drops after tens of seconds | Runs get slower later in a sweep; larger std at high P; α_obs inflated; fix by cool-down pauses and randomized thread-count order |
| **Heterogeneous cores** | Apple silicon has performance + efficiency cores; threads beyond the P-core count land on slower E-cores | Kink in S(P) at P = number of P-cores; the static `parallel for` waits for the slowest thread so time(P) rises |
| **Shared memory bandwidth / cache contention** | All cores share one memory system and last-level cache; this kernel is compute-bound so effect is small, but would grow for stream-like kernels | Gradual loss of efficiency with P |
| **NUMA (server machines)** | Threads/data on different sockets pay remote-memory latency | Step in time when P crosses a socket; use `OMP_PROC_BIND`, first-touch init. Not present on a laptop |
| **OS noise / background load** | Scheduler, Spotlight, browser | Outliers; run with the machine idle, report std |
| **SMT / oversubscription** | More threads than cores (or hyper-threads) | Efficiency collapse (see §1: α≈1) |
| **Turbo boost** | Single-thread run is clocked higher than the all-core run | E(P) < 1 even with α = 0; baseline T(1) is optimistic |

## 4. How to read the figure `plots/scaling_comparison.png`
1. Dotted line = ideal S = P. 2. Blue = Gustafson fit with the regression α. 3. Orange points = measured ±1 s.e.
4. Red dashed = Amdahl with the same number as a *fixed-size* serial fraction — it lies far below the weak-scaling
data, illustrating that the same α means different things in the two models (Tasks 03, 08).
If the measured points bend away from the Gustafson line at high P, attribute the deviation using the table above and
compare with the communication-penalty model of Task 12.
