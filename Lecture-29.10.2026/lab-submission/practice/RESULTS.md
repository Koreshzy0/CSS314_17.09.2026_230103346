# Karp–Flatt Practice Lab — RESULTS

## Hardware profile
- Machine: MacBook Air (TODO: chip, e.g. Apple M1/M2)
- Logical cores (`os.cpu_count()`): TODO
- OS: macOS (TODO: version)
- Python: TODO (`python3 --version`)

## Summary table

| Experiment | Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|---|
| A | Serial Tail | 3.469 | 1.998 | 1.260 | 0.152 | 0.151 | Constant e → Fixed Amdahl serialization |
| B | Lock Contention | 3.917 | 11.846 | 12.045 | 5.048 | 3.767 | Exploding e, negative speedup → Mutex & bus thrashing |
| C | Privatized State | 0.330 | 0.194 | 0.125 | 0.176 | 0.172 | Speedup restored (E4 = 66%); see note on startup overhead |

## Experiment A: The Stubborn Serial Tail
Speedup was 1.74x on 2 workers and 2.75x on 4 workers. The Karp–Flatt metric
stayed essentially constant (e(2) = 0.152, e(4) = 0.151, ratio e(4)/e(2) = 1.00,
inside the rubric window [0.80, 1.25]). A flat e(p) is the signature of a fixed
serial code path rather than contention. The intentional 0.40 s sleep is about
11.5% of T1, consistent with the measured e; the small excess comes from
process-pool startup and task dispatch, which are also serial. The performance
loss is therefore purely algorithmic (Amdahl).

## Experiment B: The Contention Traffic Jam
Adding cores made the program slower: T4 = 12.045 s versus T1 = 3.917 s
(speedup 0.33x, i.e. negative scaling). The Karp–Flatt values are far above the
expected thresholds (e(2) = 5.048, e(4) = 3.767; a repeat run gave 4.908 and
3.751), so the loss cannot be explained by a small serial code fraction.
Every iteration acquires one shared mutex, so the lock and the cache line
holding the counter bounce between cores and processes stall waiting for the
lock. Runtime saturates at roughly 12 s for both P = 2 and P = 4 because the
lock's hand-off throughput is already exhausted; as e(p) divides by (p − 1),
e(4) is therefore slightly lower than e(2) instead of growing further. The
diagnosis is lock thrashing and coherence traffic, not sequential algorithm code.

## Experiment C: The Privatization Fix
Each worker accumulates into a private local variable and results are combined
once at the end. This removes the contention: T4 = 0.125 s < T1 = 0.330 s,
speedup 2.64x at P = 4 and 1.70x at P = 2, with parallel efficiency
E4 = 66%. e(p) dropped from 3.8–5.0 (Experiment B) to about 0.17. The remaining
e is mostly fixed process-pool startup cost, which is large relative to a
sub-second run; a larger workload amortizes it.

## How to reproduce
```
python3 exp_a_serial.py
python3 exp_b_contention.py
python3 exp_c_fixed.py
```
