# Task 09 — Multi-Component Workload

| Phase | Share of parallel runtime | Treated as |
|---|---|---|
| A (disk I/O + broadcast) | 2% | serial |
| B (boundary sync) | 6% | serial |
| C (flux computation) | 92% | parallel |

Assumption (state it explicitly): phases A and B execute with no speedup from the added cores (I/O and
synchronization are modelled as serial / non-scaling), phase C is perfectly parallel.

**α = 0.02 + 0.06 = 0.08**, p = 0.92.

| P | S(P) = P − 0.08(P−1) | E(P) |
|---|---|---|
| 512 | 512 − 0.08·511 = **471.12** | 0.9202 |
| 2048 | 2048 − 0.08·2047 = **1884.24** | 0.9200 |

Composite view: the 8% non-scaling share costs about 8% of the machine at any P (E → 0.92). Phase B (6%)
is the larger target; halving it raises the limit E from 0.92 to 0.95. Caveat: boundary synchronization normally grows with P
(Task 12), in which case B should be modelled as B(P) and α would increase with P.
