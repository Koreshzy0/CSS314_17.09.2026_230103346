# Task 16 — GEMM Scaling Under Constant Runtime

**Derivation.** Work W(N) = 2N³ flops. With P cores at rate r each and ideal parallel execution, T = 2N³/(P r).
Gustafson fixes T = T₀ = 2N₀³/r (the single-core time of the baseline problem N₀). Hence

N³/P = N₀³ ⇒ **N(P) = N₀ · P^(1/3)**.

Memory: total = 3·8·N² bytes ∝ P^(2/3); **per core ∝ N²/P = N₀²·P^(−1/3)** — it *decreases*, so GEMM is
compute-bound and memory is not the limiter (memory-bounded Sun–Ni scaling would allow N ∝ P^(1/2), with runtime ∝ P^(1/2)).

Baseline N₀ = 1000 (matrix A only, fp64, per-core share of one matrix):

| P | N(P) | N/N₀ | A-matrix per core |
|---|---|---|---|
| 1 | 1,000 | 1.000 | 8.00 MB |
| 8 | 2,000 | 2.000 | 4.00 MB |
| 64 | 4,000 | 4.000 | 2.00 MB |
| 512 | 8,000 | 8.000 | 1.00 MB |
| 4096 | 16,000 | 16.000 | 0.50 MB |
| 32768 | 32,000 | 32.000 | 0.25 MB |

Practical caveat: a distributed GEMM must also move data. With 2D SUMMA the communication volume per core ∝ N²/√P =
N₀²P^(2/3)/P^(1/2) = N₀²P^(1/6), so communication grows slowly relative to computation per core — weak scaling holds but
isn't perfect, i.e. α(P) creeps upward.
