# Task 18 — N-Body Scaling: Direct O(N²) vs. Barnes–Hut O(N log N)

## Derivation: how N must scale with P at constant time per iteration
Per-iteration time on P cores (parallel part only): T = c·W(N)/P. Constant T ⇒ W(N)/P = W(N₀).

| Method | W(N) | Scaling law |
|---|---|---|
| Direct all-pairs | N² | N² = P·N₀² ⇒ **N = N₀·√P** |
| Barnes–Hut | N log N | N log N = P·N₀ log N₀ ⇒ **N ≈ P·N₀·(log N₀/log N)**, i.e. almost linear in P (slightly sub-linear) |

## How the serial/parallel ratio shifts
Let the non-scaling work be proportional to particle count (reading/writing the N particle states, global tree
top-levels, domain decomposition bookkeeping): t_ser = β·N, while the parallel time per iteration stays T₀.
Then α(P) = t_ser/(t_ser + T₀). Example: N₀ = 10⁵ and β·N₀/T₀ = 0.1% (α(1) ≈ 0.1%):

| P | N direct | N Barnes–Hut | α direct | α BH | S direct | S BH |
|---|---|---|---|---|---|---|
| 1 | 100,000 | 100,000 | 0.0010 | 0.0010 | 1 | 1 |
| 4 | 200,000 | 359,955 | 0.0020 | 0.0036 | 4 | 4 |
| 16 | 400,000 | 1,307,920 | 0.0040 | 0.0129 | 16 | 16 |
| 64 | 800,000 | 4,790,168 | 0.0079 | 0.0457 | 64 | 61 |
| 256 | 1,600,000 | 17,662,362 | 0.0157 | 0.1501 | 252 | 218 |
| 1024 | 3,200,000 | 65,504,387 | 0.0310 | 0.3958 | 992 | 619 |
| 4096 | 6,400,000 | 244,167,412 | 0.0602 | 0.7094 | 3850 | 1191 |

**Findings.**
1. Direct method: N grows as √P, so any O(N) serial/communication cost grows as √P — the serial fraction creeps up
   slowly, and the all-gather of positions (O(N) per rank vs. O(N²/P)=const compute per rank) makes the comm/comp ratio
   grow ∝ √P.
2. Barnes–Hut: because each interaction is cheaper, constant time lets N grow almost linearly with P. All O(N) pieces
   (I/O, tree top-levels, load balancing, essential-tree exchange) grow ∝ P — the *serial fraction degrades much
   faster* than in the all-pairs code, and the model prediction α = const fails sooner.
3. Paradox: the better algorithm is *less* weak-scalable. Gustafson's law favours a method that wastes more flops per
   unit of data (more arithmetic per byte, O(N²)/O(N) → compute-bound), which is the classic "scalable but slow" trap;
   the correct comparison metric is time-to-solution at equal N, not scalability curves.
