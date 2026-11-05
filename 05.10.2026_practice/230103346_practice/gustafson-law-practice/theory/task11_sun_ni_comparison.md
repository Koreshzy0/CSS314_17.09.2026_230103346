# Task 11 — Gustafson's Law vs. the Sun–Ni Memory-Bounded Model

## Formulations
Let the baseline workload be $W=W_s+W_p$ (serial + parallelizable), $f=W_s/W$. Let $G(P)$ be the factor by which the
*parallel* part grows when $P$ nodes (each with memory $M$) are used.

$$ S^*(P)=\frac{W_s+G(P)\,W_p}{W_s+G(P)\,W_p/P}=\frac{f+(1-f)\,G(P)}{f+(1-f)\,G(P)/P}. $$

| $G(P)$ | Resulting law |
|---|---|
| $1$ | $S=\dfrac{1}{f+(1-f)/P}$ — **Amdahl** (fixed size) |
| $P$ | $S=f+(1-f)P$ — **Gustafson** (fixed time) |
| $G(P)$ set by memory: workload grows as the memory $PM$ allows | **Sun–Ni** (memory-bounded) |

Memory-driven examples: if work $\propto N^3$ and memory $\propto N^2$ (dense matrix ops), $N\propto\sqrt P$ so
$G(P)=P^{3/2}$; for 3D grids (work $\propto N^3$, memory $\propto N^3$) $G(P)=P$; if memory $\propto N^3$ but work
$\propto N^4$ (stencil with CFL time stepping) $G(P)=P^{4/3}$.

## When Gustafson (G = P) is wrong
**Overestimates** achievable throughput (true $G<P$, i.e. the problem cannot grow proportionally):
* memory per node is insufficient or only a part of the memory is usable (replicated data, halo cells, index structures);
* work grows slower than the data — e.g. bandwidth-bound codes where more data does not mean proportionally more
  parallel flops, or when a fixed-size input (a given dataset) is all that exists;
* communication or contention grows with $P$ (Task 12), so the constant-time premise is not met.

**Underestimates** (true $G>P$):
* Work grows faster than memory: e.g. dense linear algebra, $G=P^{3/2}$. The Sun–Ni speedup
  $S^*\approx (1-f)P^{3/2}/ (f + (1-f)P^{1/2})$ is *superlinear in P* — but only because the runtime is allowed to
  grow ($T\propto P^{1/2}$); the fixed-time constraint of Gustafson would force $N\propto P^{1/3}$ instead (Task 16).
* Cache effects: when the per-core working set shrinks into cache, real speedup can exceed $P$.

## Takeaway
Gustafson is the $G(P)=P$ member of a family. Amdahl is the pessimistic ($G=1$) end, memory-bound
scaling ($G>P$ for compute-heavy kernels) the optimistic end; the right member is determined by the algorithm's
work-vs-memory relation and by whether a time budget is binding.
