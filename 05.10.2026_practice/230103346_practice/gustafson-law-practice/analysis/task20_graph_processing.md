# Task 20 — Why Weak Scaling Is Hard for Graph Analytics (BFS, PageRank)

**Reference point — dense linear algebra / stencils:** regular access, work and data volume both known in advance,
surface/volume ratio constant (Task 17), balanced partitions, flops/byte ratio high ⇒ $\alpha$ constant, runtime
constant under growth ⇒ Gustafson holds.

## Difficulties on power-law graphs
1. **No well-defined "work per core" to keep constant.** Weak scaling needs a problem unit with fixed cost. For graphs
   the cost depends on the *edge structure*, not on vertex count: edges per vertex vary by orders of magnitude (hubs).
   Doubling $|V|$ and $|E|$ does not double diameter, frontier shape or cache behaviour.
2. **Frontier imbalance.** BFS work per level is data-dependent: few vertices in the first and last levels, a huge
   frontier in the middle (the "explosion" at low diameter, ~6–10 levels for social graphs). Cores idle in thin levels ⇒
   large effective serial fraction per level, and a barrier per level ⇒ $\alpha$ grows with the number of levels
   (which in turn grows with graph size, $\propto \log N$ or more for road networks).
3. **Degree skew and partitioning.** Hubs create hot partitions; vertex-cut partitioning replicates hub vertices and
   adds synchronization (mirror updates). Load imbalance factor $\lambda = \max_i W_i/\bar W>1$ increases with $P$
   because the maximum of more samples from a heavy tail grows: runtime $T\approx\lambda(P)T_0$ ⇒ not constant.
4. **Irregular memory access.** Random neighbour lookups ⇒ cache/TLB misses, latency-bound rather than flop-bound;
   throughput is limited by memory/network *message rate*, not compute. Adding cores adds little when each core stalls
   on random reads. Arithmetic intensity ~0.1 flop/byte vs. GEMM's O(N).
5. **Communication does not shrink with partitioning.** Edge cut is large for power-law graphs (no good small
   separators, unlike planar/mesh graphs where cut $\sim N^{2/3}$): the cut grows ∝ $|E|$ so
   comm/comp does *not* stay constant when scaling weakly, and all-to-all frontier exchange scales as $O(P)$ messages
   (or needs 2D partitioning, $O(\sqrt P)$).
6. **Data-dependent synchronization.** PageRank iterates until convergence: iteration count depends on graph structure
   and size (spectral gap), BFS has level-synchronous barriers, asynchronous variants have races/staleness — so
   the constant-time premise (Task 13) fails.
7. **Not true weak scaling in benchmarks:** Graph500 scales $|V|=2^{scale}$ with edge factor 16, but BFS time/step still
   grows because of 2–5 above; reports use TEPS (traversed edges/s) instead of speedup.

## Summary table
| Property | Dense LA / stencil | Power-law graph |
|---|---|---|
| Work per unit of data | known, regular | data-dependent |
| Load balance as $P$ grows | stays ~1 | gets worse (heavy tails) |
| Comm/comp under scaling | constant | grows (large cuts, all-to-all) |
| Memory behaviour | streaming, cache-blocked | random, latency-bound |
| Iterations / levels | fixed | grow with size, convergence-dependent |
| $\alpha(P)$ | ≈ constant | increasing |

**Mitigations:** 2D edge partitioning, direction-optimizing BFS, degree-aware delegation of hubs, asynchronous/bulk-
synchronous hybrid, graph reordering for locality, and judging by TEPS/efficiency relative to a model like Task 12
rather than constant-$\alpha$ Gustafson.
