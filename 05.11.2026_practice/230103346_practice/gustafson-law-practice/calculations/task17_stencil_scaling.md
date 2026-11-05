# Task 17 — 3D 7-Point Stencil: Surface-to-Volume Under Weak Scaling

**Decomposition.** Cubic domain N³ split over P ranks in a ∛P×∛P×∛P grid; each rank owns a sub-cube of side n = N/∛P.

* Computation per step ∝ n³ (interior updates, ~8 flops/point for 7-point stencil).
* Communication per step ∝ 6n² (six faces, halo depth 1) — O(N²) at the global level, but only the local surface matters.
* Communication-to-computation ratio **ρ = 6n²/n³ = 6/n**.

**Weak scaling (Gustafson).** Work per rank constant ⇒ n = n₀ constant ⇒ N = n₀·P^(1/3) and

ρ_weak = 6/n₀ = const **independent of P**.

| P | Global N | Local volume | Local surface | ρ = surface/volume |
|---|---|---|---|---|
| 1 | 128 | 128³ = 2,097,152 | 6·128² = 98,304 | 0.0469 |
| 8 | 256 | 128³ = 2,097,152 | 6·128² = 98,304 | 0.0469 |
| 64 | 512 | 128³ = 2,097,152 | 6·128² = 98,304 | 0.0469 |
| 512 | 1024 | 128³ = 2,097,152 | 6·128² = 98,304 | 0.0469 |
| 4096 | 2048 | 128³ = 2,097,152 | 6·128² = 98,304 | 0.0469 |

**Strong scaling for contrast** (global N fixed = 1024): n = 1024/P^(1/3), so ρ_strong = 6P^(1/3)/1024 **grows** as P^(1/3):

| P | n | ρ_strong |
|---|---|---|
| 1 | 1024.0 | 0.0059 |
| 8 | 512.0 | 0.0117 |
| 64 | 256.0 | 0.0234 |
| 512 | 128.0 | 0.0469 |
| 4096 | 64.0 | 0.0937 |

**Conclusion.** Under weak scaling the ratio neither improves nor degrades — it is *constant*, which is why stencil
codes are the textbook Gustafson-friendly workload (the nCUBE applications were of this type). Two remaining
degradations: (i) latency-bound neighbor exchange and global reductions (e.g. residual norms, O(log P)) add α(P);
(ii) if the global time step must shrink with N (CFL), constant runtime fails (Task 13).
