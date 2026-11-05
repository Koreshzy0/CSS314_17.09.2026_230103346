#!/usr/bin/env python3
"""Generates tables embedded in Task 12, 18, 19 deliverables. Run from repo root."""
import math, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
Ps = [2**k for k in range(0, 21, 2)]  # 1 .. 1,048,576

# ---------- Task 12: modified Gustafson with communication term ----------
def S_comm(P, a, C): return (a + (1 - a) * P) / (1 + C)
a, g = 0.04, 0.01
rows = ""
for P in [16, 64, 256, 1024, 4096, 16384, 65536, 1048576]:
    S0 = P - a * (P - 1)
    Cl, Cs = g * math.log2(P), 0.001 * math.sqrt(P)
    rows += f"| {P:,} | {S0/P:.4f} | {S_comm(P,a,Cl)/P:.4f} | {S_comm(P,a,Cs)/P:.4f} |\n"
t12 = r"""# Task 12 — Gustafson Model with Communication Overhead C(P)

## Model
Take the parallel run of the *baseline* (communication-free) schedule as one unit, s + p = 1, and add the time the
machine must spend communicating, which grows with P:

$$T_P = s + p + C(P) = 1 + C(P),\qquad C(P)=\gamma\,f(P).$$

The scaled workload is unchanged (work still s + pP in single-core units), so

$$\boxed{S_C(P)=\frac{s+pP}{1+C(P)}=\frac{P-\alpha(P-1)}{1+C(P)}},\qquad E_C(P)=\frac{1-\alpha+\alpha/P}{1+C(P)}.$$

Equivalent "effective serial fraction" form: S_C = P − α_eff(P)(P−1) with
α_eff = (P − S_C)/(P−1) = [α + (P−α... )] — in practice we fit α_eff(P) from data (Task 23).
For C = 0 the classical law is recovered.

## Choices of f(P)
* f(P) = log₂ P — tree collectives / reductions (latency term L·log P).
* f(P) = √P — 2D nearest-neighbor mesh diameter / bisection-limited all-to-all on a 2D torus.
* f(P) = P^(1/3) — 3D torus diameter (not tabulated).

## Asymptotics
For f = log₂ P: S_C ≈ (1−α)P/(1+γ log₂ P), so E_C → **0 as 1/(γ log₂ P)** — slowly, but the 1−α floor of classical
Gustafson no longer exists. For f = √P: E_C → (1−α)/(γ√P) → 0 faster; the speedup then grows only like √P.
Weak scaling survives only if C(P) stays bounded, i.e. communication per core must be O(1) (nearest-neighbor halo
exchange with constant surface/volume, Task 17).

## Numeric example (α = 0.04, γ = 0.01 for log₂ P; γ = 0.001 for √P)

| P | E (classical) | E, C = 0.01·log₂P | E, C = 0.001·√P |
|---|---|---|---|
@@ROWS@@
Reading: with only 1% per log₂-level of overhead, efficiency at one million cores is @@E1@@ instead of 0.96,
and the √P model drops to @@E2@@.
"""
t12 = t12.replace('@@ROWS@@', rows).replace('@@E1@@', f'{S_comm(2**20,a,g*20)/2**20:.2f}').replace('@@E2@@', f'{S_comm(2**20,a,0.001*1024)/2**20:.2f}')
t12 = t12.replace("[α + (P−α... )] — in practice we fit α_eff(P) from data (Task 23).", "(S_C is measured in Task 23 and α_eff(P) is fitted from it).")
(ROOT/"theory"/"task12_communication_penalty.md").write_text(t12)

# ---------- Task 18: N-body ----------
N0, beta_ratio = 1e5, 0.001   # beta*N0/T0 = 0.1% serial share at P=1
def N_bh(P):
    target = P * N0 * math.log2(N0); N = N0 * P
    for _ in range(200): N = target / math.log2(N)
    return N
rows = ""
for P in [1, 4, 16, 64, 256, 1024, 4096]:
    Nap, Nbh = N0 * math.sqrt(P), N_bh(P)
    fa, fb = beta_ratio * Nap / N0, beta_ratio * Nbh / N0       # serial time / parallel time
    al_a, al_b = fa / (1 + fa), fb / (1 + fb)
    Sa, Sb = P - al_a * (P - 1), P - al_b * (P - 1)
    rows += f"| {P} | {Nap:,.0f} | {Nbh:,.0f} | {al_a:.4f} | {al_b:.4f} | {Sa:.0f} | {Sb:.0f} |\n"
t18 = f"""# Task 18 — N-Body Scaling: Direct O(N²) vs. Barnes–Hut O(N log N)

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
{rows}
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
"""
(ROOT/"analysis"/"task18_nbody_scaling.md").write_text(t18)

# ---------- Task 19: Monte Carlo ----------
T0 = 10.0                      # compute seconds per rank (constant under weak scaling)
L, Lflat, tseed, tout = 20e-6, 5e-6, 1e-6, 50e-6
rows = ""
for P in Ps:
    tree = math.ceil(math.log2(P)) * L if P > 1 else 0
    flat = (P - 1) * Lflat
    seed = P * tseed
    out = P * tout
    E = lambda ov: T0 / (T0 + ov)
    rows += f"| {P:,} | {tree*1e3:.3f} | {E(tree):.5f} | {E(flat):.5f} | {E(tree+seed):.5f} | {E(tree+out):.5f} | {E(tree+seed+out):.5f} |\n"
t19 = f"""# Task 19 — Monte Carlo: the ideal weak-scaling workload

## Where does the non-zero serial fraction α come from?
| Source | Typical cost model | Scaling | Remedy |
|---|---|---|---|
| PRNG seed management | master generates P seeds / jump-ahead state sequentially: t_seed·P | O(P) | counter-based RNG (Philox/Threefry) or hash(seed, rank): O(1), reproducible |
| Statistical coupling of streams | leapfrog/skip-ahead and stream correlation checks | O(1)–O(P) | independent substreams (SplitMix/SeedSequence.spawn) |
| Reduction of partial estimates | tree: ⌈log₂P⌉·(L + m/B); flat (to root): (P−1)·L | O(log P) vs O(P) | tree/Rabenseifner reduce, hierarchical (node → rack) |
| Output serialization | rank-0 collects/writes P records: t_out·P | O(P) | parallel I/O, write only aggregated statistics |
| Start-up/finalize (MPI_Init, load config) | job launcher cost | O(1)–O(log P) | constant at fixed α |

## Impact of the reduction tree on S(P)
Compute per rank is constant (T₀ = {T0:g} s). Non-scaling overhead O(P) (seconds) turns the classical S = P − α(P−1)
into S_P = P·T₀/(T₀ + overhead(P)), i.e. α_eff(P) = overhead/(T₀+overhead). With tree reduction overhead(P) = ⌈log₂P⌉·L:
at P = 2²⁰ it is only 20·L = {20*L*1e3:.2f} ms, α_eff ≈ {20*L/(T0+20*L):.1e}: **a tree reduction costs essentially nothing**
and Monte Carlo scales almost perfectly (S ≈ P).

## Numeric model (L = 20 μs per tree level, 5 μs per message for the flat reduce, t_seed = 1 μs, t_out = 50 μs per rank)

Parallel efficiency E = T₀/(T₀ + overhead):

| P | tree overhead [ms] | E tree only | E flat reduce | E tree + serial seeding | E tree + serial output | E tree + seeding + output |
|---|---|---|---|---|---|---|
{rows}
**Reading.** The log-depth tree keeps E > 0.9999 up to 10⁶ ranks. The O(P) terms are the real danger: sequential
seeding costs ≈1 s and rank-0 output ≈ 52 s at 2²⁰ ranks (E ≈ {T0/(T0+20*L+2**20*tout):.2f}). Lesson: for embarrassingly-parallel
codes, α is dominated by O(P) bookkeeping, not by the arithmetic or the reduction tree; replacing it with O(1)/O(log P)
mechanisms restores Gustafson-ideal scaling.
"""
(ROOT/"analysis"/"task19_monte_carlo.md").write_text(t19)
print("ok")
