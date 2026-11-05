#!/usr/bin/env python3
"""Generates the numeric deliverables for Tasks 06-10 and 16-17 (markdown with computed tables).
Run:  python3 calculations/compute_all.py   (from repo root)"""
import math, pathlib
OUT = pathlib.Path(__file__).parent

def G(P, a): return P - a * (P - 1)
def A(P, s): return 1.0 / (s + (1 - s) / P)
def w(name, text): (OUT / name).write_text(text)

# ---------------- Task 06 ----------------
a = 0.04
rows = ""
for P in [16, 64, 256, 1024, 4096, 16384]:
    S = G(P, a); rows += f"| {P} | {P} − 0.04·{P-1} = {S:.2f} | {S/P:.5f} | {100*(1-S/P):.3f}% | {a*(P-1):.2f} |\n"
w("task06_parametric_table.md", f"""# Task 06 — Parametric Speedup, α = 0.04

Formulas: S(P) = P − α(P−1), E(P) = S/P = 1 − α(P−1)/P.

| P | S(P) | E(P) | Efficiency loss | Core-equivalents lost, α(P−1) |
|---|---|---|---|---|
{rows}
**Evaluation.** Each ×4 in P (exponential growth) barely changes E: it drops 0.9625 → 0.9606 → 0.9602 → … and
converges to 1 − α = 0.96. Efficiency degradation is *bounded* (never worse than 4%), whereas the absolute number of
cores lost to the serial section, α(P−1), grows linearly (0.6 cores at P=16, 655 at P=16384). Speedup stays
≈ 0.96·P: weak scaling remains efficient at any scale as long as α stays constant (an assumption broken in practice, Task 12).
""")

# ---------------- Task 07 ----------------
P, Sreq = 8192, 7500
amax = (P - Sreq) / (P - 1)
w("task07_tolerable_serial.md", f"""# Task 07 — Maximum Tolerable Serial Fraction

Requirement: S(P) = P − α(P−1) ≥ {Sreq} with P = {P}.

1. P − α(P−1) ≥ S_req
2. α(P−1) ≤ P − S_req
3. **α_max = (P − S_req)/(P − 1)** = ({P} − {Sreq})/{P-1} = {P-Sreq}/{P-1}

**α_max = {amax:.6f} ≈ {100*amax:.3f}%**

Check: S = {P} − {amax:.6f}·{P-1} = {G(P, amax):.4f} ✓ (equality).
Corresponding efficiency E = {G(P,amax)/P:.4f} (= S_req/P = 91.55%).

Interpretation: a ≈ 8.45% serial share of the parallel wall-clock time is tolerable. For comparison, Amdahl
with s = 0.0845 would cap speedup at 11.8 — the same number means something very different under weak scaling.
""")

# ---------------- Task 08 ----------------
s, P = 0.08, 128
SA, SG = A(P, s), G(P, s)
w("task08_cross_divergence.md", f"""# Task 08 — Amdahl vs. Gustafson, s = α = 0.08, P = 128

**Amdahl:** S_A = 1/(s + (1−s)/P) = 1/(0.08 + 0.92/128) = 1/(0.08 + {0.92/128:.6f}) = 1/{s+0.92/128:.6f} = **{SA:.3f}**

**Gustafson:** S_G = P − α(P−1) = 128 − 0.08·127 = 128 − {0.08*127:.2f} = **{SG:.2f}**

| Quantity | Value |
|---|---|
| Absolute difference S_G − S_A | {SG-SA:.2f} |
| Ratio S_G / S_A | {SG/SA:.2f}× |
| Discrepancy relative to Amdahl, (S_G−S_A)/S_A | {100*(SG-SA)/SA:.1f}% |
| Share of S_G that Amdahl does not predict, (S_G−S_A)/S_G | {100*(SG-SA)/SG:.1f}% |

Divergence vs P (s = α = 0.08):

| P | S_Amdahl | S_Gustafson | ratio |
|---|---|---|---|
""" + "".join(f"| {p} | {A(p,s):.2f} | {G(p,s):.2f} | {G(p,s)/A(p,s):.2f} |\n" for p in [2,8,32,128,512,2048,8192]) + f"""
**Why they diverge.** Amdahl keeps the workload fixed, so adding cores only shrinks the parallel term and
S_A → 1/s = {1/s:.1f}. Gustafson lets the parallel work grow with P (fixed time), so S_G grows linearly with slope 1−α = 0.92.
They coincide only at P = 1; the gap grows without bound (S_G − S_A ≈ 0.92P − 12.5). Same α, different question:
"how much faster?" vs "how much more work in equal time?".
""")

# ---------------- Task 09 ----------------
parts = {"A (disk I/O + broadcast)": 0.02, "B (boundary sync)": 0.06, "C (flux computation)": 0.92}
alpha = 0.02 + 0.06
body = "".join(f"| {k} | {v*100:.0f}% | {'serial' if k[0] in 'AB' else 'parallel'} |\n" for k, v in parts.items())
w("task09_multicomponent.md", f"""# Task 09 — Multi-Component Workload

| Phase | Share of parallel runtime | Treated as |
|---|---|---|
{body}
Assumption (state it explicitly): phases A and B execute with no speedup from the added cores (I/O and
synchronization are modelled as serial / non-scaling), phase C is perfectly parallel.

**α = 0.02 + 0.06 = {alpha:.2f}**, p = 0.92.

| P | S(P) = P − 0.08(P−1) | E(P) |
|---|---|---|
| 512 | 512 − 0.08·511 = **{G(512,alpha):.2f}** | {G(512,alpha)/512:.4f} |
| 2048 | 2048 − 0.08·2047 = **{G(2048,alpha):.2f}** | {G(2048,alpha)/2048:.4f} |

Composite view: the 8% non-scaling share costs about 8% of the machine at any P (E → 0.92). Phase B (6%)
is the larger target; halving it raises the limit E from 0.92 to 0.95. Caveat: boundary synchronization normally grows with P
(Task 12), in which case B should be modelled as B(P) and α would increase with P.
""")

# ---------------- Task 10 ----------------
GiB = 2**30
def Nmax(P, bytes_per_pt, mem=32*GiB): return (P * mem / bytes_per_pt) ** (1/3)
rows = ""
for bpp, label in [(8, "1 fp64 array"), (16, "2 fp64 arrays (Jacobi/double buffer) — **baseline**"), (40, "5 fp64 arrays (e.g. u,u_new,rhs,r,p)")]:
    for P in (64, 512):
        rows += f"| {label} | {bpp} | {P} | {P*32} GiB | {P*32*GiB/bpp:.3e} | {Nmax(P,bpp):,.0f} |\n"
w("task10_memory_growth.md", f"""# Task 10 — Memory Ceiling for a Weak-Scaling 3D Grid

**Model.** Aggregate footprint M(P) = c·P with c = 32 GiB (per-node cap). A cubic grid N³ with b bytes/grid point
needs M = b·N³ ≤ P·32 GiB, therefore

**N_max(P) = ( P · 32 GiB / b )^(1/3)**, i.e. N_max ∝ P^(1/3).

Weak scaling W(P) = P·W(1) means N³ ∝ P, so N(P) = N(1)·P^(1/3); memory is *exactly* saturated when each node holds
the maximum local cube n_max = (32 GiB/b)^(1/3).

Assumption (b is not given in the task): baseline b = 16 B (two double-precision arrays).

| Case | b [B/pt] | Nodes P | Total memory | Grid points | N_max |
|---|---|---|---|---|---|
{rows}
**Baseline result:** n_max per node = {Nmax(1,16):,.0f}³ ; **P = 64 → N ≈ {Nmax(64,16):,.0f}** ; **P = 512 → N ≈ {Nmax(512,16):,.0f}**.
Going 64 → 512 nodes (8×) only doubles N (8^(1/3) = 2): N(512)/N(64) = {Nmax(512,16)/Nmax(64,16):.3f}.

Consequence: the memory ceiling is a hard stop for weak scaling; beyond it one must either stop growing the problem
(→ strong scaling regime) or use out-of-core techniques. Halo/ghost cells reduce the usable N slightly.
""")

# ---------------- Task 16 ----------------
N0 = 1000
rows = "".join(f"| {P} | {N0*P**(1/3):,.0f} | {P**(1/3):.3f} | {(N0*P**(1/3))**2/P/1e6*8:.2f} MB |\n" for P in [1,8,64,512,4096,32768])
w("task16_gemm_scaling.md", f"""# Task 16 — GEMM Scaling Under Constant Runtime

**Derivation.** Work W(N) = 2N³ flops. With P cores at rate r each and ideal parallel execution, T = 2N³/(P r).
Gustafson fixes T = T₀ = 2N₀³/r (the single-core time of the baseline problem N₀). Hence

N³/P = N₀³ ⇒ **N(P) = N₀ · P^(1/3)**.

Memory: total = 3·8·N² bytes ∝ P^(2/3); **per core ∝ N²/P = N₀²·P^(−1/3)** — it *decreases*, so GEMM is
compute-bound and memory is not the limiter (memory-bounded Sun–Ni scaling would allow N ∝ P^(1/2), with runtime ∝ P^(1/2)).

Baseline N₀ = {N0} (matrix A only, fp64, per-core share of one matrix):

| P | N(P) | N/N₀ | A-matrix per core |
|---|---|---|---|
{rows}
Practical caveat: a distributed GEMM must also move data. With 2D SUMMA the communication volume per core ∝ N²/√P =
N₀²P^(2/3)/P^(1/2) = N₀²P^(1/6), so communication grows slowly relative to computation per core — weak scaling holds but
isn't perfect, i.e. α(P) creeps upward.
""")

# ---------------- Task 17 ----------------
n = 128
rows = ""
for P in [1, 8, 64, 512, 4096]:
    Nglob = n * round(P ** (1/3)); rows += f"| {P} | {Nglob} | {n}³ = {n**3:,} | 6·{n}² = {6*n*n:,} | {6*n*n/n**3:.4f} |\n"
rows2 = "".join(f"| {P} | {N0*1} | {6*(1024/P**(1/3))**2/(1024/P**(1/3))**3:.4f} |\n" for P in [1, 8, 64, 512, 4096])
w("task17_stencil_scaling.md", f"""# Task 17 — 3D 7-Point Stencil: Surface-to-Volume Under Weak Scaling

**Decomposition.** Cubic domain N³ split over P ranks in a ∛P×∛P×∛P grid; each rank owns a sub-cube of side n = N/∛P.

* Computation per step ∝ n³ (interior updates, ~8 flops/point for 7-point stencil).
* Communication per step ∝ 6n² (six faces, halo depth 1) — O(N²) at the global level, but only the local surface matters.
* Communication-to-computation ratio **ρ = 6n²/n³ = 6/n**.

**Weak scaling (Gustafson).** Work per rank constant ⇒ n = n₀ constant ⇒ N = n₀·P^(1/3) and

ρ_weak = 6/n₀ = const **independent of P**.

| P | Global N | Local volume | Local surface | ρ = surface/volume |
|---|---|---|---|---|
{rows}
**Strong scaling for contrast** (global N fixed = 1024): n = 1024/P^(1/3), so ρ_strong = 6P^(1/3)/1024 **grows** as P^(1/3):

| P | n | ρ_strong |
|---|---|---|
""" + "".join(f"| {P} | {1024/P**(1/3):.1f} | {6/(1024/P**(1/3)):.4f} |\n" for P in [1, 8, 64, 512, 4096]) + """
**Conclusion.** Under weak scaling the ratio neither improves nor degrades — it is *constant*, which is why stencil
codes are the textbook Gustafson-friendly workload (the nCUBE applications were of this type). Two remaining
degradations: (i) latency-bound neighbor exchange and global reductions (e.g. residual norms, O(log P)) add α(P);
(ii) if the global time step must shrink with N (CFL), constant runtime fails (Task 13).
""")
print("ok")
