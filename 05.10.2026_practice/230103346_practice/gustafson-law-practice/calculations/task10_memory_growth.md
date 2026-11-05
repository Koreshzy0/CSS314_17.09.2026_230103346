# Task 10 — Memory Ceiling for a Weak-Scaling 3D Grid

**Model.** Aggregate footprint M(P) = c·P with c = 32 GiB (per-node cap). A cubic grid N³ with b bytes/grid point
needs M = b·N³ ≤ P·32 GiB, therefore

**N_max(P) = ( P · 32 GiB / b )^(1/3)**, i.e. N_max ∝ P^(1/3).

Weak scaling W(P) = P·W(1) means N³ ∝ P, so N(P) = N(1)·P^(1/3); memory is *exactly* saturated when each node holds
the maximum local cube n_max = (32 GiB/b)^(1/3).

Assumption (b is not given in the task): baseline b = 16 B (two double-precision arrays).

| Case | b [B/pt] | Nodes P | Total memory | Grid points | N_max |
|---|---|---|---|---|---|
| 1 fp64 array | 8 | 64 | 2048 GiB | 2.749e+11 | 6,502 |
| 1 fp64 array | 8 | 512 | 16384 GiB | 2.199e+12 | 13,004 |
| 2 fp64 arrays (Jacobi/double buffer) — **baseline** | 16 | 64 | 2048 GiB | 1.374e+11 | 5,161 |
| 2 fp64 arrays (Jacobi/double buffer) — **baseline** | 16 | 512 | 16384 GiB | 1.100e+12 | 10,321 |
| 5 fp64 arrays (e.g. u,u_new,rhs,r,p) | 40 | 64 | 2048 GiB | 5.498e+10 | 3,802 |
| 5 fp64 arrays (e.g. u,u_new,rhs,r,p) | 40 | 512 | 16384 GiB | 4.398e+11 | 7,605 |

**Baseline result:** n_max per node = 1,290³ ; **P = 64 → N ≈ 5,161** ; **P = 512 → N ≈ 10,321**.
Going 64 → 512 nodes (8×) only doubles N (8^(1/3) = 2): N(512)/N(64) = 2.000.

Consequence: the memory ceiling is a hard stop for weak scaling; beyond it one must either stop growing the problem
(→ strong scaling regime) or use out-of-core techniques. Halo/ghost cells reduce the usable N slightly.
