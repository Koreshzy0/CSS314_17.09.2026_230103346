# Task 13 — When Gustafson's Law Fails

The law needs (i) work per core to scale such that runtime stays constant, (ii) a constant serial fraction $\alpha$,
(iii) no extra overhead growth. Two concrete algorithms violate these.

## Case 1 — CFL-limited explicit PDE solver (3D wave/hydrodynamics, weak-scaled)
* Setup: grid $N^3$, memory-filling subdomains, $N\propto P^{1/3}$.
* **Complexity change:** explicit stability requires $\Delta t\le c\,\Delta x\propto 1/N$, so the number of time steps to
  reach a fixed physical time grows $\propto N$. Total work $\propto N^3\cdot N=N^4$, and per core
  $N^4/P\propto P^{1/3}$.
* Runtime $T(P)\propto P^{1/3}$ — **not constant**. Doubling the resolution costs 16× not 8×. With
  $P=512$ vs $P=1$: runtime $8^{1}\!=\!8\times$ longer for the same physical time simulated.
* The "scaled speedup" computed from time-to-physical-time collapses; Gustafson's constant-runtime premise breaks
  because *algorithmic* work per cell grows with $N$. (Remedy: implicit methods / local time stepping.)

## Case 2 — Distributed 3D FFT (spectral solvers) and global-iteration Krylov solvers
* 3D FFT: $O(N^3\log N)$ computation, with global all-to-all transposes moving $O(N^3)$ data. Per-core compute:
  $\propto \log N$ (grows slowly), but data crossing the bisection ∝ $N^3/2$ must pass a bisection bandwidth that grows
  only as $P^{2/3}$ (3D torus) or $P^{1/2}$ (2D). Communication time per step
  $\propto N^3/B_{bisect}\propto P/P^{2/3}=P^{1/3}$ — grows with $P$.
* Conjugate gradients on a Poisson problem: iteration count $\propto\sqrt{\kappa}\propto N$, and each iteration has
  two global dot-product reductions ($O(\log P)$ latency). Time $\propto N\cdot(\text{work per iteration}+\log P)$ grows
  as $P^{1/3}$ even if the work per iteration per core is constant.
* **Synchronization contention:** global reductions force all cores to wait on the slowest one (OS noise, load
  imbalance), so $\alpha$ *grows* with $P$.

## Why the constant-runtime premise breaks
| Failure mode | Equation | Effect |
|---|---|---|
| Algorithmic work superlinear in data | $T\propto N^{4}/P$ | runtime grows $P^{1/3}$ |
| Iteration count depends on $N$ | $T\propto N\cdot t_{iter}$ | same |
| Global communication volume | $T_{comm}\propto P^{1/3}$ | $\alpha(P)$ increases |
| Reductions/barriers, jitter | $T\propto\log P$, max over $P$ noise samples | extra non-scaling time |

**Evaluation.** In all cases, measured weak-scaling efficiency $E=T_1/T_P$ falls continuously and the Gustafson
prediction $E\to1-\alpha$ is too optimistic. Use the modified model of Task 12 or Sun–Ni (Task 11) instead.
