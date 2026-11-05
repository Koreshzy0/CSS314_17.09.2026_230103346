# Task 01 — Strong Scaling (Amdahl) vs. Weak Scaling (Gustafson)

## 1. The two starting points

| | Amdahl (1967) | Gustafson–Barsis (1988) |
|---|---|---|
| Question asked | "How much faster can I solve *this* problem?" | "How much *bigger* a problem can I solve in the same time?" |
| What is held fixed | Total workload $W$ | Parallel wall-clock time $T_P$ |
| What varies with $P$ | Runtime $T(P)$ shrinks | Workload $W(P)$ grows |
| Serial fraction measured | On one processor: $s = T_{ser}/T_1$ | On the parallel machine: $\alpha = T_{ser}/T_P$ |
| Speedup | $S_A(P)=\dfrac{1}{s+(1-s)/P}$ | $S_G(P)=P-\alpha(P-1)=\alpha+(1-\alpha)P$ |
| Scaling regime | Strong scaling | Weak scaling |

## 2. How the problem dimension behaves as $P\to\infty$

* **Amdahl:** the problem size $N$ is a constant, $N(P)=N_0$. The serial part $sT_1$ is constant, the parallel part
  $(1-s)T_1/P\to 0$, so $T(P)\to sT_1$ and $S_A\to 1/s$. Work per core vanishes, so overhead eventually dominates.
* **Gustafson:** parallel runtime is held constant, so *work per core* is constant and total workload grows
  linearly, $W(P)=\alpha W_0+(1-\alpha)P\,W_0$ (for an $O(N^3)$ code, $N\propto P^{1/3}$, see Task 16).
  The parallel part grows without bound while the serial part stays constant, so the serial *fraction of the
  uniprocessor work* $\alpha/(\alpha+(1-\alpha)P)\to 0$ and $S_G$ grows linearly without a ceiling.

## 3. Why "fixed-size" and "fixed-time"

* Amdahl's law is **fixed-size** because the same $W$ appears in $T_1$ and $T_P$; only $T_P$ shrinks.
  It answers a time-to-solution question.
* Gustafson's law is **fixed-time** because the normalization $s+p=1$ fixes the parallel run at one unit of time
  regardless of $P$; what scales is the amount of work finished in that unit. It answers a capability
  (resolution/accuracy) question.

## 4. Philosophical and physical divergence

1. **Who chooses the problem size?** For Amdahl the problem is given (a fixed deck to shuffle). For Gustafson the user
   has a time budget and *spends* extra processors on finer meshes, more particles, better accuracy — which is how
   supercomputers are actually used in simulation.
2. **Where is the serial part measured?** On one core (Amdahl) it is a fraction of a fixed number; on $P$ cores
   (Gustafson) it is a fraction of wall-clock time. Different frames of reference (Task 03).
3. **Verdict.** Amdahl gives an upper bound $1/s$ (pessimistic); Gustafson gives a slope $(1-\alpha)$ (optimistic).
   Both are correct algebra answering different questions; real applications sit between them (Sun–Ni, Task 11).
4. **Physical limit.** Weak scaling is ultimately bounded by memory capacity, communication growth and algorithmic
   complexity (Tasks 10–13), not by the serial fraction.
