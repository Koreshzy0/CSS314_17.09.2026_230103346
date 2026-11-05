# Task 02 — Derivation of Gustafson's Scaled Speedup

**Setup.** Run the scaled problem on $P$ identical cores. Normalize the *parallel* wall-clock time to one unit:

$$ s + p = 1 \qquad (1)$$

$s$ = time in the serial section (one core busy, $P-1$ idle), $p$ = time in the parallel section (all $P$ cores
busy). Let $\alpha \equiv s$ be the serial fraction of the parallel run, so $p = 1-\alpha$.

**Step 1 — Work done by the parallel run, in single-core units.** The serial section is executed by one core:
$s$ units of single-core work. The parallel section keeps $P$ cores busy for $p$ units, i.e. $P\,p$ units of
single-core work (perfect parallelization inside that section, no overhead).

**Step 2 — Time for one core to do the same scaled workload.**

$$ T_1 = s + pP \qquad (2)$$

**Step 3 — Speedup = single-core time / parallel time for the same scaled workload.**

$$ S(P) = \frac{T_1}{T_P} = \frac{s+pP}{s+p} = s + pP \qquad (3)$$

using (1) in the denominator.

**Step 4 — Eliminate $p$.** With $p = 1-\alpha$: $\;S(P) = \alpha + (1-\alpha)P.$

**Step 5 — Rearrange.** $S(P) = \alpha + P - \alpha P = P - \alpha(P-1)$. $\blacksquare$

**Checks.**

| Check | Result |
|---|---|
| $P=1$ | $S=1$ ✓ |
| $\alpha=0$ | $S=P$ (ideal) ✓ |
| $\alpha=1$ | $S=P-(P-1)=1$ ✓ |
| Slope | $dS/dP = 1-\alpha$, constant, no ceiling |

**Efficiency.** $E(P)=S(P)/P = 1-\alpha\dfrac{P-1}{P}$.
