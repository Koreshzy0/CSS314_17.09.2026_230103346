# Task 03 — The Serial-Fraction Invariance Paradox

**Memo: frame of reference for measuring overhead.**

## The apparent contradiction
Amdahl: a serial fraction $s=4\%$ caps speedup at $25\times$. Gustafson: the same $4\%$ gives $S(1024)\approx 983$.
Both cannot hold for "the same program with 4% serial code" — unless the two 4% are different quantities.

## They are different quantities
Let the scaled run on $P$ cores have serial time $s$ and parallel time $p$, $s+p=1$ (wall-clock fractions measured
*on the parallel machine*): $\alpha=s$. The same scaled workload on one core takes $W=s+pP$. The serial fraction
*as Amdahl defines it* (uniprocessor, for this workload) is

$$ f = \frac{s}{s+pP} = \frac{\alpha}{\alpha+(1-\alpha)P}. \qquad (1)$$

## Proof that they are the same law
$$ f+\frac{1-f}{P} = \frac{s}{W}+\frac{pP}{W P}=\frac{s+p}{W}=\frac1W
\;\Rightarrow\; S_A(f,P)=W=s+pP=P-\alpha(P-1). $$

Gustafson's law **is** Amdahl's law evaluated with a serial fraction $f(P)$ that shrinks as the workload grows.
Example: $\alpha=0.04,\ P=1024$: $f=0.04/(0.04+0.96\cdot1024)=4.07\times10^{-5}$.

## Why measuring on the parallel machine resolves the paradox
* A **uniprocessor** $s$ is tied to one fixed problem size; it implicitly assumes serial work scales with the parallel work.
* A **parallel-machine** $\alpha$ captures the wall-clock share of work that does **not** grow when the problem is
  expanded (setup, parameter parsing, final output, constant-size coordination). In scientific codes this is nearly
  independent of resolution, so $\alpha$ stays constant while $f\to0$.
* So serial overhead does not throttle scaling: it is an absolute constant while useful work grows $\propto P$.

## Caveat
$\alpha$ is constant only if the serial section really does not grow with $P$. If it does (collectives $O(\log P)$,
serialized I/O $O(P)$), then $\alpha=\alpha(P)$ and the invariance fails (Tasks 12, 13, 19).
