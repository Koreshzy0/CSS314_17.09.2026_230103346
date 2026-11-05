# Task 05 — Reconstructing the 1988 nCUBE/10 Experiment

> Numbers are reconstructed from the task statement plus recalled published results (Gustafson, Montry, Benner,
> Sandia National Labs; Gustafson, *"Reevaluating Amdahl's Law"*, CACM 31(5), 1988). Check exact published figures
> against the original papers before citing.

## Scenario
* Machine: 1,024-processor nCUBE/10 hypercube (distributed memory, small per-node memory, ~0.5 MB).
* Three real Sandia applications: beam stress analysis, surface-wave simulation, unstable fluid flow.
  Reported speedups of roughly 1,000× and above (work that won the first Gordon Bell Prize).
* Serial fraction observed on the parallel machine: ≈ 0.4–0.8%.

## Why it "contradicted" Amdahl
With $s=0.004$:
$$S_A(1024)=\frac{1}{0.004+0.996/1024}\approx 201,\qquad \text{ceiling }1/s=250.$$
Observed >1000 is 4–5× above the supposed ceiling.

## How problem expansion removes the ceiling
1. The problem was **not fixed**: each node ran a memory-filling subdomain, so the 1,024-node run solved a ≈1,024×
   larger problem (finer mesh) in about the same time.
2. The serial work (setup, I/O, bookkeeping) did not grow with mesh size, so as a fraction of the uniprocessor time
   it was $f=0.004/(0.004+0.996\cdot1024)\approx3.9\times10^{-6}$, not 0.4%.
3. Gustafson: $S(1024)=1024-0.004\cdot1023=1019.9$ — consistent with the measured ≈1000+.
4. The large problem did not fit in one node's memory, so the 1-processor time was *estimated*; this is the origin of
   the term "scaled speedup".

## Lesson
Amdahl's ceiling bounds a fixed-size problem. When resources grow, users enlarge the problem; the relevant bound is
the slope $1-\alpha$, not $1/s$.
