# Task 06 — Parametric Speedup, α = 0.04

Formulas: S(P) = P − α(P−1), E(P) = S/P = 1 − α(P−1)/P.

| P | S(P) | E(P) | Efficiency loss | Core-equivalents lost, α(P−1) |
|---|---|---|---|---|
| 16 | 16 − 0.04·15 = 15.40 | 0.96250 | 3.750% | 0.60 |
| 64 | 64 − 0.04·63 = 61.48 | 0.96062 | 3.938% | 2.52 |
| 256 | 256 − 0.04·255 = 245.80 | 0.96016 | 3.984% | 10.20 |
| 1024 | 1024 − 0.04·1023 = 983.08 | 0.96004 | 3.996% | 40.92 |
| 4096 | 4096 − 0.04·4095 = 3932.20 | 0.96001 | 3.999% | 163.80 |
| 16384 | 16384 − 0.04·16383 = 15728.68 | 0.96000 | 4.000% | 655.32 |

**Evaluation.** Each ×4 in P (exponential growth) barely changes E: it drops 0.9625 → 0.9606 → 0.9602 → … and
converges to 1 − α = 0.96. Efficiency degradation is *bounded* (never worse than 4%), whereas the absolute number of
cores lost to the serial section, α(P−1), grows linearly (0.6 cores at P=16, 655 at P=16384). Speedup stays
≈ 0.96·P: weak scaling remains efficient at any scale as long as α stays constant (an assumption broken in practice, Task 12).
