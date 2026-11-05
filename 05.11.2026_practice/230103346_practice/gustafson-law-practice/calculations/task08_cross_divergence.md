# Task 08 — Amdahl vs. Gustafson, s = α = 0.08, P = 128

**Amdahl:** S_A = 1/(s + (1−s)/P) = 1/(0.08 + 0.92/128) = 1/(0.08 + 0.007188) = 1/0.087188 = **11.470**

**Gustafson:** S_G = P − α(P−1) = 128 − 0.08·127 = 128 − 10.16 = **117.84**

| Quantity | Value |
|---|---|
| Absolute difference S_G − S_A | 106.37 |
| Ratio S_G / S_A | 10.27× |
| Discrepancy relative to Amdahl, (S_G−S_A)/S_A | 927.4% |
| Share of S_G that Amdahl does not predict, (S_G−S_A)/S_G | 90.3% |

Divergence vs P (s = α = 0.08):

| P | S_Amdahl | S_Gustafson | ratio |
|---|---|---|---|
| 2 | 1.85 | 1.92 | 1.04 |
| 8 | 5.13 | 7.44 | 1.45 |
| 32 | 9.20 | 29.52 | 3.21 |
| 128 | 11.47 | 117.84 | 10.27 |
| 512 | 12.23 | 471.12 | 38.54 |
| 2048 | 12.43 | 1884.24 | 151.59 |
| 8192 | 12.48 | 7536.72 | 603.78 |

**Why they diverge.** Amdahl keeps the workload fixed, so adding cores only shrinks the parallel term and
S_A → 1/s = 12.5. Gustafson lets the parallel work grow with P (fixed time), so S_G grows linearly with slope 1−α = 0.92.
They coincide only at P = 1; the gap grows without bound (S_G − S_A ≈ 0.92P − 12.5). Same α, different question:
"how much faster?" vs "how much more work in equal time?".
