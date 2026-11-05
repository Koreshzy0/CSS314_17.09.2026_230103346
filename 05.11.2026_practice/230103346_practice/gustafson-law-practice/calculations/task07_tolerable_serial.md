# Task 07 — Maximum Tolerable Serial Fraction

Requirement: S(P) = P − α(P−1) ≥ 7500 with P = 8192.

1. P − α(P−1) ≥ S_req
2. α(P−1) ≤ P − S_req
3. **α_max = (P − S_req)/(P − 1)** = (8192 − 7500)/8191 = 692/8191

**α_max = 0.084483 ≈ 8.448%**

Check: S = 8192 − 0.084483·8191 = 7500.0000 ✓ (equality).
Corresponding efficiency E = 0.9155 (= S_req/P = 91.55%).

Interpretation: a ≈ 8.45% serial share of the parallel wall-clock time is tolerable. For comparison, Amdahl
with s = 0.0845 would cap speedup at 11.8 — the same number means something very different under weak scaling.
